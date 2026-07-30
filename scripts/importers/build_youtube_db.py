#!/usr/bin/env python3
"""Build the canonical YouTube research DB from per-channel full.json exports.

Single source of truth: ~/.agent-core/research/youtube/youtube.db
Raw scrape exports (full.json) are inputs; this is idempotent and rebuildable.
NotebookLM bridge: channels.notebook_id + videos.notebook_source_id.

Usage:
  python3 build_youtube_db.py                    # ingest all channels
  python3 build_youtube_db.py --channel MilkRoadAI
  python3 build_youtube_db.py --set-notebook MilkRoadAI <notebook-uuid>
"""
import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / ".agent-core" / "research" / "youtube"
DB = ROOT / "youtube.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
  handle        TEXT PRIMARY KEY,
  name          TEXT,
  channel_id    TEXT,
  video_count   INTEGER,
  last_scraped  TEXT,
  notebook_id   TEXT,
  ingested_at   TEXT
);
CREATE TABLE IF NOT EXISTS videos (
  id                 TEXT PRIMARY KEY,
  channel_handle     TEXT NOT NULL REFERENCES channels(handle),
  title              TEXT,
  published          TEXT,
  url                TEXT,
  duration           TEXT,
  duration_seconds   INTEGER,
  views              INTEGER,
  likes              INTEGER,
  comments           INTEGER,
  description        TEXT,
  tags               TEXT,          -- JSON array; scraper doesn't emit yet, nullable
  notebook_source_id TEXT,          -- NotebookLM bridge
  ingested_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_handle);
CREATE INDEX IF NOT EXISTS idx_videos_published ON videos(published);
CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5(
  title, description, content=videos, content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS videos_ai AFTER INSERT ON videos BEGIN
  INSERT INTO videos_fts(rowid, title, description)
  VALUES (new.rowid, new.title, new.description);
END;
CREATE TRIGGER IF NOT EXISTS videos_au AFTER UPDATE ON videos BEGIN
  INSERT INTO videos_fts(videos_fts, rowid, title, description)
  VALUES('delete', old.rowid, old.title, old.description);
  INSERT INTO videos_fts(rowid, title, description)
  VALUES (new.rowid, new.title, new.description);
END;
CREATE TRIGGER IF NOT EXISTS videos_ad AFTER DELETE ON videos BEGIN
  INSERT INTO videos_fts(videos_fts, rowid, title, description)
  VALUES('delete', old.rowid, old.title, old.description);
END;
"""


def duration_to_seconds(d: str) -> int | None:
    if not d:
        return None
    total, m = 0, re.findall(r"(\d+)\s*([hms])", d)
    for val, unit in m:
        total += int(val) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total or None


def ingest_channel(con: sqlite3.Connection, handle: str) -> int:
    fj = ROOT / handle / "full.json"
    if not fj.exists():
        print(f"  skip {handle}: no full.json")
        return 0
    data = json.loads(fj.read_text())
    vids = data.get("videos", data) if isinstance(data, dict) else data
    ch = data.get("channel", {}) if isinstance(data, dict) else {}
    now = datetime.now(timezone.utc).isoformat()

    # ON CONFLICT with every column named — never INSERT OR REPLACE (zeros unpassed cols)
    con.execute(
        """INSERT INTO channels(handle, name, channel_id, video_count, last_scraped, ingested_at)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(handle) DO UPDATE SET
             name=excluded.name, channel_id=excluded.channel_id,
             video_count=excluded.video_count, last_scraped=excluded.last_scraped,
             ingested_at=excluded.ingested_at""",
        (handle, ch.get("name"), ch.get("id"), len(vids), ch.get("scraped_at"), now),
    )
    for v in vids:
        con.execute(
            """INSERT INTO videos(id, channel_handle, title, published, url, duration,
                                  duration_seconds, views, likes, comments, description,
                                  tags, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 channel_handle=excluded.channel_handle, title=excluded.title,
                 published=excluded.published, url=excluded.url,
                 duration=excluded.duration, duration_seconds=excluded.duration_seconds,
                 views=excluded.views, likes=excluded.likes, comments=excluded.comments,
                 description=excluded.description,
                 tags=COALESCE(excluded.tags, videos.tags),
                 ingested_at=excluded.ingested_at""",
            (
                v.get("id"), handle, v.get("title"), v.get("published"), v.get("url"),
                v.get("duration"), duration_to_seconds(v.get("duration", "")),
                v.get("views"), v.get("likes"), v.get("comments"), v.get("description"),
                json.dumps(v["tags"]) if v.get("tags") else None, now,
            ),
        )
    print(f"  {handle}: {len(vids)} videos")
    return len(vids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel")
    ap.add_argument("--set-notebook", nargs=2, metavar=("HANDLE", "NOTEBOOK_ID"))
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    if args.set_notebook:
        handle, nb = args.set_notebook
        con.execute("UPDATE channels SET notebook_id=? WHERE handle=?", (nb, handle))
        con.commit()
        print(f"channels.{handle}.notebook_id = {nb}")
        return

    handles = [args.channel] if args.channel else sorted(
        p.name for p in ROOT.iterdir() if p.is_dir() and (p / "full.json").exists()
    )
    total = sum(ingest_channel(con, h) for h in handles)
    con.commit()
    n_ch = con.execute("SELECT COUNT(*) FROM channels").fetchone()[0]
    n_v = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"\nDB: {DB}\nchannels={n_ch} videos={n_v} (+{total} this run)")


if __name__ == "__main__":
    main()
