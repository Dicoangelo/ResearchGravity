#!/usr/bin/env python3
"""
Checkpoint-resumable YouTube channel scraper (v2).

Fixes the two flaws in v1:
  1. v1 only wrote output at the very end — an interruption lost everything.
     v2 appends every enriched record to meta.jsonl immediately.
  2. v1 fetched metadata serially (332 videos ~30 min).
     v2 uses a worker pool and skips ids already in meta.jsonl.

Phases:
  A  flat playlist  -> index.json   (1 request, instant, gives id/title/duration/order)
  B  per-video -J   -> meta.jsonl   (parallel, incremental, resumable)
  C  assemble       -> full.json / urls.txt / videos.txt

Usage:
  python3 scripts/importers/youtube_channel_keyless.py rubentech999 \
      [--workers 8] [--limit N] [--phase A|B|C|all]
"""
import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

BASE = Path.home() / ".agent-core" / "research" / "youtube"
YTDLP = ["yt-dlp", "--ignore-config", "--no-warnings"]
_lock = threading.Lock()


def fmt_duration(sec):
    if not sec:
        return ""
    s = int(sec)
    h, rem = divmod(s, 3600)
    m, ss = divmod(rem, 60)
    return f"{h}h {m}m {ss}s" if h else f"{m}m {ss}s"


def fmt_date(ud):
    ud = str(ud or "")
    return f"{ud[0:4]}-{ud[4:6]}-{ud[6:8]}" if len(ud) == 8 else ""


# ---------- Phase A ----------
def phase_a(handle, outdir, limit=None):
    url = f"https://www.youtube.com/@{handle}/videos"
    cmd = YTDLP + ["--flat-playlist", "-J"]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    cmd += [url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR flat playlist: {r.stderr[:400]}", file=sys.stderr)
        sys.exit(1)
    d = json.loads(r.stdout)
    entries = [e for e in d.get("entries", []) if e.get("id")]
    index = {
        "channel": {
            "id": d.get("channel_id", ""),
            "handle": handle,
            "name": d.get("channel", ""),
            "subscribers": d.get("channel_follower_count"),
            "description": d.get("description", ""),
            "tags": d.get("tags", []),
            "indexed_at": datetime.now().isoformat(),
        },
        # position 0 == newest (channel /videos tab order)
        "videos": [
            {"position": i, "id": e["id"], "title": e.get("title", ""),
             "duration_seconds": e.get("duration") or 0,
             "url": f"https://youtube.com/watch?v={e['id']}"}
            for i, e in enumerate(entries)
        ],
        "total": len(entries),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[A] index.json written — {len(entries)} videos", flush=True)
    return index


# ---------- Phase B ----------
def load_done(jsonl):
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except json.JSONDecodeError:
                continue
    return done


def fetch_one(vid):
    r = subprocess.run(
        YTDLP + ["-J", "--skip-download", f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        v = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return {
        "id": vid,
        "title": v.get("title", ""),
        "published": fmt_date(v.get("upload_date")),
        "url": f"https://youtube.com/watch?v={vid}",
        "duration": fmt_duration(v.get("duration")),
        "duration_seconds": v.get("duration") or 0,
        "views": v.get("view_count") or 0,
        "likes": v.get("like_count") or 0,
        "comments": v.get("comment_count") or 0,
        "categories": v.get("categories") or [],
        "tags": (v.get("tags") or [])[:25],
        "description": v.get("description", "") or "",
    }


def phase_b(outdir, index, workers=8, limit=None):
    jsonl = outdir / "meta.jsonl"
    done = load_done(jsonl)
    todo = [v["id"] for v in index["videos"] if v["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[B] have={len(done)} todo={len(todo)} workers={workers}", flush=True)
    if not todo:
        return

    fh = jsonl.open("a", encoding="utf-8")
    counter = {"n": 0, "fail": 0}

    def work(vid):
        rec = fetch_one(vid)
        with _lock:
            if rec:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()          # checkpoint on every single record
                counter["n"] += 1
            else:
                counter["fail"] += 1
            tot = counter["n"] + counter["fail"]
            if tot % 20 == 0:
                print(f"[B] {tot}/{len(todo)} ok={counter['n']} fail={counter['fail']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"[B] complete ok={counter['n']} fail={counter['fail']}", flush=True)


# ---------- Phase C ----------
def phase_c(outdir, index):
    done = load_done(outdir / "meta.jsonl")
    pos = {v["id"]: v["position"] for v in index["videos"]}
    videos = sorted(done.values(), key=lambda r: pos.get(r["id"], 10**9))

    channel = dict(index["channel"])
    channel["scraped_at"] = datetime.now().isoformat()

    (outdir / "urls.txt").write_text(
        "\n".join(v["url"] for v in videos) + "\n", encoding="utf-8")

    lines = [f"{channel['name']} — {len(videos)} videos", "=" * 60, ""]
    for v in videos:
        lines.append(f"{v['published']} | {v['url']}")
        lines.append(v["title"])
        lines.append("")
    (outdir / "videos.txt").write_text("\n".join(lines), encoding="utf-8")

    (outdir / "full.json").write_text(
        json.dumps({"channel": channel, "videos": videos, "total": len(videos)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[C] wrote full.json / urls.txt / videos.txt — {len(videos)} videos",
          flush=True)

    # register in channels.json
    reg = BASE / "channels.json"
    data = json.loads(reg.read_text()) if reg.exists() else {"channels": []}
    chans = [c for c in data.get("channels", []) if c.get("handle") != channel["handle"]]
    chans.append({"handle": channel["handle"], "name": channel["name"],
                  "id": channel["id"], "video_count": len(videos),
                  "last_scraped": datetime.now().isoformat()})
    data["channels"] = chans
    reg.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[C] registered in channels.json", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("handle")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--phase", default="all", choices=["A", "B", "C", "all"])
    a = p.parse_args()

    handle = a.handle.lstrip("@")
    outdir = BASE / handle
    idx_path = outdir / "index.json"

    if a.phase in ("A", "all") or not idx_path.exists():
        index = phase_a(handle, outdir, a.limit)
    else:
        index = json.loads(idx_path.read_text())

    if a.phase in ("B", "all"):
        phase_b(outdir, index, a.workers, a.limit)
    if a.phase in ("C", "all"):
        phase_c(outdir, index)


if __name__ == "__main__":
    main()
