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
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

BASE = Path.home() / ".agent-core" / "research" / "youtube"
YTDLP = ["yt-dlp", "--ignore-config", "--no-warnings"]
_lock = threading.Lock()

# Escalating backoff, matching the bar set by the API-based importer
# (scripts/importers/youtube_channel.py). Eight workers with no throttle
# handling is exactly what earns a 429.
BACKOFF = [30, 60, 120, 240, 300]

# Transient — worth retrying. Substring match against yt-dlp's stderr.
TRANSIENT = (
    "http error 429",
    "http error 403",
    "too many requests",
    "sign in to confirm",
    "temporary failure",
    "connection reset",
    "read timed out",
    "unable to download webpage",
    "the read operation timed out",
)

# Permanent — the video is genuinely gone or gated. Retrying wastes the budget
# and, worse, makes a throttled run look like a run full of dead videos.
PERMANENT = (
    "video unavailable",
    "private video",
    "has been removed",
    "members-only",
    "this live event has ended",
    "account associated with this video has been terminated",
)


def classify(stderr: str) -> str:
    """transient | permanent | unknown — decides whether a retry is worth it."""
    s = (stderr or "").lower()
    if any(p in s for p in PERMANENT):
        return "permanent"
    if any(t in s for t in TRANSIENT):
        return "transient"
    return "unknown"


def run_ytdlp(args, what, attempts=None):
    """
    Run yt-dlp, retrying transient failures with escalating backoff.

    Returns (stdout, None) on success, (None, reason) on give-up. The reason is
    load-bearing: the caller must be able to tell "this video is private" from
    "we got throttled", because those mean opposite things about whether the
    resulting dataset is complete.
    """
    # Late-bound: one initial try plus one per backoff step. Computed here
    # rather than as a default argument so it tracks BACKOFF if that is tuned.
    if attempts is None:
        attempts = len(BACKOFF) + 1

    last = "unknown"
    for attempt in range(attempts):
        r = subprocess.run(YTDLP + args, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout, None

        kind = classify(r.stderr)
        last = f"{kind}: {(r.stderr or '').strip()[:200]}"
        if kind == "permanent":
            return None, last
        if attempt == attempts - 1:
            break

        # Jitter so parallel workers do not resynchronise into a second storm.
        delay = BACKOFF[min(attempt, len(BACKOFF) - 1)] * (0.75 + random.random() * 0.5)
        with _lock:
            print(
                f"[retry] {what}: {kind}, sleeping {delay:.0f}s "
                f"(attempt {attempt + 1}/{attempts})",
                file=sys.stderr, flush=True,
            )
        time.sleep(delay)

    return None, last


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
    cmd = ["--flat-playlist", "-J"]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    cmd += [url]
    out, err = run_ytdlp(cmd, f"flat playlist @{handle}")
    if out is None:
        # Exiting here is correct — without the index there is nothing to
        # resume from, so a partial run is not possible. But only after the
        # backoff ladder has been exhausted, not on the first 403.
        print(f"ERROR flat playlist: {err}", file=sys.stderr)
        sys.exit(1)
    d = json.loads(out)
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
    """Returns (record, None) or (None, reason)."""
    out, err = run_ytdlp(
        ["-J", "--skip-download", f"https://www.youtube.com/watch?v={vid}"],
        f"video {vid}",
    )
    if out is None:
        return None, err
    try:
        v = json.loads(out)
    except json.JSONDecodeError:
        return None, "unknown: malformed JSON from yt-dlp"
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
    }, None


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
    # Which ids failed, and why. Without this, a throttled run and a run over a
    # channel full of private videos produce the same output — an incomplete
    # meta.jsonl that phase C assembles into a dataset looking complete.
    failures = {}
    counter = {"n": 0}

    def work(vid):
        rec, reason = fetch_one(vid)
        with _lock:
            if rec:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()          # checkpoint on every single record
            else:
                failures[vid] = reason
            counter["n"] += 1
            if counter["n"] % 20 == 0:
                print(
                    f"[B] {counter['n']}/{len(todo)} "
                    f"ok={counter['n'] - len(failures)} fail={len(failures)}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()

    ok = len(todo) - len(failures)
    transient = {v: r for v, r in failures.items() if r and r.startswith("transient")}
    unknown = {v: r for v, r in failures.items() if r and r.startswith("unknown")}
    permanent = {v: r for v, r in failures.items() if r and r.startswith("permanent")}

    if failures:
        (outdir / "failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[B] complete ok={ok} failed={len(failures)} "
        f"(permanent={len(permanent)} transient={len(transient)} "
        f"unknown={len(unknown)})",
        flush=True,
    )

    # Transient failures survived the full backoff ladder, which means the
    # dataset is incomplete for a reason that will not fix itself by assembling
    # it. Say so loudly and exit non-zero — a partial dataset that reports
    # success is the failure mode this whole scraper is trying to avoid.
    if transient or unknown:
        print(
            f"[B] INCOMPLETE — {len(transient) + len(unknown)} video(s) failed for "
            f"non-permanent reasons after {len(BACKOFF)} retries. Ids and reasons in "
            f"{outdir / 'failures.json'}. Re-run to resume; already-fetched videos "
            f"are skipped.",
            file=sys.stderr, flush=True,
        )
        return False
    return True


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

    complete = True
    if a.phase in ("B", "all"):
        complete = phase_b(outdir, index, a.workers, a.limit)

    # Phase C still runs on an incomplete fetch — a partial dataset is useful,
    # and re-running resumes. But the exit code must not claim success, or a
    # throttled run looks identical to a clean one to anything downstream.
    if a.phase in ("C", "all"):
        phase_c(outdir, index)

    if not complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
