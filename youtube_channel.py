#!/usr/bin/env python3
"""YouTube Channel Scraper — ResearchGravity Integration

Scrapes all videos from a YouTube channel and outputs in multiple formats
for research tracking and analysis.

Usage:
  python3 youtube_channel.py https://youtube.com/@ChannelName
  python3 youtube_channel.py @ChannelName --tier 1 --category labs
  python3 youtube_channel.py @ChannelName --limit 50 --output-dir ~/custom
  python3 youtube_channel.py @ChannelName --transcripts --transcript-lang en

Environment:
  YOUTUBE_API_KEY    Required. Get from Google Cloud Console.
                     https://console.cloud.google.com/apis/credentials

Output Files:
  urls.txt               - One URL per line (for batch processing)
  videos.txt             - Date + URL + Title (human readable)
  full.json              - Complete metadata (machine readable)
  transcripts/           - Plain text transcripts per video (with --transcripts)
  transcripts_index.json - Transcript status index (with --transcripts)
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_agent_core_dir() -> Path:
    """Global research storage directory."""
    return Path.home() / ".agent-core"


def get_youtube_dir() -> Path:
    """YouTube data storage directory."""
    youtube_dir = get_agent_core_dir() / "research" / "youtube"
    youtube_dir.mkdir(parents=True, exist_ok=True)
    return youtube_dir


def get_api_key() -> str:
    """Get YouTube API key from environment or config."""
    # 1. Check environment variable
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if api_key:
        return api_key

    # 2. Check config file
    config_file = get_agent_core_dir() / "config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            api_key = config.get("youtube", {}).get("api_key")
            if api_key:
                return api_key
        except (json.JSONDecodeError, KeyError):
            pass

    # 3. No key found
    print("ERROR: YouTube API key not found.", file=sys.stderr)
    print("\nSet it via environment variable:", file=sys.stderr)
    print("  export YOUTUBE_API_KEY='your-key-here'", file=sys.stderr)
    print("\nOr add to ~/.agent-core/config.json:", file=sys.stderr)
    print('  {"youtube": {"api_key": "your-key-here"}}', file=sys.stderr)
    print("\nGet a key at: https://console.cloud.google.com/apis/credentials", file=sys.stderr)
    sys.exit(1)


def parse_channel_input(channel_input: str) -> str:
    """Extract channel handle from various input formats."""
    # Remove whitespace
    channel_input = channel_input.strip()

    # Handle full URLs
    # https://youtube.com/@ChannelName or https://www.youtube.com/@ChannelName/videos
    url_match = re.search(r'youtube\.com/@([^/?\s]+)', channel_input)
    if url_match:
        return url_match.group(1)

    # Handle @ChannelName format
    if channel_input.startswith('@'):
        return channel_input[1:]

    # Assume it's just the handle
    return channel_input


def resolve_channel(handle: str, api_key: str) -> dict:
    """Resolve channel handle to channel ID and uploads playlist."""
    url = f"https://www.googleapis.com/youtube/v3/channels?part=contentDetails,snippet&forHandle={handle}&key={api_key}"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("ERROR: API quota exceeded or invalid key.", file=sys.stderr)
        elif e.code == 404:
            print(f"ERROR: Channel @{handle} not found.", file=sys.stderr)
        else:
            print(f"ERROR: API request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not data.get("items"):
        print(f"ERROR: Channel @{handle} not found.", file=sys.stderr)
        sys.exit(1)

    item = data["items"][0]
    return {
        "id": item["id"],
        "handle": handle,
        "name": item["snippet"]["title"],
        "description": item["snippet"].get("description", ""),
        "uploads_playlist": item["contentDetails"]["relatedPlaylists"]["uploads"]
    }


def fetch_all_videos(playlist_id: str, api_key: str, limit: Optional[int] = None) -> list[dict]:
    """Fetch all videos from uploads playlist."""
    videos = []
    next_page = None
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"

    while True:
        url = f"{base_url}?part=snippet&playlistId={playlist_id}&maxResults=50&key={api_key}"
        if next_page:
            url += f"&pageToken={next_page}"

        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            print(f"ERROR: Failed to fetch playlist: {e}", file=sys.stderr)
            break

        for item in data.get("items", []):
            video_id = item["snippet"]["resourceId"]["videoId"]
            videos.append({
                "id": video_id,
                "title": item["snippet"]["title"],
                "published": item["snippet"]["publishedAt"][:10],
                "url": f"https://youtube.com/watch?v={video_id}"
            })

            if limit and len(videos) >= limit:
                return videos

        next_page = data.get("nextPageToken")
        if not next_page:
            break

        # Progress indicator
        print(f"  Fetched {len(videos)} videos...", file=sys.stderr)

    return videos


def fetch_video_metadata(video_ids: list[str], api_key: str) -> dict[str, dict]:
    """Fetch detailed metadata for videos in batches of 50."""
    metadata = {}

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        ids_str = ",".join(batch)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={ids_str}&key={api_key}"

        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            print(f"WARNING: Failed to fetch metadata batch: {e}", file=sys.stderr)
            continue

        for item in data.get("items", []):
            vid = item["id"]
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})
            snippet = item.get("snippet", {})

            # Parse duration (ISO 8601 to readable)
            duration = details.get("duration", "")
            duration_str = parse_duration(duration)

            metadata[vid] = {
                "duration": duration_str,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "description": snippet.get("description", "")[:500]  # First 500 chars
            }

        print(f"  Metadata: {min(i+50, len(video_ids))}/{len(video_ids)}", file=sys.stderr)

    return metadata


def parse_duration(iso_duration: str) -> str:
    """Convert ISO 8601 duration to human readable format."""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
    if not match:
        return iso_duration

    h, m, s = match.groups(default='0')
    h, m, s = int(h), int(m), int(s)

    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def write_outputs(channel: dict, videos: list[dict], output_dir: Path, urls_only: bool = False) -> None:
    """Write output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. urls.txt - always write
    urls_file = output_dir / "urls.txt"
    urls_file.write_text("\n".join(v["url"] for v in videos))
    print(f"  Written: {urls_file}")

    if urls_only:
        return

    # 2. videos.txt - date + url + title
    videos_file = output_dir / "videos.txt"
    lines = [f"{channel['name']} — {len(videos)} videos", "=" * 60, ""]
    for v in videos:
        lines.append(f"{v['published']} | {v['url']}")
        lines.append(v["title"])
        lines.append("")
    videos_file.write_text("\n".join(lines))
    print(f"  Written: {videos_file}")

    # 3. full.json - complete metadata
    full_file = output_dir / "full.json"
    output = {
        "channel": {
            "id": channel["id"],
            "handle": channel["handle"],
            "name": channel["name"],
            "scraped_at": datetime.now().isoformat()
        },
        "videos": videos,
        "total": len(videos)
    }
    full_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Written: {full_file}")


def update_channels_registry(channel: dict, video_count: int) -> None:
    """Update the channels registry with this scrape."""
    registry_file = get_youtube_dir() / "channels.json"

    if registry_file.exists():
        registry = json.loads(registry_file.read_text())
    else:
        registry = {"channels": []}

    # Update or add channel entry
    existing = next((c for c in registry["channels"] if c["handle"] == channel["handle"]), None)
    entry = {
        "handle": channel["handle"],
        "name": channel["name"],
        "id": channel["id"],
        "video_count": video_count,
        "last_scraped": datetime.now().isoformat()
    }

    if existing:
        existing.update(entry)
    else:
        registry["channels"].append(entry)

    registry_file.write_text(json.dumps(registry, indent=2))


def fetch_single_transcript(ytt_api, video_id: str, lang: str) -> dict:
    """Fetch transcript for a single video with language priority.

    Priority: manual in preferred lang > generated in preferred lang > any available.
    Returns dict with keys: text, language, type (manual/generated), word_count, status.
    """
    try:
        transcript_list = ytt_api.list(video_id)
    except Exception:
        return {"status": "no_captions"}

    # Try manual transcript in preferred language first
    transcript = None
    t_type = None
    t_lang = None
    try:
        transcript = transcript_list.find_manually_created_transcript([lang])
        t_type = "manual"
        t_lang = lang
    except Exception:
        pass

    # Try generated transcript in preferred language
    if transcript is None:
        try:
            transcript = transcript_list.find_generated_transcript([lang])
            t_type = "generated"
            t_lang = lang
        except Exception:
            pass

    # Fall back to any available transcript
    if transcript is None:
        try:
            for t in transcript_list:
                transcript = t
                t_type = "manual" if not t.is_generated else "generated"
                t_lang = t.language_code
                break
        except Exception:
            return {"status": "no_captions"}

    if transcript is None:
        return {"status": "no_captions"}

    try:
        entries = transcript.fetch()
        text = " ".join(snippet.text for snippet in entries).strip()
    except Exception:
        return {"status": "fetch_error"}

    word_count = len(text.split())
    return {
        "status": "ok",
        "text": text,
        "language": t_lang,
        "type": t_type,
        "word_count": word_count,
    }


def fetch_transcripts(videos: list[dict], output_dir: Path, lang: str = "en",
                      delay: float = 1.5) -> dict:
    """Fetch transcripts for all videos with rate limiting and resume support.

    Returns index dict mapping video_id -> transcript metadata.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("ERROR: youtube-transcript-api not installed.", file=sys.stderr)
        print("  pip install youtube-transcript-api", file=sys.stderr)
        sys.exit(1)

    ytt_api = YouTubeTranscriptApi()
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "transcripts_index.json"

    # Load existing index for resume support
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text())
        except (json.JSONDecodeError, OSError):
            index = {}
    else:
        index = {}

    consecutive_blocks = 0
    backoff_delays = [30, 60, 120, 240, 300]
    total = len(videos)
    extracted = 0
    skipped = 0

    for i, video in enumerate(videos):
        vid = video["id"]

        # Skip already-extracted videos
        if vid in index and index[vid].get("status") == "ok":
            skipped += 1
            continue

        # Rate limiting
        if i > 0 and consecutive_blocks == 0:
            time.sleep(delay)

        result = fetch_single_transcript(ytt_api, vid, lang)

        if result["status"] == "ok":
            consecutive_blocks = 0
            extracted += 1

            # Write plain text file
            txt_file = transcripts_dir / f"{vid}.txt"
            txt_file.write_text(result["text"])

            # Update index
            index[vid] = {
                "status": "ok",
                "language": result["language"],
                "type": result["type"],
                "word_count": result["word_count"],
            }
        elif result["status"] == "no_captions":
            consecutive_blocks = 0
            index[vid] = {"status": "no_captions"}
        else:
            # Possible IP block or transient error
            consecutive_blocks += 1
            index[vid] = {"status": result["status"]}

            if consecutive_blocks <= len(backoff_delays):
                wait = backoff_delays[consecutive_blocks - 1]
                print(f"  Block detected ({consecutive_blocks}x), backing off {wait}s...",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  Aborting after {consecutive_blocks} consecutive blocks. "
                      f"Progress saved.", file=sys.stderr)
                break

        # Checkpoint every 25 videos
        if (i + 1) % 25 == 0 or i == total - 1:
            index_file.write_text(json.dumps(index, indent=2))

        # Progress
        done = extracted + skipped
        status_counts = f"{extracted} extracted, {skipped} resumed"
        remaining = total - (i + 1)
        print(f"  Transcripts: [{i+1}/{total}] {status_counts}, {remaining} remaining",
              file=sys.stderr)

    # Final save
    index_file.write_text(json.dumps(index, indent=2))
    print(f"  Transcript index: {index_file}")

    ok_count = sum(1 for v in index.values() if v.get("status") == "ok")
    no_cap = sum(1 for v in index.values() if v.get("status") == "no_captions")
    errors = sum(1 for v in index.values() if v.get("status") not in ("ok", "no_captions"))
    print(f"  Summary: {ok_count} extracted, {no_cap} no captions, {errors} errors")

    return index


def log_to_session(channel: dict, video_count: int, tier: int, category: str) -> None:
    """Log to active research session if one exists."""
    local_agent = Path.cwd() / ".agent" / "research"
    scratchpad_file = local_agent / "scratchpad.json"

    if not scratchpad_file.exists():
        print("  No active session found, skipping session log.", file=sys.stderr)
        return

    try:
        scratchpad = json.loads(scratchpad_file.read_text())

        if "urls_visited" not in scratchpad:
            scratchpad["urls_visited"] = []

        scratchpad["urls_visited"].append({
            "url": f"https://youtube.com/@{channel['handle']}",
            "source": "YouTube Channel",
            "tier": tier,
            "category": category,
            "channel_name": channel["name"],
            "videos_count": video_count,
            "timestamp": datetime.now().isoformat()
        })

        scratchpad_file.write_text(json.dumps(scratchpad, indent=2))
        print(f"  Logged to session: {scratchpad_file}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: Failed to update session: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape all videos from a YouTube channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s @NateBJones
  %(prog)s https://youtube.com/@anthropic --tier 1 --category labs
  %(prog)s @3blue1brown --limit 100
  %(prog)s @TwoMinutePapers --log-to-session
        """
    )

    parser.add_argument("channel", help="Channel URL or @handle")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=2,
                        help="Source tier for research tracking (default: 2)")
    parser.add_argument("--category", choices=["labs", "research", "industry", "video", "education"],
                        default="video", help="Category (default: video)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max videos to fetch (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Custom output directory")
    parser.add_argument("--urls-only", action="store_true",
                        help="Only output urls.txt file")
    parser.add_argument("--log-to-session", action="store_true",
                        help="Add to active research session")
    parser.add_argument("--transcripts", action="store_true",
                        help="Extract video transcripts (requires youtube-transcript-api)")
    parser.add_argument("--transcript-lang", default="en",
                        help="Preferred transcript language (default: en)")
    parser.add_argument("--transcript-delay", type=float, default=1.5,
                        help="Seconds between transcript requests (default: 1.5)")

    args = parser.parse_args()

    # Get API key
    api_key = get_api_key()

    # Parse channel input
    handle = parse_channel_input(args.channel)
    print(f"Resolving channel @{handle}...")

    # Resolve channel
    channel = resolve_channel(handle, api_key)
    print(f"  Found: {channel['name']} ({channel['id']})")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_youtube_dir() / handle

    # Fetch videos
    print("Fetching videos from uploads playlist...")
    videos = fetch_all_videos(channel["uploads_playlist"], api_key, args.limit)
    print(f"  Total: {len(videos)} videos")

    # Fetch detailed metadata (unless urls-only)
    if not args.urls_only:
        print("Fetching video metadata...")
        video_ids = [v["id"] for v in videos]
        metadata = fetch_video_metadata(video_ids, api_key)

        # Merge metadata into videos
        for video in videos:
            if video["id"] in metadata:
                video.update(metadata[video["id"]])

    # Fetch transcripts if requested
    if args.transcripts:
        print("Extracting transcripts...")
        transcript_index = fetch_transcripts(
            videos, output_dir, lang=args.transcript_lang, delay=args.transcript_delay
        )
        # Merge transcript metadata into video records
        for video in videos:
            vid = video["id"]
            if vid in transcript_index:
                entry = transcript_index[vid]
                video["transcript"] = {
                    "status": entry.get("status"),
                    "language": entry.get("language"),
                    "type": entry.get("type"),
                    "word_count": entry.get("word_count"),
                }

    # Write outputs
    print(f"Writing output files to {output_dir}...")
    write_outputs(channel, videos, output_dir, args.urls_only)

    # Update registry
    update_channels_registry(channel, len(videos))

    # Log to session if requested
    if args.log_to_session:
        log_to_session(channel, len(videos), args.tier, args.category)

    print(f"\nDone! Scraped {len(videos)} videos from @{handle}")


if __name__ == "__main__":
    main()
