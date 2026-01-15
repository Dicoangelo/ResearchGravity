#!/usr/bin/env python3
"""YouTube Channel Scraper — ResearchGravity Integration

Scrapes all videos from a YouTube channel and outputs in multiple formats
for research tracking and analysis.

Usage:
  python3 youtube_channel.py https://youtube.com/@ChannelName
  python3 youtube_channel.py @ChannelName --tier 1 --category labs
  python3 youtube_channel.py @ChannelName --limit 50 --output-dir ~/custom

Environment:
  YOUTUBE_API_KEY    Required. Get from Google Cloud Console.
                     https://console.cloud.google.com/apis/credentials

Output Files:
  urls.txt     - One URL per line (for batch processing)
  videos.txt   - Date + URL + Title (human readable)
  full.json    - Complete metadata (machine readable)
"""

import argparse
import json
import os
import re
import sys
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
            print(f"ERROR: API quota exceeded or invalid key.", file=sys.stderr)
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
    print(f"Fetching videos from uploads playlist...")
    videos = fetch_all_videos(channel["uploads_playlist"], api_key, args.limit)
    print(f"  Total: {len(videos)} videos")

    # Fetch detailed metadata (unless urls-only)
    if not args.urls_only:
        print(f"Fetching video metadata...")
        video_ids = [v["id"] for v in videos]
        metadata = fetch_video_metadata(video_ids, api_key)

        # Merge metadata into videos
        for video in videos:
            if video["id"] in metadata:
                video.update(metadata[video["id"]])

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
