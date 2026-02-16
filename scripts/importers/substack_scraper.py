#!/usr/bin/env python3
"""Substack Archive Scraper — ResearchGravity Integration

Scrapes all articles from a Substack publication and outputs in multiple formats
for research tracking and analysis.

Usage:
  python3 substack_scraper.py https://example.substack.com
  python3 substack_scraper.py example --tier 1 --category research
  python3 substack_scraper.py @username --limit 50 --output-dir ~/custom
  python3 substack_scraper.py example --full-content --content-delay 2.0

Output Files:
  urls.txt           - One URL per line (for batch processing)
  articles.txt       - Date + URL + Title (human readable)
  full.json          - Complete metadata (machine readable)
  content/           - Full article text per post (with --full-content)
  content_index.json - Content extraction status (with --full-content)
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def get_agent_core_dir() -> Path:
    """Global research storage directory."""
    return Path.home() / ".agent-core"


def get_substack_dir() -> Path:
    """Substack data storage directory."""
    substack_dir = get_agent_core_dir() / "research" / "substack"
    substack_dir.mkdir(parents=True, exist_ok=True)
    return substack_dir


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self.current_skip = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.current_skip = True
        elif tag in ('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'):
            self.text_parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.current_skip = False
        elif tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.text_parts.append('\n')

    def handle_data(self, data):
        if not self.current_skip:
            self.text_parts.append(data)

    def get_text(self) -> str:
        text = ''.join(self.text_parts)
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    parser = HTMLTextExtractor()
    parser.feed(unescape(html))
    return parser.get_text()


def parse_substack_input(substack_input: str) -> tuple[str, str]:
    """Extract publication name and construct base URL from various input formats.

    Returns: (publication_name, base_url)
    """
    substack_input = substack_input.strip()

    # Handle full URLs: https://example.substack.com/archive or https://example.substack.com
    url_match = re.match(r'https?://([^.]+)\.substack\.com(?:/.*)?', substack_input)
    if url_match:
        name = url_match.group(1)
        return name, f"https://{name}.substack.com"

    # Handle substack.com/@username format
    username_match = re.match(r'https?://(?:www\.)?substack\.com/@([^/?\s]+)', substack_input)
    if username_match:
        name = username_match.group(1)
        return name, f"https://substack.com/@{name}"

    # Handle @username format
    if substack_input.startswith('@'):
        name = substack_input[1:]
        return name, f"https://substack.com/@{name}"

    # Handle custom domain: https://example.com
    if substack_input.startswith('http'):
        parsed = urlparse(substack_input)
        name = parsed.netloc.replace('www.', '').split('.')[0]
        return name, f"{parsed.scheme}://{parsed.netloc}"

    # Assume it's just the publication name
    return substack_input, f"https://{substack_input}.substack.com"


def fetch_rss_feed(base_url: str) -> str:
    """Fetch the RSS feed content."""
    feed_url = f"{base_url}/feed"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    request = urllib.request.Request(feed_url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"ERROR: RSS feed not found at {feed_url}", file=sys.stderr)
            print("  Try the full Substack URL or check the publication name.", file=sys.stderr)
        else:
            print(f"ERROR: Failed to fetch RSS feed: {e}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: Network error: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_archive_api(base_url: str, limit: Optional[int] = None) -> list[dict]:
    """Fetch articles using Substack's internal API for complete archive.

    The RSS feed only returns ~20 recent posts. This uses the archive API
    to get the complete history.
    """
    articles = []
    offset = 0
    batch_size = 50

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    while True:
        # Substack archive API endpoint
        api_url = f"{base_url}/api/v1/archive?sort=new&offset={offset}&limit={batch_size}"

        request = urllib.request.Request(api_url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Might be a custom domain, try without /api/v1
                print(f"  Archive API not available, falling back to RSS only.", file=sys.stderr)
                return []
            print(f"WARNING: Failed to fetch archive batch at offset {offset}: {e}", file=sys.stderr)
            break
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"WARNING: Error fetching archive: {e}", file=sys.stderr)
            break

        if not data:
            break

        for item in data:
            # Skip if not a post (could be a thread, podcast, etc.)
            if item.get('type') != 'newsletter' and item.get('type') != 'post':
                if item.get('type') not in ('thread', 'podcast'):
                    pass  # Include most types

            post_date = item.get('post_date', '')[:10] if item.get('post_date') else ''

            article = {
                'id': str(item.get('id', '')),
                'title': item.get('title', 'Untitled'),
                'slug': item.get('slug', ''),
                'published': post_date,
                'url': item.get('canonical_url', ''),
                'subtitle': item.get('subtitle', ''),
                'description': item.get('description', ''),
                'word_count': item.get('wordcount', 0),
                'likes': item.get('reactions', {}).get('❤', 0) if isinstance(item.get('reactions'), dict) else 0,
                'comments': item.get('comment_count', 0),
                'is_paid': item.get('audience') == 'only_paid',
                'type': item.get('type', 'post'),
            }

            if not article['url'] and article['slug']:
                article['url'] = f"{base_url}/p/{article['slug']}"

            articles.append(article)

            if limit and len(articles) >= limit:
                return articles

        offset += batch_size

        # Progress indicator
        print(f"  Fetched {len(articles)} articles...", file=sys.stderr)

        if len(data) < batch_size:
            break

        # Small delay to be respectful
        time.sleep(0.5)

    return articles


def parse_rss_feed(xml_content: str, base_url: str) -> tuple[dict, list[dict]]:
    """Parse RSS feed and extract publication info and articles."""
    root = ET.fromstring(xml_content)

    channel = root.find('channel')
    if channel is None:
        print("ERROR: Invalid RSS feed structure.", file=sys.stderr)
        sys.exit(1)

    # Extract publication info
    publication = {
        'name': channel.findtext('title', 'Unknown'),
        'description': channel.findtext('description', ''),
        'url': base_url,
        'image': '',
    }

    image = channel.find('image')
    if image is not None:
        publication['image'] = image.findtext('url', '')

    # Extract articles
    articles = []
    for item in channel.findall('item'):
        pub_date = item.findtext('pubDate', '')
        # Parse date: "Wed, 15 Jan 2025 12:00:00 GMT" -> "2025-01-15"
        date_str = ''
        if pub_date:
            try:
                # Try parsing RSS date format
                dt = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                date_str = dt.strftime('%Y-%m-%d')
            except ValueError:
                try:
                    dt = datetime.strptime(pub_date[:25], '%a, %d %b %Y %H:%M:%S')
                    date_str = dt.strftime('%Y-%m-%d')
                except ValueError:
                    date_str = pub_date[:10]

        # Get content if available
        content = ''
        content_encoded = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
        if content_encoded is not None and content_encoded.text:
            content = html_to_text(content_encoded.text)

        url = item.findtext('link', '')

        # Extract slug from URL
        slug = ''
        slug_match = re.search(r'/p/([^/?]+)', url)
        if slug_match:
            slug = slug_match.group(1)

        article = {
            'id': item.findtext('guid', url),
            'title': item.findtext('title', 'Untitled'),
            'slug': slug,
            'published': date_str,
            'url': url,
            'description': html_to_text(item.findtext('description', '')),
            'content_preview': content[:500] if content else '',
            'word_count': len(content.split()) if content else 0,
        }

        articles.append(article)

    return publication, articles


def fetch_full_content(url: str) -> dict:
    """Fetch full article content from a Substack post URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

    # Try to find the article body
    # Substack uses a specific div class for content
    content_match = re.search(
        r'<div[^>]*class="[^"]*available-content[^"]*"[^>]*>(.*?)</div>\s*(?:<div[^>]*class="[^"]*subscription-widget)',
        html,
        re.DOTALL | re.IGNORECASE
    )

    if not content_match:
        # Try alternative pattern
        content_match = re.search(
            r'<div[^>]*class="[^"]*body[^"]*markup[^"]*"[^>]*>(.*?)</div>',
            html,
            re.DOTALL | re.IGNORECASE
        )

    if not content_match:
        # Try the post-content class
        content_match = re.search(
            r'<div[^>]*class="[^"]*post-content[^"]*"[^>]*>(.*?)</div>\s*<div[^>]*class="[^"]*post-footer',
            html,
            re.DOTALL | re.IGNORECASE
        )

    if content_match:
        content_html = content_match.group(1)
        text = html_to_text(content_html)
        word_count = len(text.split())
        return {
            'status': 'ok',
            'text': text,
            'word_count': word_count,
        }

    # Fallback: try to extract from JSON-LD
    jsonld_match = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL)
    if jsonld_match:
        try:
            data = json.loads(jsonld_match.group(1))
            if isinstance(data, list):
                data = data[0]
            if 'articleBody' in data:
                text = data['articleBody']
                return {
                    'status': 'ok',
                    'text': text,
                    'word_count': len(text.split()),
                }
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    return {'status': 'no_content'}


def fetch_all_content(articles: list[dict], output_dir: Path, delay: float = 2.0) -> dict:
    """Fetch full content for all articles with rate limiting and resume support."""
    content_dir = output_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "content_index.json"

    # Load existing index for resume support
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text())
        except (json.JSONDecodeError, OSError):
            index = {}
    else:
        index = {}

    total = len(articles)
    extracted = 0
    skipped = 0
    errors = 0

    for i, article in enumerate(articles):
        article_id = article.get('id') or article.get('slug') or str(i)

        # Skip already-extracted articles
        if article_id in index and index[article_id].get('status') == 'ok':
            skipped += 1
            continue

        # Rate limiting
        if i > 0:
            time.sleep(delay)

        url = article.get('url')
        if not url:
            index[article_id] = {'status': 'no_url'}
            errors += 1
            continue

        result = fetch_full_content(url)

        if result['status'] == 'ok':
            extracted += 1

            # Write plain text file
            safe_slug = re.sub(r'[^\w\-]', '_', article.get('slug', article_id))[:100]
            txt_file = content_dir / f"{safe_slug}.txt"
            txt_file.write_text(result['text'])

            index[article_id] = {
                'status': 'ok',
                'word_count': result['word_count'],
                'file': txt_file.name,
            }
        else:
            errors += 1
            index[article_id] = {'status': result.get('status', 'error')}

        # Checkpoint every 10 articles
        if (i + 1) % 10 == 0 or i == total - 1:
            index_file.write_text(json.dumps(index, indent=2))

        # Progress
        done = extracted + skipped + errors
        print(f"  Content: [{done}/{total}] {extracted} extracted, {skipped} resumed, {errors} errors",
              file=sys.stderr)

    # Final save
    index_file.write_text(json.dumps(index, indent=2))
    print(f"  Content index: {index_file}")

    return index


def write_outputs(publication: dict, articles: list[dict], output_dir: Path, urls_only: bool = False) -> None:
    """Write output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. urls.txt - always write
    urls_file = output_dir / "urls.txt"
    urls_file.write_text("\n".join(a["url"] for a in articles if a.get("url")))
    print(f"  Written: {urls_file}")

    if urls_only:
        return

    # 2. articles.txt - date + url + title
    articles_file = output_dir / "articles.txt"
    lines = [f"{publication['name']} — {len(articles)} articles", "=" * 60, ""]
    for a in articles:
        lines.append(f"{a['published']} | {a['url']}")
        lines.append(a["title"])
        if a.get('subtitle'):
            lines.append(f"  {a['subtitle']}")
        lines.append("")
    articles_file.write_text("\n".join(lines))
    print(f"  Written: {articles_file}")

    # 3. full.json - complete metadata
    full_file = output_dir / "full.json"
    output = {
        "publication": {
            "name": publication["name"],
            "description": publication.get("description", ""),
            "url": publication["url"],
            "scraped_at": datetime.now().isoformat()
        },
        "articles": articles,
        "total": len(articles)
    }
    full_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"  Written: {full_file}")


def update_publications_registry(publication: dict, article_count: int) -> None:
    """Update the publications registry with this scrape."""
    registry_file = get_substack_dir() / "publications.json"

    if registry_file.exists():
        registry = json.loads(registry_file.read_text())
    else:
        registry = {"publications": []}

    name = publication["name"]
    existing = next((p for p in registry["publications"] if p["name"] == name), None)
    entry = {
        "name": name,
        "url": publication["url"],
        "article_count": article_count,
        "last_scraped": datetime.now().isoformat()
    }

    if existing:
        existing.update(entry)
    else:
        registry["publications"].append(entry)

    registry_file.write_text(json.dumps(registry, indent=2))


def log_to_session(publication: dict, article_count: int, tier: int, category: str) -> None:
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
            "url": publication["url"],
            "source": "Substack Publication",
            "tier": tier,
            "category": category,
            "publication_name": publication["name"],
            "articles_count": article_count,
            "timestamp": datetime.now().isoformat()
        })

        scratchpad_file.write_text(json.dumps(scratchpad, indent=2))
        print(f"  Logged to session: {scratchpad_file}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  WARNING: Failed to update session: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape all articles from a Substack publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stratechery
  %(prog)s https://www.platformer.news --tier 1 --category research
  %(prog)s https://example.substack.com --limit 100
  %(prog)s newsletter --full-content --log-to-session
        """
    )

    parser.add_argument("substack", help="Substack URL, publication name, or @username")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=2,
                        help="Source tier for research tracking (default: 2)")
    parser.add_argument("--category", choices=["labs", "research", "industry", "newsletter", "education"],
                        default="newsletter", help="Category (default: newsletter)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max articles to fetch (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Custom output directory")
    parser.add_argument("--urls-only", action="store_true",
                        help="Only output urls.txt file")
    parser.add_argument("--rss-only", action="store_true",
                        help="Only use RSS feed (faster but limited to ~20 recent posts)")
    parser.add_argument("--full-content", action="store_true",
                        help="Fetch full article content (slow, respects rate limits)")
    parser.add_argument("--content-delay", type=float, default=2.0,
                        help="Seconds between content requests (default: 2.0)")
    parser.add_argument("--log-to-session", action="store_true",
                        help="Add to active research session")

    args = parser.parse_args()

    # Parse input
    name, base_url = parse_substack_input(args.substack)
    print(f"Resolving publication: {name} ({base_url})...")

    # Fetch RSS feed first (always needed for publication info)
    print("Fetching RSS feed...")
    xml_content = fetch_rss_feed(base_url)
    publication, rss_articles = parse_rss_feed(xml_content, base_url)
    print(f"  Found: {publication['name']}")
    print(f"  RSS articles: {len(rss_articles)}")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        safe_name = re.sub(r'[^\w\-]', '_', name)
        output_dir = get_substack_dir() / safe_name

    # Fetch complete archive unless --rss-only
    if args.rss_only:
        articles = rss_articles
    else:
        print("Fetching complete archive via API...")
        api_articles = fetch_archive_api(base_url, args.limit)

        if api_articles:
            articles = api_articles
            print(f"  Total from API: {len(articles)} articles")
        else:
            # Fall back to RSS if API fails
            articles = rss_articles
            print(f"  Using RSS feed: {len(articles)} articles")

    # Apply limit if specified and not already applied
    if args.limit and len(articles) > args.limit:
        articles = articles[:args.limit]

    # Fetch full content if requested
    if args.full_content:
        print("Fetching full article content...")
        content_index = fetch_all_content(articles, output_dir, delay=args.content_delay)

        # Merge content metadata into articles
        for article in articles:
            article_id = article.get('id') or article.get('slug')
            if article_id and article_id in content_index:
                entry = content_index[article_id]
                article['content'] = {
                    'status': entry.get('status'),
                    'word_count': entry.get('word_count'),
                    'file': entry.get('file'),
                }

    # Write outputs
    print(f"Writing output files to {output_dir}...")
    write_outputs(publication, articles, output_dir, args.urls_only)

    # Update registry
    update_publications_registry(publication, len(articles))

    # Log to session if requested
    if args.log_to_session:
        log_to_session(publication, len(articles), args.tier, args.category)

    print(f"\nDone! Scraped {len(articles)} articles from {publication['name']}")


if __name__ == "__main__":
    main()
