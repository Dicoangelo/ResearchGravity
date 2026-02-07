#!/usr/bin/env python3
"""
ChatGPT Deep Work Importer
===========================

Imports scored ChatGPT conversations into ResearchGravity storage.

Maps:
  - ChatGPT conversation → RG session (with quality metadata)
  - Assistant messages (>500 chars) → RG findings (searchable knowledge)

Usage:
    python3 chatgpt_importer.py <export_path> [--tier deep_work] [--dry-run]

Requires:
    - quality_scores.json in export_path (from chatgpt_quality_scorer.py)
    - conversations.json in export_path (ChatGPT export)
"""

import asyncio
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent to path for storage imports
sys.path.insert(0, str(Path(__file__).parent))

from storage.engine import StorageEngine, get_engine


# Minimum assistant message length to extract as a finding
MIN_FINDING_LENGTH = 500

# Map quality scorer purposes to finding types
PURPOSE_TO_TYPE = {
    "research": "research",
    "coding": "technical",
    "thinking": "insight",
    "learning": "learning",
    "random": "observation",
}


def load_quality_scores(export_path: Path) -> Dict[str, Dict]:
    """Load quality scores and index by conversation_id."""
    scores_file = export_path / "quality_scores.json"
    if not scores_file.exists():
        print(f"❌ quality_scores.json not found in {export_path}")
        print("   Run chatgpt_quality_scorer.py first.")
        sys.exit(1)

    data = json.loads(scores_file.read_text())
    return {item["conversation_id"]: item for item in data}


def load_conversations(export_path: Path) -> List[Dict]:
    """Load raw conversations from ChatGPT export."""
    conv_file = export_path / "conversations.json"
    if not conv_file.exists():
        print(f"❌ conversations.json not found in {export_path}")
        sys.exit(1)

    return json.loads(conv_file.read_text())


def extract_messages(conversation: Dict) -> List[Dict]:
    """Extract messages from a ChatGPT conversation mapping."""
    messages = []
    mapping = conversation.get("mapping", {})

    for msg_id, msg_data in mapping.items():
        message = msg_data.get("message")
        if not message:
            continue

        author = message.get("author", {})
        role = author.get("role", "")
        content = message.get("content", {})
        parts = content.get("parts", [])

        if role == "system":
            continue

        # Filter out non-string parts (images, files, etc.)
        text = "\n".join(
            p if isinstance(p, str)
            else p.get("text", "") if isinstance(p, dict)
            else str(p)
            for p in parts
        ) if parts else ""

        if not text or not text.strip():
            continue

        messages.append({
            "role": role,
            "content": text.strip(),
            "create_time": message.get("create_time", 0),
        })

    messages.sort(key=lambda m: m["create_time"])
    return messages


def conversation_to_session(
    conversation: Dict,
    scores: Dict,
    messages: List[Dict],
) -> Dict[str, Any]:
    """Map a ChatGPT conversation to a ResearchGravity session."""
    conv_id = conversation.get("id", conversation.get("conversation_id", ""))
    metrics = scores.get("metrics", {})
    title = conversation.get("title") or scores.get("title", "Untitled")
    create_time = conversation.get("create_time") or scores.get("create_time", 0)
    update_time = conversation.get("update_time") or scores.get("update_time", 0)

    # Calculate transcript tokens (rough: 4 chars per token)
    total_chars = sum(len(m["content"]) for m in messages)
    transcript_tokens = total_chars // 4

    return {
        "id": f"chatgpt-{conv_id}",
        "topic": title,
        "status": "archived",
        "project": "chatgpt-import",
        "started_at": datetime.fromtimestamp(create_time).isoformat() if create_time else None,
        "archived_at": datetime.fromtimestamp(update_time).isoformat() if update_time else None,
        "transcript_tokens": transcript_tokens,
        "finding_count": 0,  # Updated after findings extracted
        "url_count": 0,
        "metadata": {
            "source": "chatgpt_export",
            "conversation_id": conv_id,
            "quality_score": metrics.get("quality_score", 0),
            "cognitive_mode": metrics.get("cognitive_mode", "unknown"),
            "purpose": metrics.get("purpose", "unknown"),
            "depth": metrics.get("depth", 0),
            "focus": metrics.get("focus", 0),
            "signal": metrics.get("signal", 0),
            "signal_strength": metrics.get("signal_strength", 0),
            "message_count": metrics.get("message_count", len(messages)),
            "total_chars": metrics.get("total_chars", total_chars),
        },
    }


def extract_findings(
    session_id: str,
    messages: List[Dict],
    purpose: str,
    quality_score: float,
) -> List[Dict[str, Any]]:
    """Extract findings from assistant messages in a conversation."""
    findings = []
    finding_type = PURPOSE_TO_TYPE.get(purpose, "observation")

    for msg in messages:
        if msg["role"] != "assistant":
            continue

        content = msg["content"]
        if len(content) < MIN_FINDING_LENGTH:
            continue

        # Truncate very long messages to first 3000 chars for storage
        if len(content) > 3000:
            content = content[:3000] + "\n\n[truncated]"

        # Deterministic ID from content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        finding_id = f"chatgpt-{content_hash}"

        findings.append({
            "id": finding_id,
            "session_id": session_id,
            "content": content,
            "type": finding_type,
            "confidence": quality_score,
            "evidence": {
                "sources": ["chatgpt_export"],
                "confidence": quality_score,
            },
            "project": "chatgpt-import",
        })

    return findings


async def import_deep_work(
    export_path: Path,
    tier: str = "deep_work",
    dry_run: bool = False,
):
    """Import conversations of the specified tier into ResearchGravity."""

    print(f"📦 Loading ChatGPT export from {export_path}...")
    scores_index = load_quality_scores(export_path)
    conversations = load_conversations(export_path)
    print(f"   Loaded {len(conversations)} conversations, {len(scores_index)} scores\n")

    # Filter to target tier
    target_ids = {
        cid for cid, data in scores_index.items()
        if data.get("metrics", {}).get("cognitive_mode") == tier
    }

    print(f"🎯 Found {len(target_ids)} {tier} conversations\n")

    if not target_ids:
        print("   Nothing to import.")
        return

    # Build conversation lookup
    conv_lookup = {}
    for conv in conversations:
        cid = conv.get("id", conv.get("conversation_id", ""))
        if cid in target_ids:
            conv_lookup[cid] = conv

    print(f"   Matched {len(conv_lookup)} conversations in export\n")

    if dry_run:
        print("🏃 DRY RUN — showing what would be imported:\n")
        for i, (cid, conv) in enumerate(conv_lookup.items()):
            scores = scores_index[cid]
            title = conv.get("title") or scores.get("title", "Untitled")
            qs = scores["metrics"]["quality_score"]
            purpose = scores["metrics"]["purpose"]
            msgs = extract_messages(conv)
            assistant_msgs = [m for m in msgs if m["role"] == "assistant" and len(m["content"]) >= MIN_FINDING_LENGTH]
            print(f"   {i+1:3d}. [{qs:.3f}] [{purpose:8s}] {title[:60]:60s} → {len(assistant_msgs)} findings")
            if i >= 29:
                print(f"   ... and {len(conv_lookup) - 30} more")
                break
        print()
        total_findings = 0
        for cid, conv in conv_lookup.items():
            msgs = extract_messages(conv)
            total_findings += len([m for m in msgs if m["role"] == "assistant" and len(m["content"]) >= MIN_FINDING_LENGTH])
        print(f"   Total: {len(conv_lookup)} sessions, ~{total_findings} findings")
        print(f"\n   Run without --dry-run to import.")
        return

    # Initialize storage engine
    print("🔧 Initializing storage engine...")
    engine = await get_engine()
    print("   ✅ Storage engine ready\n")

    # Import
    print("📥 Importing conversations...\n")
    total_sessions = 0
    total_findings = 0
    errors = []

    for i, (cid, conv) in enumerate(conv_lookup.items()):
        scores = scores_index[cid]
        title = conv.get("title") or scores.get("title", "Untitled")

        try:
            # Extract messages
            messages = extract_messages(conv)

            # Create session
            session = conversation_to_session(conv, scores, messages)

            # Extract findings
            findings = extract_findings(
                session_id=session["id"],
                messages=messages,
                purpose=scores["metrics"].get("purpose", "random"),
                quality_score=scores["metrics"].get("quality_score", 0),
            )

            # Update finding count
            session["finding_count"] = len(findings)

            # Store session
            await engine.store_session(session, source="chatgpt_import")

            # Store findings
            for finding in findings:
                await engine.store_finding(finding, source="chatgpt_import")

            total_sessions += 1
            total_findings += len(findings)

            if (i + 1) % 50 == 0:
                print(f"   Imported {i + 1}/{len(conv_lookup)}... ({total_findings} findings)")

        except Exception as e:
            errors.append({"conversation_id": cid, "title": title, "error": str(e)})
            if len(errors) <= 5:
                print(f"   ⚠️  Error on '{title[:40]}': {e}")

    # Close engine
    await engine.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"IMPORT COMPLETE")
    print(f"{'='*60}")
    print(f"   Sessions imported: {total_sessions}")
    print(f"   Findings extracted: {total_findings}")
    if errors:
        print(f"   Errors: {len(errors)}")
    print(f"{'='*60}\n")

    # Save import log
    log_path = export_path / "import_log.json"
    log_data = {
        "imported_at": datetime.now().isoformat(),
        "tier": tier,
        "sessions_imported": total_sessions,
        "findings_extracted": total_findings,
        "errors": errors,
    }
    log_path.write_text(json.dumps(log_data, indent=2))
    print(f"📝 Import log saved to: {log_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 chatgpt_importer.py <export_path> [--tier deep_work] [--dry-run]")
        sys.exit(1)

    export_path = Path(sys.argv[1]).expanduser()

    if not export_path.exists():
        print(f"❌ Export path not found: {export_path}")
        sys.exit(1)

    tier = "deep_work"
    dry_run = False

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--tier" and i + 1 < len(args):
            tier = args[i + 1]
            i += 2
        elif args[i] == "--dry-run":
            dry_run = True
            i += 1
        else:
            i += 1

    asyncio.run(import_deep_work(export_path, tier=tier, dry_run=dry_run))


if __name__ == "__main__":
    main()
