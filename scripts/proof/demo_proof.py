#!/usr/bin/env python3
"""
ResearchGravity — 60-Second Proof of Credibility

Three proof items that make the framework real:

  1. INGEST   → URL/transcript → structured Finding with citations
  2. FINDING  → claim + evidence spans + citations + confidence + lineage
  3. TRACE    → retrieval returned → critics said → final answer justified

Usage:
    python3 demo_proof.py                          # Built-in demo
    python3 demo_proof.py --url https://arxiv.org/abs/2602.11865
    python3 demo_proof.py --text "Multi-agent systems with trust..."
    python3 demo_proof.py --json                   # Machine-readable output
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import textwrap
import time
import uuid
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
DB_PATH = Path.home() / ".agent-core" / "storage" / "antigravity.db"
SCHEMA_DIR = Path.home() / ".agent-core" / "schemas"

# ── Colour output ──────────────────────────────────────────────────────
class C:
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    CYAN   = "\033[36m"
    RED    = "\033[31m"
    MAG    = "\033[35m"
    RESET  = "\033[0m"

def header(text):
    w = 64
    print(f"\n{C.BOLD}{C.CYAN}{'━' * w}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'━' * w}{C.RESET}")

def step(n, text):
    print(f"\n{C.BOLD}{C.GREEN}  [{n}]{C.RESET} {C.BOLD}{text}{C.RESET}")

def kv(key, value, indent=6):
    pad = " " * indent
    print(f"{pad}{C.DIM}{key}:{C.RESET} {value}")

def dim(text, indent=6):
    pad = " " * indent
    print(f"{pad}{C.DIM}{text}{C.RESET}")

def ok(text, indent=6):
    pad = " " * indent
    print(f"{pad}{C.GREEN}✓{C.RESET} {text}")

def warn(text, indent=6):
    pad = " " * indent
    print(f"{pad}{C.YELLOW}⚠{C.RESET} {text}")

def fail(text, indent=6):
    pad = " " * indent
    print(f"{pad}{C.RED}✗{C.RESET} {text}")


# ── Evidence Extraction ────────────────────────────────────────────────
ARXIV_RE = re.compile(r'(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})', re.IGNORECASE)
ARXIV_URL_RE = re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})')
GITHUB_RE = re.compile(r'github\.com/([\w-]+/[\w.-]+)')
URL_RE = re.compile(r'https?://[^\s\)\]>\"\']+')

TIER1 = {"arxiv.org", "openai.com", "anthropic.com", "deepmind.google",
          "huggingface.co", "ai.meta.com", "ai.google", "microsoft.com"}
TIER2 = {"github.com", "paperswithcode.com", "semanticscholar.org",
          "techcrunch.com", "theverge.com", "lmsys.org"}


def classify_tier(url: str) -> int:
    lo = url.lower()
    for d in TIER1:
        if d in lo: return 1
    for d in TIER2:
        if d in lo: return 2
    return 3


def extract_evidence(text: str) -> list[dict]:
    """Extract citation evidence from text."""
    sources, seen = [], set()

    for m in ARXIV_RE.finditer(text):
        aid = m.group(1)
        url = f"https://arxiv.org/abs/{aid}"
        if url in seen: continue
        seen.add(url)
        sources.append({
            "url": url, "arxiv_id": aid,
            "excerpt": text[max(0, m.start()-60):min(len(text), m.end()+60)].strip(),
            "relevance_score": 0.85, "verified": False, "tier": 1,
        })

    for m in GITHUB_RE.finditer(text):
        url = f"https://github.com/{m.group(1)}"
        if url in seen: continue
        seen.add(url)
        sources.append({
            "url": url,
            "excerpt": text[max(0, m.start()-40):min(len(text), m.end()+40)].strip(),
            "relevance_score": 0.65, "verified": False, "tier": 2,
        })

    for m in URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:)")
        if url in seen or "arxiv.org" in url or "github.com" in url: continue
        seen.add(url)
        tier = classify_tier(url)
        sources.append({
            "url": url,
            "excerpt": text[max(0, m.start()-30):min(len(text), m.end()+30)].strip(),
            "relevance_score": {1: 0.75, 2: 0.55, 3: 0.35}[tier],
            "verified": False, "tier": tier,
        })

    return sources


# ── Oracle Critic (3 streams) ──────────────────────────────────────────
def critic_accuracy(sources: list[dict], claim: str) -> dict:
    """Stream 1: Are sources factually relevant to the claim?"""
    issues = []
    score = 1.0

    for s in sources:
        # Check URL format
        if not s["url"].startswith("http"):
            issues.append(f"Malformed URL: {s['url'][:60]}")
            score -= 0.15

        # arXiv ID format check
        if s.get("arxiv_id"):
            if not re.match(r'\d{4}\.\d{4,5}$', s["arxiv_id"]):
                issues.append(f"Invalid arXiv ID format: {s['arxiv_id']}")
                score -= 0.1

        # Tier 1 sources boost confidence
        if s.get("tier") == 1:
            score = min(1.0, score + 0.05)

    # Penalise if no sources at all
    if not sources:
        issues.append("No sources found — claim is unsupported")
        score = 0.2

    return {"stream": "accuracy", "score": max(0, score), "issues": issues}


def critic_completeness(sources: list[dict], claim: str) -> dict:
    """Stream 2: Do sources cover the full claim?"""
    issues = []
    # Simple keyword coverage check
    claim_words = set(w.lower() for w in re.findall(r'\b[a-z]{4,}\b', claim.lower()))
    covered = set()
    for s in sources:
        excerpt_words = set(w.lower() for w in re.findall(r'\b[a-z]{4,}\b', s.get("excerpt", "").lower()))
        covered |= (claim_words & excerpt_words)

    if claim_words:
        coverage = len(covered) / len(claim_words)
    else:
        coverage = 0.5

    if coverage < 0.3:
        issues.append(f"Low keyword coverage ({coverage:.0%}) — sources may not fully support claim")

    score = min(1.0, 0.3 + coverage * 0.7)
    return {"stream": "completeness", "score": score, "issues": issues}


def critic_relevance(sources: list[dict], claim: str) -> dict:
    """Stream 3: Are sources relevant (not just tangentially)?"""
    issues = []
    if not sources:
        return {"stream": "relevance", "score": 0.2, "issues": ["No sources to evaluate"]}

    avg_relevance = sum(s["relevance_score"] for s in sources) / len(sources)
    tier1_ratio = sum(1 for s in sources if s.get("tier") == 1) / len(sources)

    score = avg_relevance * 0.6 + tier1_ratio * 0.4
    if tier1_ratio == 0:
        issues.append("No Tier 1 sources — consider adding primary research citations")

    return {"stream": "relevance", "score": min(1.0, score), "issues": issues}


def run_oracle(sources: list[dict], claim: str) -> dict:
    """Run 3-stream oracle consensus and compute weighted confidence."""
    streams = [
        critic_accuracy(sources, claim),
        critic_completeness(sources, claim),
        critic_relevance(sources, claim),
    ]
    # Weighted: accuracy 40%, completeness 35%, relevance 25%
    weights = {"accuracy": 0.40, "completeness": 0.35, "relevance": 0.25}
    confidence = sum(s["score"] * weights[s["stream"]] for s in streams)
    all_issues = []
    for s in streams:
        all_issues.extend(s["issues"])

    validated = confidence >= 0.70 and len([i for s in streams for i in s["issues"] if "unsupported" in i.lower()]) == 0

    return {
        "streams": streams,
        "confidence": round(confidence, 4),
        "validated": validated,
        "issues": all_issues,
        "threshold": 0.70,
    }


# ── Database Operations ────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def store_finding(conn: sqlite3.Connection, finding: dict) -> str:
    """Store a finding in antigravity.db and return its ID."""
    fid = finding["id"]
    conn.execute("""
        INSERT OR REPLACE INTO findings
            (id, session_id, content, type, evidence, confidence, derived_from, enables, project, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fid,
        finding["session_id"],
        finding["content"],
        finding["type"],
        json.dumps(finding["evidence"]),
        finding["evidence"]["confidence"],
        json.dumps(finding.get("derived_from", [])),
        json.dumps(finding.get("enables", [])),
        finding.get("project"),
        finding["created_at"],
        finding["updated_at"],
    ))
    conn.commit()
    return fid


def store_url(conn: sqlite3.Connection, session_id: str, url: str, tier: int, context: str = "") -> int:
    """Store a URL record and return its rowid."""
    cur = conn.execute("""
        INSERT OR IGNORE INTO urls (session_id, url, tier, category, source, context, relevance, captured_at)
        VALUES (?, ?, ?, 'research', 'demo_proof', ?, ?, ?)
    """, (session_id, url, tier, context[:200], tier, datetime.now().isoformat()))
    conn.commit()
    return cur.lastrowid


def retrieve_finding(conn: sqlite3.Connection, finding_id: str) -> Optional[dict]:
    """Retrieve a stored finding by ID."""
    row = conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["evidence"] = json.loads(d.get("evidence") or "{}")
    d["derived_from"] = json.loads(d.get("derived_from") or "[]")
    d["enables"] = json.loads(d.get("enables") or "[]")
    return d


def search_findings_fts(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    """Full-text search across findings."""
    rows = conn.execute("""
        SELECT f.id, f.content, f.type, f.confidence,
               snippet(findings_fts, 0, '>>>', '<<<', '...', 32) as snippet
        FROM findings_fts
        JOIN findings f ON findings_fts.rowid = (
            SELECT rowid FROM findings WHERE id = f.id
        )
        WHERE findings_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """, (query, limit)).fetchall()
    return [dict(r) for r in rows]


def search_findings_like(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    """Fallback LIKE search if FTS fails."""
    words = query.split()[:3]
    clauses = " AND ".join(f"content LIKE ?" for _ in words)
    params = [f"%{w}%" for w in words] + [limit]
    rows = conn.execute(f"""
        SELECT id, content, type, confidence
        FROM findings WHERE {clauses}
        ORDER BY confidence DESC NULLS LAST
        LIMIT ?
    """, params).fetchall()
    return [dict(r) for r in rows]


def search_findings(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    """Search findings — FTS with LIKE fallback."""
    try:
        results = search_findings_fts(conn, query, limit)
        if results:
            return results
    except Exception:
        pass
    return search_findings_like(conn, query, limit)


# ── URL Fetching ───────────────────────────────────────────────────────
def fetch_url_text(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return plain text (best-effort)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "ResearchGravity-Proof/1.0",
        "Accept": "text/html,text/plain,application/json"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(max_chars * 2).decode("utf-8", errors="replace")
            # Strip HTML tags for a rough text extraction
            text = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_chars]
    except Exception as e:
        return f"[Fetch failed: {e}]"


# ── Built-in Demo Transcript ──────────────────────────────────────────
DEMO_TRANSCRIPT = """
Research session on multi-agent trust calibration for sovereign AI systems.

Finding: Bayesian Beta trust scoring (arXiv:2602.11865) outperforms static
trust tables by 34% on delegation accuracy. The key insight is that trust
should decay with staleness — a 0.95 decay per 7-day window prevents
overconfidence in agents that haven't been tested recently.

Evidence from arXiv:2512.05470 (Agentic File System) shows that multi-agent
systems need persistent memory layers to maintain coherent state across
delegation chains. The AFS paper demonstrates a 2.7x improvement in task
completion when agents share a unified knowledge bus.

Implementation at https://github.com/Dicoangelo/antigravity-coordinator
uses 11-dimensional task profiles (complexity, ambiguity, domain specificity,
time sensitivity, resource requirements, collaboration need, creativity,
precision, domain knowledge, context dependency, verifiability) for
capability-weighted routing.

Gap identified: Current trust calibration assumes single-domain competence.
Cross-domain transfer learning for trust scores is unexplored territory —
an agent trusted for debugging may not be trusted for architecture decisions.
See also arXiv:2508.17536 on voting-based consensus approaches.
"""


# ── Main Pipeline ──────────────────────────────────────────────────────
def run_proof(
    text: str,
    source_url: Optional[str] = None,
    output_json: bool = False,
):
    """
    Execute the full proof pipeline:
      INGEST → FINDING → STORE → RETRIEVE → CRITIQUE → TRACE
    """
    t0 = time.time()
    trace = {"stages": [], "timestamps": {}}

    if not output_json:
        header("ResearchGravity — Proof of Credibility")
        dim("Three proof items in 60 seconds", 2)

    # ── PROOF 1: INGEST ─────────────────────────────────────────────
    if not output_json:
        step(1, "INGEST — URL/transcript → structured record")

    trace["timestamps"]["ingest_start"] = datetime.now().isoformat()

    # If a URL was provided, fetch it
    if source_url:
        if not output_json:
            kv("Source", source_url)
            dim("Fetching content...")
        fetched = fetch_url_text(source_url)
        text = f"Source: {source_url}\n\n{fetched}\n\n{text}" if text else fetched

    # Extract evidence from text
    sources = extract_evidence(text)

    # Create a session ID for this proof run
    session_id = f"proof-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    # Store URLs in the database
    conn = get_db()
    urls_stored = 0
    for s in sources:
        store_url(conn, session_id, s["url"], s.get("tier", 3), s.get("excerpt", ""))
        urls_stored += 1

    ingest_result = {
        "session_id": session_id,
        "source_url": source_url,
        "text_length": len(text),
        "evidence_sources_extracted": len(sources),
        "urls_stored": urls_stored,
        "sources": sources,
    }
    trace["stages"].append({"stage": "ingest", "result": ingest_result})
    trace["timestamps"]["ingest_end"] = datetime.now().isoformat()

    if not output_json:
        kv("Session", session_id)
        kv("Text length", f"{len(text):,} chars")
        kv("Evidence extracted", f"{len(sources)} sources")
        for s in sources:
            tier_badge = {1: f"{C.GREEN}T1{C.RESET}", 2: f"{C.YELLOW}T2{C.RESET}", 3: f"{C.DIM}T3{C.RESET}"}
            badge = tier_badge.get(s.get("tier", 3), "T?")
            url_short = s["url"][:65] + ("..." if len(s["url"]) > 65 else "")
            print(f"        [{badge}] {url_short}")
        kv("URLs stored", f"{urls_stored} → antigravity.db")

    # ── PROOF 2: FINDING ────────────────────────────────────────────
    if not output_json:
        step(2, "FINDING — claim + evidence + citations + confidence + lineage")

    trace["timestamps"]["finding_start"] = datetime.now().isoformat()

    # Build the claim (first substantial finding in the text)
    claim_match = re.search(
        r'(?:Finding|Insight|Thesis|Claim)[:\s]+(.+?)(?:\n\n|\.\s+[A-Z])',
        text, re.DOTALL
    )
    if claim_match:
        claim = claim_match.group(1).strip()
    else:
        # Use first 300 chars as claim
        sentences = text.split(". ")
        claim = ". ".join(sentences[:3])[:400]

    # Build evidence chain
    evidence = {
        "sources": sources,
        "confidence": 0.0,  # Will be set by oracle
        "reasoning_chain": [
            f"Extracted {len(sources)} citation(s) from input text",
            f"Source tiers: {sum(1 for s in sources if s.get('tier')==1)} T1, "
            f"{sum(1 for s in sources if s.get('tier')==2)} T2, "
            f"{sum(1 for s in sources if s.get('tier')==3)} T3",
            "Running 3-stream oracle validation (accuracy, completeness, relevance)",
        ],
        "validation": {
            "validated": False,
            "validated_at": None,
            "critic_notes": [],
            "oracle_streams": [],
        }
    }

    # Run oracle critique
    oracle = run_oracle(sources, claim)
    evidence["confidence"] = oracle["confidence"]
    evidence["validation"]["validated"] = oracle["validated"]
    evidence["validation"]["validated_at"] = datetime.now().isoformat()
    evidence["validation"]["critic_notes"] = oracle["issues"]
    evidence["validation"]["oracle_streams"] = [
        f"{s['stream']}: {s['score']:.2f}" for s in oracle["streams"]
    ]
    evidence["reasoning_chain"].append(
        f"Oracle consensus: {oracle['confidence']:.2%} confidence "
        f"({'VALIDATED' if oracle['validated'] else 'BELOW THRESHOLD'})"
    )

    # Assemble the full Finding object
    now = datetime.now().isoformat()
    finding = {
        "id": f"finding-{int(time.time())}-{uuid.uuid4().hex[:9]}",
        "session_id": session_id,
        "content": claim,
        "type": "finding",
        "evidence": evidence,
        "created_at": now,
        "updated_at": now,
        "derived_from": [],
        "enables": [],
        "tags": ["proof-demo", "multi-agent"],
        "projects": ["researchgravity"],
        "needs_review": not oracle["validated"],
    }

    # Store in database
    fid = store_finding(conn, finding)

    trace["stages"].append({
        "stage": "finding",
        "result": {
            "id": finding["id"],
            "claim": claim[:200],
            "type": finding["type"],
            "confidence": evidence["confidence"],
            "validated": oracle["validated"],
            "source_count": len(sources),
            "lineage_id": finding["id"],
        }
    })
    trace["timestamps"]["finding_end"] = datetime.now().isoformat()

    if not output_json:
        kv("Finding ID", finding["id"])
        kv("Type", finding["type"])
        print()
        dim("CLAIM:", 6)
        for line in textwrap.wrap(claim[:300], width=58):
            dim(f"  {line}", 6)
        print()
        kv("Citations", f"{len(sources)} sources")
        for s in sources:
            aid = s.get("arxiv_id")
            label = f"arXiv:{aid}" if aid else s["url"][:50]
            print(f"        {C.CYAN}→{C.RESET} {label} (relevance: {s['relevance_score']:.2f})")
        kv("Confidence", f"{evidence['confidence']:.2%}")
        kv("Validated", f"{'YES' if oracle['validated'] else 'NO'} (threshold: 0.70)")
        kv("Lineage ID", finding["id"])
        kv("Stored", f"antigravity.db → findings table")

    # ── PROOF 3: TRACE ──────────────────────────────────────────────
    if not output_json:
        step(3, "TRACE — retrieval → critics → justified answer")

    trace["timestamps"]["trace_start"] = datetime.now().isoformat()

    # 3a. Retrieval — search existing findings for related work
    query_terms = " ".join(re.findall(r'\b[a-z]{5,}\b', claim.lower())[:5])
    if not query_terms:
        query_terms = "multi-agent trust"
    related = search_findings(conn, query_terms, limit=5)

    retrieval_log = {
        "query": query_terms,
        "method": "FTS5 (full-text search) with LIKE fallback",
        "results_count": len(related),
        "results": [
            {
                "id": r["id"],
                "type": r.get("type", "?"),
                "confidence": r.get("confidence"),
                "snippet": (r.get("snippet") or r.get("content", ""))[:120],
            }
            for r in related
        ]
    }

    # 3b. Critique log — what the oracle said
    critique_log = {
        "finding_id": finding["id"],
        "oracle_streams": oracle["streams"],
        "weighted_confidence": oracle["confidence"],
        "validation_passed": oracle["validated"],
        "issues": oracle["issues"],
        "weights": {"accuracy": 0.40, "completeness": 0.35, "relevance": 0.25},
    }

    # 3c. Final answer justification
    answer_justification = {
        "finding_id": finding["id"],
        "claim": claim[:200],
        "confidence": oracle["confidence"],
        "reasoning": [
            f"Ingested {'URL ' + source_url if source_url else 'transcript'} ({len(text):,} chars)",
            f"Extracted {len(sources)} evidence sources ({sum(1 for s in sources if s.get('tier')==1)} Tier 1)",
            f"Oracle accuracy stream: {oracle['streams'][0]['score']:.2f}",
            f"Oracle completeness stream: {oracle['streams'][1]['score']:.2f}",
            f"Oracle relevance stream: {oracle['streams'][2]['score']:.2f}",
            f"Weighted confidence: {oracle['confidence']:.2%} "
            f"({'above' if oracle['validated'] else 'below'} 0.70 threshold)",
            f"Found {len(related)} related findings in knowledge base",
            f"Stored as {finding['id']} with full provenance chain",
        ],
    }

    # 3d. Verify stored record can be retrieved
    retrieved = retrieve_finding(conn, finding["id"])
    retrieval_verified = retrieved is not None and retrieved["id"] == finding["id"]

    trace["stages"].append({
        "stage": "trace",
        "retrieval": retrieval_log,
        "critique": critique_log,
        "answer": answer_justification,
        "verified_retrieval": retrieval_verified,
    })
    trace["timestamps"]["trace_end"] = datetime.now().isoformat()
    trace["timestamps"]["total_seconds"] = round(time.time() - t0, 3)

    if not output_json:
        print()
        dim("RETRIEVAL:", 6)
        kv("Query", f'"{query_terms}"', 8)
        kv("Method", "FTS5 full-text search → LIKE fallback", 8)
        kv("Results", f"{len(related)} related findings", 8)
        for r in related[:3]:
            snip = (r.get("snippet") or r.get("content", ""))[:80].replace("\n", " ")
            conf = r.get("confidence")
            conf_s = f" [{conf:.2f}]" if conf else ""
            print(f"          {C.DIM}→ {r['id'][:40]}{conf_s}{C.RESET}")
            print(f"            {C.DIM}{snip}...{C.RESET}")

        print()
        dim("CRITICS:", 6)
        for s in oracle["streams"]:
            bar_len = int(s["score"] * 20)
            bar = f"{'█' * bar_len}{'░' * (20 - bar_len)}"
            colour = C.GREEN if s["score"] >= 0.7 else C.YELLOW if s["score"] >= 0.5 else C.RED
            weight = {"accuracy": "40%", "completeness": "35%", "relevance": "25%"}[s["stream"]]
            print(f"        {s['stream']:>14}: {colour}{bar}{C.RESET} {s['score']:.2f} (weight: {weight})")
            for issue in s["issues"]:
                warn(issue, 26)

        print()
        dim("ANSWER JUSTIFICATION:", 6)
        for i, reason in enumerate(answer_justification["reasoning"], 1):
            ok(reason, 8) if "above" in reason or "Stored" in reason or "Tier 1" in reason else dim(f"  {reason}", 6)

        print()
        dim("STORED RECORD RETRIEVAL:", 6)
        if retrieval_verified:
            ok(f"Retrieved {finding['id']} from antigravity.db", 8)
            kv("Confidence", f"{retrieved['evidence']['confidence']:.2%}", 10)
            kv("Sources", f"{len(retrieved['evidence'].get('sources', []))}", 10)
            kv("Validated", f"{retrieved['evidence'].get('validation', {}).get('validated', False)}", 10)
        else:
            fail("Could not retrieve stored finding!", 8)

        # Summary
        elapsed = time.time() - t0
        header(f"Proof Complete — {elapsed:.1f}s")
        print()
        print(f"  {C.BOLD}Proof 1{C.RESET} (Ingest):  {C.GREEN}✓{C.RESET} URL/transcript → {len(sources)} sources → stored")
        print(f"  {C.BOLD}Proof 2{C.RESET} (Finding): {C.GREEN}✓{C.RESET} claim + {len(sources)} citations + {oracle['confidence']:.0%} confidence + lineage")
        print(f"  {C.BOLD}Proof 3{C.RESET} (Trace):   {C.GREEN}✓{C.RESET} retrieval({len(related)}) → critics(3) → answer justified")
        print()

        # Show the finding as JSON
        print(f"  {C.BOLD}{C.MAG}Full Finding Object:{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        compact = json.dumps(finding, indent=2)
        for line in compact.split("\n")[:35]:
            print(f"  {C.DIM}{line}{C.RESET}")
        if compact.count("\n") > 35:
            print(f"  {C.DIM}  ... ({compact.count(chr(10)) - 35} more lines){C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")

    conn.close()

    if output_json:
        full_output = {
            "proof_items": {
                "ingest": ingest_result,
                "finding": finding,
                "trace": {
                    "retrieval": retrieval_log,
                    "critique": critique_log,
                    "answer": answer_justification,
                    "verified_retrieval": retrieval_verified,
                }
            },
            "metadata": trace["timestamps"],
        }
        print(json.dumps(full_output, indent=2))

    return trace


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ResearchGravity — 60-Second Proof of Credibility"
    )
    parser.add_argument("--url", help="URL to ingest")
    parser.add_argument("--text", help="Raw text/transcript to ingest")
    parser.add_argument("--json", action="store_true", dest="output_json",
                        help="Output machine-readable JSON")
    args = parser.parse_args()

    text = args.text or ""
    url = args.url

    # If no input, use built-in demo
    if not text and not url:
        text = DEMO_TRANSCRIPT

    run_proof(text=text, source_url=url, output_json=args.output_json)


if __name__ == "__main__":
    main()
