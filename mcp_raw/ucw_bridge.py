"""
UCW Bridge — Extract semantic layers from MCP messages

Every message gets tagged with:
  Data:     What was said (raw information)
  Light:    What it means (insights, meaning)
  Instinct: What it signals (patterns, emergence)
"""

import hashlib
from typing import Dict, List, Tuple, Any


# Keywords by domain for topic extraction
_DOMAIN_KEYWORDS = {
    "mcp_protocol": ["mcp", "protocol", "stdio", "json-rpc", "transport"],
    "database": ["database", "sql", "schema", "query", "postgres", "sqlite"],
    "ucw": ["ucw", "cognitive wallet", "coherence", "sovereignty"],
    "ai_agents": ["agent", "multi-agent", "orchestrat", "coordinat"],
    "research": ["research", "paper", "arxiv", "finding", "hypothesis"],
    "coding": ["function", "class", "import", "variable", "refactor", "debug"],
}

_INTENT_SIGNALS = {
    "search":   ["search", "find", "look", "where"],
    "create":   ["create", "build", "write", "make", "generate"],
    "analyze":  ["analyze", "review", "check", "explain", "why"],
    "retrieve": ["get", "read", "list", "show", "fetch"],
    "execute":  ["call", "run", "execute", "invoke"],
}


def extract_layers(msg: Dict[str, Any], direction: str) -> Tuple[Dict, Dict, Dict]:
    """Extract UCW Data / Light / Instinct layers from an MCP message."""
    data = _data_layer(msg, direction)
    light = _light_layer(msg, data)
    instinct = _instinct_layer(light)
    return data, light, instinct


def coherence_signature(intent: str, topic: str, timestamp_ns: int, content: str) -> str:
    """SHA-256 coherence signature for cross-platform matching (5-min buckets)."""
    bucket = timestamp_ns // (5 * 60 * 1_000_000_000)
    blob = f"{intent}::{topic}::{bucket}::{content[:1024]}"
    return hashlib.sha256(blob.encode()).hexdigest()


# ── internal helpers ──────────────────────────────────────────────────────────

def _data_layer(msg: Dict, direction: str) -> Dict:
    method = msg.get("method", "")
    params = msg.get("params", {})
    result = msg.get("result", {})

    if direction == "in":
        if method == "tools/call":
            content = f"Tool call: {params.get('name', '')} | args={params.get('arguments', {})}"
        elif method.endswith("/list"):
            content = f"List {method.split('/')[0]}"
        elif method.endswith("/read"):
            content = f"Read resource: {params.get('uri', '')}"
        else:
            content = f"Method: {method}"
    else:
        if "error" in msg:
            content = f"Error: {msg['error'].get('message', '')}"
        elif isinstance(result, dict) and "content" in result:
            parts = result.get("content", [])
            content = " ".join(
                p.get("text", "")[:500] for p in parts if isinstance(p, dict)
            )[:2000]
        else:
            content = str(result)[:2000]

    return {
        "method": method,
        "params": params,
        "result": result,
        "content": content,
        "tokens_est": max(1, len(content) // 4),
    }


def _light_layer(msg: Dict, data: Dict) -> Dict:
    content = data.get("content", "")
    method = data.get("method", "")
    cl = content.lower()

    intent = _classify(cl, _INTENT_SIGNALS, default="explore")
    topic = _classify(cl, _DOMAIN_KEYWORDS, default="general")
    concepts = _extract_concepts(cl)

    return {
        "intent": intent,
        "topic": topic,
        "concepts": concepts,
        "summary": content[:200],
    }


def _instinct_layer(light: Dict) -> Dict:
    concepts = light.get("concepts", [])
    topic = light.get("topic", "general")

    # Coherence potential: more concepts + non-general topic = higher
    cp = 0.0
    if topic != "general":
        cp += 0.35
    if light.get("intent") in ("create", "analyze", "search"):
        cp += 0.25
    cp += min(len(concepts) * 0.1, 0.4)
    cp = min(cp, 1.0)

    indicators: List[str] = []
    if cp > 0.7:
        indicators.append("high_coherence_potential")
    if len(concepts) >= 3:
        indicators.append("concept_cluster")
    meta_terms = {"coherence", "cognitive", "emergence", "unify", "sovereign"}
    if meta_terms & set(concepts):
        indicators.append("meta_cognitive")

    return {
        "coherence_potential": round(cp, 3),
        "emergence_indicators": indicators,
        "gut_signal": (
            "breakthrough_potential" if len(indicators) >= 2
            else "interesting" if indicators
            else "routine"
        ),
    }


def _classify(text: str, mapping: Dict[str, List[str]], *, default: str) -> str:
    best, best_score = default, 0
    for label, keywords in mapping.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best, best_score = label, score
    return best


def _extract_concepts(text: str) -> List[str]:
    targets = [
        "mcp", "ucw", "database", "schema", "coherence", "protocol",
        "cognitive", "semantic", "embedding", "sovereign", "platform",
        "research", "session", "capture", "agent", "orchestrat",
    ]
    return [t for t in targets if t in text]
