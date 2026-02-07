"""
Base Normalizer — UCW layer extraction for external platform events.

Produces cognitive_events rows from CapturedEvents, reusing
ucw_bridge.py patterns for coherence signatures and semantic layers.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Tuple

from .base import CapturedEvent


# Domain keywords — extended from mcp_raw/ucw_bridge.py with external platform terms
_DOMAIN_KEYWORDS = {
    "mcp_protocol": ["mcp", "protocol", "stdio", "json-rpc", "transport"],
    "database": ["database", "sql", "schema", "query", "postgres", "sqlite"],
    "ucw": ["ucw", "cognitive wallet", "coherence", "sovereignty"],
    "ai_agents": ["agent", "multi-agent", "orchestrat", "coordinat"],
    "research": ["research", "paper", "arxiv", "finding", "hypothesis"],
    "coding": ["function", "class", "import", "variable", "refactor", "debug"],
    "career": ["career", "resume", "interview", "job", "hiring", "salary"],
    "product": ["product", "feature", "roadmap", "mvp", "launch", "market"],
    "philosophy": ["philosophy", "consciousness", "meaning", "existence", "ethics"],
}

_INTENT_SIGNALS = {
    "search":   ["search", "find", "look", "where"],
    "create":   ["create", "build", "write", "make", "generate"],
    "analyze":  ["analyze", "review", "check", "explain", "why"],
    "retrieve": ["get", "read", "list", "show", "fetch"],
    "execute":  ["call", "run", "execute", "invoke"],
    "discuss":  ["think", "consider", "debate", "opinion", "should"],
    "learn":    ["learn", "understand", "teach", "how does"],
}

_CONCEPT_TARGETS = [
    "mcp", "ucw", "database", "schema", "coherence", "protocol",
    "cognitive", "semantic", "embedding", "sovereign", "platform",
    "research", "session", "capture", "agent", "orchestrat",
    "career", "product", "architecture", "infrastructure",
]


class BaseNormalizer:
    """Convert CapturedEvent to cognitive_events row dict with UCW layers."""

    def to_cognitive_event(self, captured: CapturedEvent, session_topic: str = "") -> dict:
        """
        Produce a dict matching the cognitive_events schema.
        Ready for INSERT into PostgreSQL.
        """
        data, light, instinct = self.extract_ucw_layers(
            captured.content, captured.role, captured.platform, session_topic,
        )

        coherence_sig = self.make_coherence_sig(
            light["intent"], light["topic"],
            captured.timestamp_ns, captured.content,
        )

        direction = "in" if captured.role in ("user", "human") else "out"

        return {
            "event_id": captured.event_id,
            "session_id": captured.session_id,
            "timestamp_ns": captured.timestamp_ns,
            "direction": direction,
            "stage": "captured",
            "method": f"{captured.platform}.message",
            "request_id": None,
            "parent_event_id": None,
            "turn": 0,
            "raw_bytes": None,
            "parsed_json": json.dumps(captured.metadata, default=str),
            "content_length": len(captured.content),
            "error": None,
            "data_layer": json.dumps(data),
            "light_layer": json.dumps(light),
            "instinct_layer": json.dumps(instinct),
            "coherence_sig": coherence_sig,
            "platform": captured.platform,
            "protocol": self._protocol_for(captured.platform),
            "quality_score": captured.quality_score,
            "cognitive_mode": captured.cognitive_mode,
        }

    def extract_ucw_layers(
        self,
        content: str,
        role: str,
        platform: str,
        session_topic: str = "",
    ) -> Tuple[Dict, Dict, Dict]:
        """Extract Data / Light / Instinct layers from external platform message."""
        data = self._data_layer(content, role, platform)
        light = self._light_layer(content, session_topic)
        instinct = self._instinct_layer(light)
        return data, light, instinct

    def make_coherence_sig(
        self, intent: str, topic: str, timestamp_ns: int, content: str,
    ) -> str:
        """SHA-256 coherence signature for cross-platform matching (5-min buckets)."""
        bucket = timestamp_ns // (5 * 60 * 1_000_000_000)
        blob = f"{intent}::{topic}::{bucket}::{content[:1024]}"
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def generate_event_id(platform: str, content_hash: str) -> str:
        return f"{platform}-{content_hash[:12]}"

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ── internal ───────────────────────────────────────────

    def _data_layer(self, content: str, role: str, platform: str) -> dict:
        return {
            "method": f"{platform}.message",
            "params": {"role": role, "platform": platform},
            "content": content[:2000],
            "tokens_est": max(1, len(content) // 4),
        }

    def _light_layer(self, content: str, session_topic: str = "") -> dict:
        cl = content.lower()
        intent = _classify(cl, _INTENT_SIGNALS, default="explore")
        topic = _classify(cl, _DOMAIN_KEYWORDS, default="general")
        concepts = _extract_concepts(cl)
        summary = content[:200]

        if session_topic and topic == "general":
            # Use session title as fallback topic hint
            topic_hint = _classify(session_topic.lower(), _DOMAIN_KEYWORDS, default="general")
            if topic_hint != "general":
                topic = topic_hint

        return {
            "intent": intent,
            "topic": topic,
            "concepts": concepts,
            "summary": summary,
        }

    def _instinct_layer(self, light: dict) -> dict:
        concepts = light.get("concepts", [])
        topic = light.get("topic", "general")

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

    def _protocol_for(self, platform: str) -> str:
        return {
            "chatgpt": "openai-api",
            "cursor": "cursor-local",
            "grok": "x-api",
        }.get(platform, "external")


# ── shared helpers (match ucw_bridge.py) ────────────────────

def _classify(text: str, mapping: Dict[str, List[str]], *, default: str) -> str:
    best, best_score = default, 0
    for label, keywords in mapping.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best, best_score = label, score
    return best


def _extract_concepts(text: str) -> List[str]:
    return [t for t in _CONCEPT_TARGETS if t in text]
