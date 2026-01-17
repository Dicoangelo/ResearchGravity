"""
UCW Schema — Core data models for Universal Cognitive Wallet

The UCW format is a portable, platform-agnostic representation of a user's
AI interaction history, designed for:
- Cross-platform portability (Claude, GPT, Gemini, local LLMs)
- Economic valuation (appreciation based on knowledge density)
- Integrity verification (cryptographic signing)
- User sovereignty (you own your cognitive capital)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import hashlib
import json


class ConceptType(Enum):
    """Types of concepts that can be captured."""
    FINDING = "finding"
    THESIS = "thesis"
    GAP = "gap"
    INNOVATION = "innovation"
    PAPER = "paper"
    TOOL = "tool"
    INSIGHT = "insight"


class ConnectionType(Enum):
    """Types of relationships between concepts."""
    ENABLES = "enables"
    INFORMS = "informs"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    REFERENCES = "references"
    DERIVES_FROM = "derives_from"


@dataclass
class URL:
    """A captured URL with metadata."""
    url: str
    tier: int  # 1-3 (research priority)
    category: str  # research, github, industry, etc.
    source: str  # arXiv, GitHub, TechCrunch, etc.
    context: str  # Surrounding text when captured
    captured_at: datetime
    relevance: Optional[int] = None  # 1-5
    signal: Optional[str] = None  # "177k stars", etc.

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "tier": self.tier,
            "category": self.category,
            "source": self.source,
            "context": self.context,
            "captured_at": self.captured_at.isoformat(),
            "relevance": self.relevance,
            "signal": self.signal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "URL":
        return cls(
            url=data["url"],
            tier=data.get("tier", 3),
            category=data.get("category", "other"),
            source=data.get("source", "Web"),
            context=data.get("context", ""),
            captured_at=datetime.fromisoformat(data["captured_at"]) if data.get("captured_at") else datetime.now(),
            relevance=data.get("relevance"),
            signal=data.get("signal"),
        )


@dataclass
class Connection:
    """A relationship between two concepts."""
    from_id: str
    to_id: str
    connection_type: ConnectionType
    strength: float  # 0-1
    context: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "type": self.connection_type.value,
            "strength": self.strength,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Connection":
        return cls(
            from_id=data["from_id"],
            to_id=data["to_id"],
            connection_type=ConnectionType(data["type"]),
            strength=data.get("strength", 0.5),
            context=data.get("context"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


@dataclass
class Concept:
    """A unit of knowledge in the cognitive wallet."""
    id: str
    content: str
    concept_type: ConceptType
    confidence: float  # 0-1
    sources: List[str]  # arXiv IDs, session IDs, URLs
    connections: List[str] = field(default_factory=list)  # Connection IDs
    created_at: datetime = field(default_factory=datetime.now)
    domain: Optional[str] = None  # AI/ML, SWE, Product, etc.
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.concept_type.value,
            "confidence": self.confidence,
            "sources": self.sources,
            "connections": self.connections,
            "created_at": self.created_at.isoformat(),
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Concept":
        return cls(
            id=data["id"],
            content=data["content"],
            concept_type=ConceptType(data["type"]),
            confidence=data.get("confidence", 0.5),
            sources=data.get("sources", []),
            connections=data.get("connections", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            domain=data.get("domain"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """A captured AI interaction session."""
    id: str
    topic: str
    date: datetime
    findings: List[str] = field(default_factory=list)  # Concept IDs
    papers: List[str] = field(default_factory=list)  # arXiv IDs
    urls: List[URL] = field(default_factory=list)
    project: Optional[str] = None
    status: str = "archived"
    transcript_hash: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "date": self.date.isoformat(),
            "findings": self.findings,
            "papers": self.papers,
            "urls": [u.to_dict() for u in self.urls],
            "project": self.project,
            "status": self.status,
            "transcript_hash": self.transcript_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            topic=data["topic"],
            date=datetime.fromisoformat(data["date"]) if data.get("date") else datetime.now(),
            findings=data.get("findings", []),
            papers=data.get("papers", []),
            urls=[URL.from_dict(u) for u in data.get("urls", [])],
            project=data.get("project"),
            status=data.get("status", "archived"),
            transcript_hash=data.get("transcript_hash"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ValueMetrics:
    """Economic metrics for the cognitive wallet."""
    total_value: float  # Estimated USD value
    concept_count: int
    connection_count: int
    session_count: int
    paper_count: int
    url_count: int
    domains: Dict[str, float] = field(default_factory=dict)  # Domain weights
    appreciation_rate: float = 0.03  # Per-session appreciation
    last_calculated: datetime = field(default_factory=datetime.now)
    history: List[Dict] = field(default_factory=list)  # Value over time

    def to_dict(self) -> dict:
        return {
            "total_value": self.total_value,
            "concept_count": self.concept_count,
            "connection_count": self.connection_count,
            "session_count": self.session_count,
            "paper_count": self.paper_count,
            "url_count": self.url_count,
            "domains": self.domains,
            "appreciation_rate": self.appreciation_rate,
            "last_calculated": self.last_calculated.isoformat(),
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValueMetrics":
        return cls(
            total_value=data.get("total_value", 0.0),
            concept_count=data.get("concept_count", 0),
            connection_count=data.get("connection_count", 0),
            session_count=data.get("session_count", 0),
            paper_count=data.get("paper_count", 0),
            url_count=data.get("url_count", 0),
            domains=data.get("domains", {}),
            appreciation_rate=data.get("appreciation_rate", 0.03),
            last_calculated=datetime.fromisoformat(data["last_calculated"]) if data.get("last_calculated") else datetime.now(),
            history=data.get("history", []),
        )


@dataclass
class CognitiveWallet:
    """
    The Universal Cognitive Wallet — portable, appreciating, tradeable AI memory.

    This is the core data structure that represents a user's cognitive capital:
    - Sessions: AI interaction history
    - Concepts: Extracted knowledge units
    - Connections: Relationships between concepts
    - Papers: Research papers referenced
    - Value: Economic metrics

    The wallet can be exported to a platform-agnostic format and imported
    into any compatible AI platform.
    """
    version: str = "1.0"
    owner_did: Optional[str] = None  # Decentralized identifier
    created: datetime = field(default_factory=datetime.now)
    concepts: Dict[str, Concept] = field(default_factory=dict)
    sessions: Dict[str, Session] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    papers: Dict[str, Dict] = field(default_factory=dict)  # arXiv ID -> metadata
    value_metrics: ValueMetrics = field(default_factory=lambda: ValueMetrics(
        total_value=0.0,
        concept_count=0,
        connection_count=0,
        session_count=0,
        paper_count=0,
        url_count=0,
    ))
    integrity_hash: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def calculate_integrity_hash(self) -> str:
        """Calculate SHA-256 hash of wallet contents for integrity verification."""
        content = json.dumps({
            "version": self.version,
            "owner_did": self.owner_did,
            "created": self.created.isoformat(),
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "connections": [c.to_dict() for c in self.connections],
            "papers": self.papers,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def update_integrity_hash(self):
        """Update the integrity hash based on current contents."""
        self.integrity_hash = self.calculate_integrity_hash()

    def verify_integrity(self) -> bool:
        """Verify wallet integrity by checking hash."""
        if not self.integrity_hash:
            return True  # No hash set, considered valid
        return self.integrity_hash == self.calculate_integrity_hash()

    def to_dict(self) -> dict:
        """Export wallet to dictionary for JSON serialization."""
        return {
            "ucw_version": self.version,
            "owner_did": self.owner_did,
            "created": self.created.isoformat(),
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "connections": [c.to_dict() for c in self.connections],
            "papers": self.papers,
            "value_metrics": self.value_metrics.to_dict(),
            "integrity_hash": self.integrity_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CognitiveWallet":
        """Import wallet from dictionary."""
        wallet = cls(
            version=data.get("ucw_version", "1.0"),
            owner_did=data.get("owner_did"),
            created=datetime.fromisoformat(data["created"]) if data.get("created") else datetime.now(),
            concepts={k: Concept.from_dict(v) for k, v in data.get("concepts", {}).items()},
            sessions={k: Session.from_dict(v) for k, v in data.get("sessions", {}).items()},
            connections=[Connection.from_dict(c) for c in data.get("connections", [])],
            papers=data.get("papers", {}),
            value_metrics=ValueMetrics.from_dict(data["value_metrics"]) if data.get("value_metrics") else ValueMetrics(
                total_value=0.0, concept_count=0, connection_count=0,
                session_count=0, paper_count=0, url_count=0
            ),
            integrity_hash=data.get("integrity_hash"),
            metadata=data.get("metadata", {}),
        )
        return wallet

    def get_stats(self) -> Dict:
        """Get summary statistics for the wallet."""
        total_urls = sum(len(s.urls) for s in self.sessions.values())
        return {
            "sessions": len(self.sessions),
            "concepts": len(self.concepts),
            "connections": len(self.connections),
            "papers": len(self.papers),
            "urls": total_urls,
            "value": self.value_metrics.total_value,
            "domains": self.value_metrics.domains,
        }
