"""
Cognitive Appreciation Engine — Calculate the economic value of a cognitive wallet.

This is a NOVEL invention (verified against arXiv 2025 prior art):
- Data markets price RAW DATA by volume
- We price SYNTHESIZED KNOWLEDGE by:
  - Concept count (knowledge density)
  - Connection count (relationship depth)
  - Domain concentration (specificity premium)
  - Time accumulation (compounding)
  - Quality/verification (verified sources)

Patent Claim: "A system for calculating the economic value of accumulated
AI interactions based on knowledge density, interconnection depth,
domain specificity, and compounding mechanics."
"""

from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re

from .schema import CognitiveWallet, Concept


# Value constants (tunable)
BASE_CONCEPT_VALUE = 0.50  # USD per concept
CONNECTION_MULTIPLIER = 0.001  # Per connection
SPECIFICITY_BONUS = 0.5  # Max bonus for domain concentration
TIME_COMPOUND_RATE = 0.001  # Per day
QUALITY_BASE = 0.5  # Minimum quality multiplier
PAPER_VALUE = 2.00  # USD per indexed paper
URL_VALUE = 0.05  # USD per captured URL


@dataclass
class ValueBreakdown:
    """Detailed breakdown of wallet value calculation."""
    base_value: float
    concept_value: float
    connection_premium: float
    specificity_premium: float
    time_premium: float
    quality_premium: float
    paper_value: float
    url_value: float
    total_value: float
    appreciation_rate: float
    domains: Dict[str, float]
    top_concepts: List[Tuple[str, float]]


class CognitiveAppreciationEngine:
    """
    Calculate the economic value of accumulated AI interactions.

    The value model is based on:
    1. Knowledge density (concept count)
    2. Relationship depth (connection count, Metcalfe's law)
    3. Domain specificity (concentration premium)
    4. Time compounding (value grows with history)
    5. Quality (verified sources, arXiv papers)
    """

    def __init__(
        self,
        base_concept_value: float = BASE_CONCEPT_VALUE,
        connection_multiplier: float = CONNECTION_MULTIPLIER,
        specificity_bonus: float = SPECIFICITY_BONUS,
        time_compound_rate: float = TIME_COMPOUND_RATE,
        quality_base: float = QUALITY_BASE,
        paper_value: float = PAPER_VALUE,
        url_value: float = URL_VALUE,
    ):
        self.base_concept_value = base_concept_value
        self.connection_multiplier = connection_multiplier
        self.specificity_bonus = specificity_bonus
        self.time_compound_rate = time_compound_rate
        self.quality_base = quality_base
        self.paper_value = paper_value
        self.url_value = url_value

    def calculate_value(self, wallet: CognitiveWallet) -> float:
        """Calculate total wallet value."""
        breakdown = self.calculate_breakdown(wallet)
        return breakdown.total_value

    def calculate_breakdown(self, wallet: CognitiveWallet) -> ValueBreakdown:
        """Calculate detailed value breakdown."""

        # Count URLs across all sessions
        total_urls = sum(len(s.urls) for s in wallet.sessions.values())

        # Base value from concepts
        concept_count = len(wallet.concepts)
        concept_value = concept_count * self.base_concept_value

        # Paper value
        paper_count = len(wallet.papers)
        paper_value = paper_count * self.paper_value

        # URL value
        url_value = total_urls * self.url_value

        # Base value (before multipliers)
        base_value = concept_value + paper_value + url_value

        # Connection premium (Metcalfe's law - value grows with connections)
        connection_count = len(wallet.connections)
        connection_premium = 1 + (connection_count * self.connection_multiplier)

        # Domain specificity premium
        domains = self.analyze_domains(wallet)
        top_domain_weight = max(domains.values()) if domains else 0
        specificity_premium = 1 + (top_domain_weight * self.specificity_bonus)

        # Time compounding (value grows with history)
        days_active = (datetime.now() - wallet.created).days
        time_premium = 1 + (days_active * self.time_compound_rate)

        # Quality adjustment (verified sources)
        quality_premium = self._calculate_quality_multiplier(wallet)

        # Total value
        total_value = (
            base_value
            * connection_premium
            * specificity_premium
            * time_premium
            * quality_premium
        )

        # Appreciation rate (based on recent growth)
        appreciation_rate = self._calculate_appreciation_rate(wallet)

        # Top concepts by value
        top_concepts = self._rank_concepts(wallet)

        return ValueBreakdown(
            base_value=base_value,
            concept_value=concept_value,
            connection_premium=connection_premium,
            specificity_premium=specificity_premium,
            time_premium=time_premium,
            quality_premium=quality_premium,
            paper_value=paper_value,
            url_value=url_value,
            total_value=round(total_value, 2),
            appreciation_rate=round(appreciation_rate, 4),
            domains=domains,
            top_concepts=top_concepts[:10],
        )

    def analyze_domains(self, wallet: CognitiveWallet) -> Dict[str, float]:
        """Analyze domain distribution of concepts."""
        domain_counts: Dict[str, int] = {}
        total = 0

        for concept in wallet.concepts.values():
            domain = concept.domain or self._infer_domain(concept)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            total += 1

        if total == 0:
            return {}

        # Convert to weights (0-1)
        return {k: v / total for k, v in domain_counts.items()}

    def _infer_domain(self, concept: Concept) -> str:
        """Infer domain from concept content."""
        content_lower = concept.content.lower()

        # Simple keyword-based inference
        if any(kw in content_lower for kw in ["arxiv", "paper", "research", "llm", "agent", "model"]):
            return "AI/ML"
        elif any(kw in content_lower for kw in ["code", "github", "api", "function", "class"]):
            return "Software Engineering"
        elif any(kw in content_lower for kw in ["product", "user", "feature", "design"]):
            return "Product"
        elif any(kw in content_lower for kw in ["market", "revenue", "business", "customer"]):
            return "Business"
        else:
            return "General"

    def _calculate_quality_multiplier(self, wallet: CognitiveWallet) -> float:
        """Calculate quality multiplier based on verified sources."""
        if not wallet.concepts:
            return self.quality_base

        # Count concepts with verified sources (arXiv papers)
        arxiv_pattern = re.compile(r'\d{4}\.\d{4,5}')
        verified = sum(
            1 for c in wallet.concepts.values()
            if any(arxiv_pattern.search(s) for s in c.sources)
        )

        verified_ratio = verified / len(wallet.concepts)
        return self.quality_base + (verified_ratio * (1 - self.quality_base))

    def _calculate_appreciation_rate(self, wallet: CognitiveWallet) -> float:
        """Calculate recent appreciation rate."""
        history = wallet.value_metrics.history
        if len(history) < 2:
            return 0.03  # Default 3% per session

        # Calculate average daily appreciation from last 7 entries
        recent = history[-7:]
        if len(recent) < 2:
            return 0.03

        first_value = recent[0].get("value", 0)
        last_value = recent[-1].get("value", 0)

        if first_value == 0:
            return 0.03

        total_change = (last_value - first_value) / first_value
        days = len(recent)
        return total_change / days if days > 0 else 0.03

    def _rank_concepts(self, wallet: CognitiveWallet) -> List[Tuple[str, float]]:
        """Rank concepts by value contribution."""
        ranked = []

        for concept_id, concept in wallet.concepts.items():
            # Value based on connections and sources
            connection_count = len(concept.connections)
            source_count = len(concept.sources)
            confidence = concept.confidence

            value = (
                self.base_concept_value
                * (1 + connection_count * 0.1)
                * (1 + source_count * 0.05)
                * confidence
            )
            ranked.append((concept.content[:50], round(value, 2)))

        return sorted(ranked, key=lambda x: -x[1])

    def project_value(
        self,
        wallet: CognitiveWallet,
        days_ahead: int,
        sessions_per_day: float = 0.5,
    ) -> float:
        """Project future wallet value."""
        current = self.calculate_value(wallet)
        appreciation = wallet.value_metrics.appreciation_rate

        # Compound appreciation
        future_value = current * ((1 + appreciation) ** (days_ahead * sessions_per_day))
        return round(future_value, 2)

    def update_wallet_metrics(self, wallet: CognitiveWallet) -> None:
        """Update wallet's value metrics with current calculation."""
        breakdown = self.calculate_breakdown(wallet)
        total_urls = sum(len(s.urls) for s in wallet.sessions.values())

        # Record current value in history
        wallet.value_metrics.history.append({
            "timestamp": datetime.now().isoformat(),
            "value": breakdown.total_value,
            "concepts": len(wallet.concepts),
            "sessions": len(wallet.sessions),
        })

        # Update metrics
        wallet.value_metrics.total_value = breakdown.total_value
        wallet.value_metrics.concept_count = len(wallet.concepts)
        wallet.value_metrics.connection_count = len(wallet.connections)
        wallet.value_metrics.session_count = len(wallet.sessions)
        wallet.value_metrics.paper_count = len(wallet.papers)
        wallet.value_metrics.url_count = total_urls
        wallet.value_metrics.domains = breakdown.domains
        wallet.value_metrics.appreciation_rate = breakdown.appreciation_rate
        wallet.value_metrics.last_calculated = datetime.now()


def calculate_value(wallet: CognitiveWallet) -> float:
    """Convenience function to calculate wallet value."""
    engine = CognitiveAppreciationEngine()
    return engine.calculate_value(wallet)


def get_value_breakdown(wallet: CognitiveWallet) -> ValueBreakdown:
    """Convenience function to get detailed value breakdown."""
    engine = CognitiveAppreciationEngine()
    return engine.calculate_breakdown(wallet)


def format_value_display(wallet: CognitiveWallet) -> str:
    """Format wallet value for display."""
    engine = CognitiveAppreciationEngine()
    breakdown = engine.calculate_breakdown(wallet)

    lines = [
        "",
        "  COGNITIVE WALLET VALUE",
        "  ═══════════════════════════════════════",
        "",
        f"  Total Value:     ${breakdown.total_value:,.2f}",
        "",
        "  Breakdown:",
        f"    Concepts ({len(wallet.concepts)}):    ${breakdown.concept_value:,.2f}",
        f"    Papers ({len(wallet.papers)}):      ${breakdown.paper_value:,.2f}",
        f"    URLs:              ${breakdown.url_value:,.2f}",
        "",
        "  Multipliers:",
        f"    Connections:       {breakdown.connection_premium:.2f}x",
        f"    Domain Focus:      {breakdown.specificity_premium:.2f}x",
        f"    Time Compounding:  {breakdown.time_premium:.2f}x",
        f"    Quality:           {breakdown.quality_premium:.2f}x",
        "",
    ]

    if breakdown.domains:
        lines.append("  Domains:")
        for domain, weight in sorted(breakdown.domains.items(), key=lambda x: -x[1]):
            pct = weight * 100
            lines.append(f"    {domain}: {pct:.0f}%")
        lines.append("")

    lines.extend([
        f"  Appreciation Rate: {breakdown.appreciation_rate * 100:.1f}% per session",
        "",
        "  ═══════════════════════════════════════",
        "",
    ])

    return "\n".join(lines)
