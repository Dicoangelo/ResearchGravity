"""
Tiered Search Layer for CPB Precision Mode v2

Implements ResearchGravity's 3-tier signal discovery methodology:
- Tier 1: Primary sources (arXiv, labs, industry news)
- Tier 2: Amplifiers (GitHub, benchmarks, social)
- Tier 3: Context (internal learnings, newsletters)

Combined with time-decay scoring and signal quantification.
"""

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

# Optional imports for API access
try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class SourceTier(Enum):
    """ResearchGravity source hierarchy."""
    TIER_1 = 1  # Primary: arXiv, labs, industry
    TIER_2 = 2  # Amplifiers: GitHub, benchmarks, social
    TIER_3 = 3  # Context: internal, newsletters


class SourceCategory(Enum):
    """Source categories for filtering."""
    RESEARCH = "research"      # arXiv, papers
    LAB = "lab"                # OpenAI, Anthropic, Google AI
    INDUSTRY = "industry"      # TechCrunch, Verge
    GITHUB = "github"          # Repositories
    BENCHMARK = "benchmark"    # METR, ARC Prize
    SOCIAL = "social"          # Twitter/X
    INTERNAL = "internal"      # Your learnings


@dataclass
class SearchResult:
    """A single search result with signal quantification."""
    url: str
    title: str
    content: str  # Abstract, description, or excerpt

    # Source metadata
    tier: SourceTier
    category: SourceCategory
    source_name: str  # e.g., "arXiv", "GitHub", "TechCrunch"

    # Signal quantification (ResearchGravity methodology)
    published_date: Optional[datetime] = None
    citations: Optional[int] = None      # For papers
    stars: Optional[int] = None          # For GitHub
    engagement: Optional[int] = None     # For social

    # Computed scores
    base_relevance: float = 0.0          # From search/embedding
    time_decay_score: float = 0.0        # After recency adjustment
    final_score: float = 0.0             # tier_weight × time_decay_score

    # For citation formatting
    signal_string: str = ""              # e.g., "★2.3k, 7 days old"

    def __post_init__(self):
        self._compute_signal_string()
        self._compute_time_decay()
        self._compute_final_score()

    def _compute_signal_string(self):
        """Generate quantitative signal string for citation."""
        signals = []

        if self.stars is not None:
            if self.stars >= 1000:
                signals.append(f"★{self.stars/1000:.1f}k")
            else:
                signals.append(f"★{self.stars}")

        if self.citations is not None:
            signals.append(f"{self.citations} citations")

        if self.engagement is not None:
            signals.append(f"{self.engagement} engagements")

        if self.published_date:
            days_old = (datetime.now() - self.published_date).days
            if days_old == 0:
                signals.append("today")
            elif days_old == 1:
                signals.append("yesterday")
            elif days_old < 7:
                signals.append(f"{days_old}d ago")
            elif days_old < 30:
                signals.append(f"{days_old // 7}w ago")
            else:
                signals.append(self.published_date.strftime("%b %d"))

        self.signal_string = ", ".join(signals) if signals else ""

    def _compute_time_decay(self):
        """Apply time-decay scoring based on source type."""
        if not self.published_date:
            self.time_decay_score = self.base_relevance * 0.7  # Penalty for unknown date
            return

        days_old = max(0, (datetime.now() - self.published_date).days)

        # Different decay rates by category (from my architecture)
        decay_rates = {
            SourceCategory.RESEARCH: 0.03,    # 23-day half-life
            SourceCategory.LAB: 0.05,         # 14-day half-life
            SourceCategory.INDUSTRY: 0.15,    # 4.6-day half-life
            SourceCategory.GITHUB: 0.02,      # 35-day half-life
            SourceCategory.BENCHMARK: 0.04,   # 17-day half-life
            SourceCategory.SOCIAL: 0.35,      # 2-day half-life
            SourceCategory.INTERNAL: 0.01,    # 69-day half-life
        }

        lambda_decay = decay_rates.get(self.category, 0.1)
        recency_multiplier = math.exp(-lambda_decay * days_old)

        self.time_decay_score = self.base_relevance * recency_multiplier

    def _compute_final_score(self):
        """Apply tier weighting to get final score."""
        tier_weights = {
            SourceTier.TIER_1: 1.0,
            SourceTier.TIER_2: 0.85,
            SourceTier.TIER_3: 0.7,
        }

        weight = tier_weights.get(self.tier, 0.5)
        self.final_score = self.time_decay_score * weight

    def to_citation(self) -> str:
        """Format as ResearchGravity-style citation."""
        if self.signal_string:
            return f"[{self.title}]({self.url}) — {self.signal_string}"
        return f"[{self.title}]({self.url})"


@dataclass
class SearchContext:
    """Aggregated search results ready for agent injection."""
    query: str
    results: list[SearchResult] = field(default_factory=list)

    # Organized by tier
    tier1_results: list[SearchResult] = field(default_factory=list)
    tier2_results: list[SearchResult] = field(default_factory=list)
    tier3_results: list[SearchResult] = field(default_factory=list)

    # For grounded generation
    citation_context: dict[str, str] = field(default_factory=dict)  # source_id -> content

    search_time_ms: int = 0

    def get_top_results(self, limit: int = 10) -> list[SearchResult]:
        """Get top results by final score."""
        return sorted(self.results, key=lambda r: r.final_score, reverse=True)[:limit]

    def build_citation_context(self, limit: int = 15):
        """Build context for grounded generation."""
        top_results = self.get_top_results(limit)

        for i, result in enumerate(top_results):
            source_id = f"[{i+1}]"
            self.citation_context[source_id] = {
                'url': result.url,
                'title': result.title,
                'content': result.content[:2000],  # Truncate for context window
                'signal': result.signal_string,
                'tier': result.tier.name,
            }

    def get_grounding_prompt(self) -> str:
        """Generate grounding context for agents."""
        if not self.citation_context:
            self.build_citation_context()

        lines = [
            "## Retrieved Sources (cite these ONLY)",
            "",
            "You MUST cite sources using [N] format. Only cite from this list:",
            ""
        ]

        for source_id, data in self.citation_context.items():
            lines.append(f"**{source_id}** {data['title']}")
            lines.append(f"   URL: {data['url']}")
            lines.append(f"   Signal: {data['signal']}")
            lines.append(f"   Content: {data['content'][:500]}...")
            lines.append("")

        return "\n".join(lines)


class TieredSearchLayer:
    """
    Implements ResearchGravity's tiered search methodology.

    Tier 1 (Primary - check first):
    - arXiv (cs.AI, cs.LG, cs.SE)
    - Lab blogs (OpenAI, Anthropic, Google AI)
    - Industry news (TechCrunch, Verge)

    Tier 2 (Amplifiers):
    - GitHub trending/search
    - Benchmarks (METR, ARC Prize)
    - Social signals (Twitter/X)

    Tier 3 (Context):
    - Internal learnings (Qdrant)
    - Newsletters (if indexed)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._load_api_keys()

    def _load_api_keys(self):
        """Load API keys from config."""
        import json
        from pathlib import Path

        config_path = Path.home() / ".agent-core" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
                self.cohere_key = cfg.get('cohere', {}).get('api_key')
                self.github_token = cfg.get('github', {}).get('token')
        else:
            self.cohere_key = None
            self.github_token = None

    async def search(self, query: str, max_results_per_tier: int = 10) -> SearchContext:
        """
        Execute tiered search in parallel.

        All tiers search simultaneously for speed.
        Results are then scored and ranked.
        """
        start_time = datetime.now()

        context = SearchContext(query=query)

        # Execute all tier searches in parallel
        tier1_task = self._search_tier1(query, max_results_per_tier)
        tier2_task = self._search_tier2(query, max_results_per_tier)
        tier3_task = self._search_tier3(query, max_results_per_tier)

        results = await asyncio.gather(
            tier1_task,
            tier2_task,
            tier3_task,
            return_exceptions=True
        )

        # Collect results
        if not isinstance(results[0], Exception):
            context.tier1_results = results[0]
            context.results.extend(results[0])

        if not isinstance(results[1], Exception):
            context.tier2_results = results[1]
            context.results.extend(results[1])

        if not isinstance(results[2], Exception):
            context.tier3_results = results[2]
            context.results.extend(results[2])

        # Build citation context
        context.build_citation_context()

        context.search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return context

    async def _search_tier1(self, query: str, limit: int) -> list[SearchResult]:
        """Search Tier 1: Primary sources."""
        results = []

        # Parallel search across Tier 1 sources
        tasks = [
            self._search_arxiv(query, limit // 2),
            self._search_web_tier1(query, limit // 2),
        ]

        tier1_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in tier1_results:
            if not isinstance(result, Exception):
                results.extend(result)

        return results

    async def _search_tier2(self, query: str, limit: int) -> list[SearchResult]:
        """Search Tier 2: Amplifiers."""
        results = []

        tasks = [
            self._search_github(query, limit // 2),
            self._search_web_tier2(query, limit // 2),
        ]

        tier2_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in tier2_results:
            if not isinstance(result, Exception):
                results.extend(result)

        return results

    async def _search_tier3(self, query: str, limit: int) -> list[SearchResult]:
        """Search Tier 3: Internal context."""
        return await self._search_internal(query, limit)

    async def _search_arxiv(self, query: str, limit: int) -> list[SearchResult]:
        """
        Search arXiv for relevant papers using keyword extraction and category filtering.

        Improvements over naive search:
        1. Extract meaningful keywords from natural language query
        2. Filter by relevant arXiv categories (cs.AI, cs.MA, cs.LG, cs.CL)
        3. Sort by relevance first, then apply time-decay scoring
        """
        if not HAS_ARXIV:
            return []

        results = []

        try:
            # Extract keywords from query
            arxiv_query = self._build_arxiv_query(query)

            client = arxiv.Client()
            search = arxiv.Search(
                query=arxiv_query,
                max_results=limit * 2,  # Get more, then filter
                sort_by=arxiv.SortCriterion.Relevance,  # Sort by relevance, not date
                sort_order=arxiv.SortOrder.Descending
            )

            for paper in client.results(search):
                # Extract arXiv ID
                arxiv_id = paper.entry_id.split('/abs/')[-1]
                if 'v' in arxiv_id:
                    arxiv_id = arxiv_id.split('v')[0]

                # Check if paper is from relevant categories
                categories = [cat for cat in paper.categories] if paper.categories else []
                relevance_boost = self._compute_category_relevance(categories, query)

                result = SearchResult(
                    url=paper.entry_id,
                    title=paper.title,
                    content=f"arXiv:{arxiv_id} - {paper.summary}",  # Include arXiv ID in content for citation
                    tier=SourceTier.TIER_1,
                    category=SourceCategory.RESEARCH,
                    source_name="arXiv",
                    published_date=paper.published.replace(tzinfo=None) if paper.published else None,
                    base_relevance=0.85 + relevance_boost,  # Boost relevant categories
                )
                results.append(result)

                if len(results) >= limit:
                    break

        except Exception as e:
            print(f"arXiv search error: {e}")

        return results

    def _build_arxiv_query(self, query: str) -> str:
        """
        Build optimized arXiv query from natural language.

        Extracts keywords and adds category filters for AI/ML papers.
        """
        # Stopwords to remove
        stopwords = {
            'what', 'are', 'the', 'best', 'practices', 'for', 'in', 'how',
            'do', 'does', 'can', 'should', 'would', 'could', 'a', 'an',
            'to', 'of', 'and', 'or', 'is', 'it', 'this', 'that', 'with',
            'from', 'by', 'on', 'at', 'as', 'be', 'was', 'were', 'been',
            'have', 'has', 'had', 'having', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between',
            '2024', '2025', '2026', 'current', 'latest', 'recent', 'new'
        }

        # Domain-specific keyword mappings for better arXiv search
        keyword_expansions = {
            'multi-agent': ['multi-agent', 'multiagent', 'MAS', 'multi agent'],
            'orchestration': ['orchestration', 'coordination', 'collaboration'],
            'llm': ['LLM', 'large language model', 'GPT', 'transformer'],
            'agent': ['agent', 'autonomous', 'agentic'],
            'rag': ['RAG', 'retrieval augmented', 'retrieval-augmented'],
            'consensus': ['consensus', 'agreement', 'voting', 'debate'],
            'reasoning': ['reasoning', 'chain-of-thought', 'CoT'],
        }

        # Extract words from query
        words = re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)?\b', query.lower())

        # Filter out stopwords
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Expand domain-specific keywords
        expanded = []
        for kw in keywords:
            if kw in keyword_expansions:
                expanded.extend(keyword_expansions[kw])
            else:
                expanded.append(kw)

        # Build arXiv query with category filter
        # Focus on AI/ML categories: cs.AI, cs.MA, cs.LG, cs.CL
        categories = "(cat:cs.AI OR cat:cs.MA OR cat:cs.LG OR cat:cs.CL)"

        if expanded:
            # Use OR for keywords to get broader results
            keyword_query = " OR ".join(f'"{kw}"' if ' ' in kw or '-' in kw else kw for kw in expanded[:6])
            return f"({keyword_query}) AND {categories}"
        else:
            # Fallback to original query with category filter
            return f"{query} AND {categories}"

    def _compute_category_relevance(self, categories: list[str], query: str) -> float:
        """
        Compute relevance boost based on arXiv categories.

        Higher boost for categories more relevant to the query.
        """
        query_lower = query.lower()

        # Category relevance for common query topics
        category_boosts = {
            'cs.AI': 0.10,   # Artificial Intelligence
            'cs.MA': 0.15,   # Multi-agent systems (highest for agent queries)
            'cs.LG': 0.08,   # Machine Learning
            'cs.CL': 0.08,   # Computation and Language (NLP/LLM)
            'cs.SE': 0.05,   # Software Engineering
            'cs.DC': 0.05,   # Distributed Computing
            'cs.NE': 0.05,   # Neural/Evolutionary Computing
        }

        # Extra boost for multi-agent queries
        if 'multi-agent' in query_lower or 'orchestration' in query_lower or 'agent' in query_lower:
            category_boosts['cs.MA'] = 0.20
            category_boosts['cs.AI'] = 0.15

        boost = 0.0
        for cat in categories:
            if cat in category_boosts:
                boost = max(boost, category_boosts[cat])

        return boost

    async def _search_github(self, query: str, limit: int) -> list[SearchResult]:
        """Search GitHub for relevant repositories."""
        if not HAS_AIOHTTP:
            return []

        results = []

        try:
            # Apply ResearchGravity's viral filter
            search_query = f"{query} stars:>100"

            headers = {'Accept': 'application/vnd.github.v3+json'}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'

            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/search/repositories?q={search_query}&sort=stars&per_page={limit}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        for repo in data.get('items', [])[:limit]:
                            # Parse dates
                            pushed_at = None
                            if repo.get('pushed_at'):
                                try:
                                    pushed_at = datetime.fromisoformat(repo['pushed_at'].replace('Z', ''))
                                except:
                                    pass

                            result = SearchResult(
                                url=repo['html_url'],
                                title=repo['full_name'],
                                content=repo.get('description', '') or '',
                                tier=SourceTier.TIER_2,
                                category=SourceCategory.GITHUB,
                                source_name="GitHub",
                                published_date=pushed_at,
                                stars=repo.get('stargazers_count', 0),
                                base_relevance=0.85,
                            )
                            results.append(result)

        except Exception as e:
            print(f"GitHub search error: {e}")

        return results

    async def _search_web_tier1(self, query: str, limit: int) -> list[SearchResult]:
        """Search web for Tier 1 sources (labs, industry news)."""
        # This would use WebSearch tool in production
        # For now, return empty - will be filled by orchestrator using Claude's WebSearch
        return []

    async def _search_web_tier2(self, query: str, limit: int) -> list[SearchResult]:
        """Search web for Tier 2 sources (benchmarks, social)."""
        return []

    async def _search_internal(self, query: str, limit: int) -> list[SearchResult]:
        """Search internal Qdrant for learnings."""
        results = []

        try:
            # Import Qdrant client
            import sys
            sys.path.insert(0, '/Users/dicoangelo/researchgravity')
            from storage.qdrant_db import get_qdrant

            qdrant = await get_qdrant()
            findings = await qdrant.search_findings(query, limit=limit)
            await qdrant.close()

            for finding in findings:
                result = SearchResult(
                    url=finding.get('url', f"internal://{finding.get('id', 'unknown')}"),
                    title=finding.get('title', 'Internal Learning'),
                    content=finding.get('content', ''),
                    tier=SourceTier.TIER_3,
                    category=SourceCategory.INTERNAL,
                    source_name="ResearchGravity",
                    base_relevance=finding.get('relevance_score', finding.get('score', 0.7)),
                )
                results.append(result)

        except Exception as e:
            print(f"Internal search error: {e}")

        return results


# Singleton instance
_search_layer: Optional[TieredSearchLayer] = None

def get_search_layer() -> TieredSearchLayer:
    """Get or create the search layer singleton."""
    global _search_layer
    if _search_layer is None:
        _search_layer = TieredSearchLayer()
    return _search_layer


async def search_tiered(query: str, max_results: int = 30) -> SearchContext:
    """Convenience function for tiered search."""
    layer = get_search_layer()
    return await layer.search(query, max_results_per_tier=max_results // 3)
