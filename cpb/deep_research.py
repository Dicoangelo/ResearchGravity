#!/usr/bin/env python3
"""
CPB Deep Research Layer - External Research API Integration (v2.5 Hardened)

Integrates external deep research APIs (Perplexity, Gemini) to augment
CPB's internal search with real-time web research capabilities.

v2.5 Enhancements:
- Caching layer with 15-min TTL to avoid redundant API calls
- Retry logic with exponential backoff (1s, 2s, 4s) for transient failures
- Provider fallback chain (Gemini → Perplexity) on failure
- Hardened Gemini grounding extraction with 3 fallback methods

Supported Providers:
- Gemini (default): gemini-2.0-flash with Google Search grounding
- Perplexity: sonar, sonar-pro models

Usage:
    results = await deep_research(query, provider="gemini")
    # Returns list of SearchResult objects ready for injection into pipeline

    # With fallback on failure:
    result, search_results, provider_used = await deep_research_with_fallback(query)
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GEMINI = True
except ImportError:
    genai = None
    genai_types = None
    HAS_GEMINI = False

from .search_layer import SearchResult, SourceTier, SourceCategory


# =============================================================================
# CACHING LAYER (v2.5)
# =============================================================================

# Cache with 15-minute TTL
_deep_research_cache: dict = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


def _cache_key(query: str, provider: str, model: str) -> str:
    """Generate cache key for deep research result."""
    # Normalize query for caching
    normalized = query.strip().lower()
    return f"{normalized}:{provider}:{model}"


def _get_cached(key: str) -> Optional['DeepResearchResult']:
    """Get cached result if still valid."""
    if key in _deep_research_cache:
        cached_time, result = _deep_research_cache[key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return result
        else:
            # Expired, remove from cache
            del _deep_research_cache[key]
    return None


def _set_cached(key: str, result: 'DeepResearchResult'):
    """Store result in cache with current timestamp."""
    _deep_research_cache[key] = (time.time(), result)


def clear_deep_research_cache():
    """Clear all cached deep research results."""
    global _deep_research_cache
    _deep_research_cache = {}


def get_cache_stats() -> dict:
    """Get cache statistics."""
    now = time.time()
    valid_count = sum(
        1 for cached_time, _ in _deep_research_cache.values()
        if now - cached_time < CACHE_TTL_SECONDS
    )
    return {
        'total_entries': len(_deep_research_cache),
        'valid_entries': valid_count,
        'ttl_seconds': CACHE_TTL_SECONDS,
    }


# =============================================================================
# RETRY LOGIC (v2.5)
# =============================================================================

async def _retry_with_backoff(
    async_fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,)
) -> any:
    """
    Retry an async function with exponential backoff.

    Args:
        async_fn: Async function to call (no arguments, use lambda)
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 10.0)
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result from async_fn

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await async_fn()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)

    # All retries exhausted
    raise last_exception


class DeepResearchProvider(Enum):
    """Supported deep research providers."""
    PERPLEXITY = "perplexity"
    GEMINI = "gemini"  # Future


@dataclass
class DeepResearchResult:
    """Result from deep research API."""
    content: str  # The synthesized research answer
    citations: list[dict]  # List of {url, title, snippet}
    provider: str
    model: str
    query: str
    search_time_ms: int = 0
    token_count: int = 0
    cost_usd: float = 0.0


class PerplexityClient:
    """
    Perplexity API client for deep research.

    Models:
    - sonar: Fast, cheap (~$0.005/query) - good for augmentation
    - sonar-pro: Deeper research (~$0.05/query) - better for complex queries

    API docs: https://docs.perplexity.ai/
    """

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_api_key()

    def _load_api_key(self) -> Optional[str]:
        """Load API key from config."""
        config_path = Path.home() / ".agent-core" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
                return cfg.get('perplexity', {}).get('api_key')
        return None

    async def research(
        self,
        query: str,
        model: str = "sonar",  # or "sonar-pro" for deeper research
        system_prompt: Optional[str] = None
    ) -> DeepResearchResult:
        """
        Execute deep research query via Perplexity API.

        Args:
            query: Research query
            model: Model to use (sonar or sonar-pro)
            system_prompt: Optional system prompt for focus

        Returns:
            DeepResearchResult with content and citations
        """
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp required for deep research. Install with: pip install aiohttp")

        if not self.api_key:
            raise ValueError("Perplexity API key not found. Add to ~/.agent-core/config.json under 'perplexity.api_key'")

        start_time = datetime.now()

        # Build request
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "return_citations": True,
            "return_related_questions": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.BASE_URL, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Perplexity API error {response.status}: {error_text}")

                data = await response.json()

        # Parse response
        choice = data.get('choices', [{}])[0]
        message = choice.get('message', {})
        content = message.get('content', '')

        # Extract citations
        citations = []
        raw_citations = data.get('citations', [])
        for i, url in enumerate(raw_citations):
            citations.append({
                'url': url,
                'title': f"Source {i+1}",  # Perplexity doesn't always return titles
                'snippet': '',
            })

        # Calculate timing and cost
        search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        usage = data.get('usage', {})
        token_count = usage.get('total_tokens', 0)

        # Approximate cost (Perplexity pricing as of 2025)
        cost_per_1k = 0.001 if model == "sonar" else 0.005  # sonar-pro is ~5x
        cost_usd = (token_count / 1000) * cost_per_1k

        return DeepResearchResult(
            content=content,
            citations=citations,
            provider="perplexity",
            model=model,
            query=query,
            search_time_ms=search_time_ms,
            token_count=token_count,
            cost_usd=cost_usd,
        )


class GeminiClient:
    """
    Gemini API client for deep research with Google Search grounding (v2.5 - new SDK).

    Models:
    - gemini-2.0-flash: Fast, grounding supported
    - gemini-1.5-pro: More capable, grounding supported

    Requires: pip install google-genai
    API docs: https://ai.google.dev/
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_api_key()
        self._client = None

    def _load_api_key(self) -> Optional[str]:
        """Load API key from config or environment."""
        import os

        # Check environment first (new SDK uses GEMINI_API_KEY or GOOGLE_API_KEY)
        if os.environ.get('GEMINI_API_KEY'):
            return os.environ['GEMINI_API_KEY']
        if os.environ.get('GOOGLE_API_KEY'):
            return os.environ['GOOGLE_API_KEY']

        # Check config file
        config_path = Path.home() / ".agent-core" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
                # Try multiple possible keys
                return (
                    cfg.get('gemini', {}).get('api_key') or
                    cfg.get('google', {}).get('api_key') or
                    cfg.get('google_ai', {}).get('api_key')
                )
        return None

    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None and HAS_GEMINI and self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def research(
        self,
        query: str,
        model: str = "gemini-2.0-flash",
        system_prompt: Optional[str] = None
    ) -> DeepResearchResult:
        """
        Execute deep research query via Gemini API with Google Search grounding.

        Args:
            query: Research query
            model: Model to use
            system_prompt: Optional system prompt

        Returns:
            DeepResearchResult with content and citations
        """
        if not HAS_GEMINI:
            raise RuntimeError("google-genai required. Install with: pip install google-genai")

        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY env var or add to ~/.agent-core/config.json")

        client = self._get_client()
        if not client:
            raise RuntimeError("Failed to create Gemini client")

        start_time = datetime.now()

        # Build prompt for research
        research_prompt = f"""Research the following query thoroughly using web search.
Provide a comprehensive answer with specific facts, data, and citations.

Query: {query}

Requirements:
1. Search for recent, authoritative sources
2. Include specific data points, statistics, and examples
3. Cite your sources clearly
4. Focus on accuracy over speculation"""

        # Configure with Google Search grounding (new SDK)
        config = genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096,
            system_instruction=system_prompt or "You are a research assistant. Provide comprehensive, well-cited answers using search results.",
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
        )

        # Execute (run in thread pool since Gemini SDK is sync)
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=research_prompt,
                config=config
            )
        )

        # Extract content and grounding metadata
        content = response.text if response.text else ""

        # Extract citations using hardened extraction (v2.5)
        citations = self._extract_gemini_citations(response, content)

        # Calculate timing and approximate cost
        search_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Approximate token count from response
        token_count = 0
        if hasattr(response, 'usage_metadata'):
            um = response.usage_metadata
            token_count = getattr(um, 'total_token_count', 0)

        # Gemini pricing (approximate as of 2025)
        # Flash: ~$0.075/1M input, ~$0.30/1M output
        # Pro: ~$1.25/1M input, ~$5/1M output
        if 'flash' in model.lower():
            cost_per_1k = 0.0002
        else:
            cost_per_1k = 0.003
        cost_usd = (token_count / 1000) * cost_per_1k

        return DeepResearchResult(
            content=content,
            citations=citations,
            provider="gemini",
            model=model,
            query=query,
            search_time_ms=search_time_ms,
            token_count=token_count,
            cost_usd=cost_usd,
        )

    def _extract_gemini_citations(self, response, content: str) -> list[dict]:
        """
        Extract citations from Gemini response with 3 fallback methods (v2.5).

        Fallback chain:
        1. grounding_metadata.grounding_chunks (primary)
        2. grounding_metadata.grounding_supports (inline citations)
        3. Parse inline URLs from response text (last resort)

        Args:
            response: Gemini API response object
            content: Response text content

        Returns:
            List of citation dicts with url, title, snippet
        """
        citations = []
        seen_urls = set()

        # Method 1: grounding_chunks (primary)
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    gm = candidate.grounding_metadata

                    # Try grounding_chunks first
                    if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                url = getattr(chunk.web, 'uri', '') or ''
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    citations.append({
                                        'url': url,
                                        'title': getattr(chunk.web, 'title', 'Web Source') or 'Web Source',
                                        'snippet': '',
                                    })
        except Exception:
            pass

        # Method 2: grounding_supports (inline citations)
        if not citations:
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata'):
                        gm = candidate.grounding_metadata

                        if hasattr(gm, 'grounding_supports') and gm.grounding_supports:
                            for support in gm.grounding_supports:
                                # grounding_supports links segments to sources
                                if hasattr(support, 'grounding_chunk_indices'):
                                    # References indices in grounding_chunks
                                    pass  # Already handled above
                                if hasattr(support, 'web') and support.web:
                                    url = getattr(support.web, 'uri', '') or ''
                                    if url and url not in seen_urls:
                                        seen_urls.add(url)
                                        citations.append({
                                            'url': url,
                                            'title': getattr(support.web, 'title', 'Web Source') or 'Web Source',
                                            'snippet': getattr(support, 'segment', '') or '',
                                        })
            except Exception:
                pass

        # Method 3: Parse inline URLs from text (last resort)
        if not citations and content:
            try:
                # Find URLs in the response text
                url_pattern = r'https?://[^\s\)\]\>\"\']+[^\s\.\,\!\?\)\]\>\"\']'
                urls = re.findall(url_pattern, content)

                for url in urls[:10]:  # Limit to 10
                    # Clean up common URL artifacts
                    url = url.rstrip('.,;:')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        # Try to extract title from URL
                        title = 'Web Source'
                        if 'arxiv.org' in url:
                            title = 'arXiv Paper'
                        elif 'github.com' in url:
                            title = 'GitHub Repository'
                        elif 'wikipedia.org' in url:
                            title = 'Wikipedia Article'

                        citations.append({
                            'url': url,
                            'title': title,
                            'snippet': '',
                        })
            except Exception:
                pass

        return citations


def deep_result_to_search_results(result: DeepResearchResult) -> list[SearchResult]:
    """
    Convert DeepResearchResult to list of SearchResult objects.

    The synthesized content becomes one Tier 1 source, and each citation
    becomes an additional Tier 1 source.
    """
    search_results = []

    # Main synthesized result as Tier 1 source
    main_result = SearchResult(
        url=f"deep-research://{result.provider}/{result.model}",
        title=f"Deep Research: {result.query[:50]}...",
        content=result.content,
        tier=SourceTier.TIER_1,
        category=SourceCategory.RESEARCH,
        source_name=f"{result.provider.title()} {result.model}",
        published_date=datetime.now(),
        base_relevance=0.95,  # High relevance - it's a direct answer
    )
    search_results.append(main_result)

    # Each citation as additional Tier 1 source
    for i, citation in enumerate(result.citations[:10]):  # Limit to 10
        url = citation.get('url', '')
        if not url:
            continue

        # Determine category from URL
        category = SourceCategory.RESEARCH
        if 'arxiv.org' in url:
            category = SourceCategory.RESEARCH
        elif 'github.com' in url:
            category = SourceCategory.GITHUB
        elif any(site in url for site in ['techcrunch', 'theverge', 'wired']):
            category = SourceCategory.INDUSTRY

        citation_result = SearchResult(
            url=url,
            title=citation.get('title', f"Citation {i+1}"),
            content=citation.get('snippet', ''),
            tier=SourceTier.TIER_1,
            category=category,
            source_name=f"via {result.provider.title()}",
            published_date=datetime.now(),  # Unknown, use now
            base_relevance=0.85,
        )
        search_results.append(citation_result)

    return search_results


# =============================================================================
# PUBLIC API
# =============================================================================

_perplexity_client: Optional[PerplexityClient] = None
_gemini_client: Optional[GeminiClient] = None

def get_perplexity_client() -> PerplexityClient:
    """Get or create Perplexity client singleton."""
    global _perplexity_client
    if _perplexity_client is None:
        _perplexity_client = PerplexityClient()
    return _perplexity_client


def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


async def deep_research(
    query: str,
    provider: str = "gemini",  # Default to Gemini since user has account
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_cache: bool = True
) -> tuple[DeepResearchResult, list[SearchResult]]:
    """
    Execute deep research and return results ready for CPB pipeline (v2.5 with caching).

    Args:
        query: Research query
        provider: Provider to use (gemini, perplexity)
        model: Model override (default: gemini-2.0-flash for gemini, sonar for perplexity)
        system_prompt: Optional system prompt
        use_cache: Whether to use caching (default: True)

    Returns:
        Tuple of (DeepResearchResult, list of SearchResult for pipeline)
    """
    # Determine model
    if provider == "perplexity":
        model = model or "sonar"
    elif provider == "gemini":
        model = model or "gemini-2.0-flash"
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'perplexity'")

    # Check cache first (v2.5)
    if use_cache:
        cache_key = _cache_key(query, provider, model)
        cached_result = _get_cached(cache_key)
        if cached_result is not None:
            # Return cached result
            search_results = deep_result_to_search_results(cached_result)
            return cached_result, search_results

    # Execute research with retry logic (v2.5)
    async def _do_research():
        if provider == "perplexity":
            client = get_perplexity_client()
            return await client.research(query, model=model, system_prompt=system_prompt)
        else:  # gemini
            client = get_gemini_client()
            return await client.research(query, model=model, system_prompt=system_prompt)

    # Retry with exponential backoff
    result = await _retry_with_backoff(
        _do_research,
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0,
        exceptions=(RuntimeError, ValueError, Exception)
    )

    # Cache the result (v2.5)
    if use_cache:
        cache_key = _cache_key(query, provider, model)
        _set_cached(cache_key, result)

    search_results = deep_result_to_search_results(result)
    return result, search_results


async def deep_research_with_fallback(
    query: str,
    preferred_provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_cache: bool = True
) -> Tuple[DeepResearchResult, list[SearchResult], str]:
    """
    Execute deep research with automatic provider fallback (v2.5).

    Tries the preferred provider first, then falls back to the alternate
    if the primary fails.

    Fallback chain:
    - If preferred is Gemini: Gemini → Perplexity
    - If preferred is Perplexity: Perplexity → Gemini
    - If no preferred: Best available → Alternate

    Args:
        query: Research query
        preferred_provider: Preferred provider (gemini, perplexity, or None for auto)
        model: Model override
        system_prompt: Optional system prompt
        use_cache: Whether to use caching (default: True)

    Returns:
        Tuple of (DeepResearchResult, list of SearchResult, actual_provider_used)

    Raises:
        RuntimeError: If all providers fail
    """
    # Determine provider order
    if preferred_provider:
        providers = [preferred_provider]
        if preferred_provider == "gemini":
            providers.append("perplexity")
        else:
            providers.append("gemini")
    else:
        # Auto-detect best available
        best, _ = get_best_available_provider()
        if best == "gemini":
            providers = ["gemini", "perplexity"]
        elif best == "perplexity":
            providers = ["perplexity", "gemini"]
        else:
            raise RuntimeError("No deep research provider available")

    last_error = None

    for provider in providers:
        # Check if provider is available
        available, msg = check_deep_research_available(provider)
        if not available:
            continue

        try:
            result, search_results = await deep_research(
                query,
                provider=provider,
                model=model if provider == preferred_provider else None,  # Use default model for fallback
                system_prompt=system_prompt,
                use_cache=use_cache
            )
            return result, search_results, provider

        except Exception as e:
            last_error = e
            # Continue to fallback provider
            continue

    # All providers failed
    raise RuntimeError(f"All deep research providers failed. Last error: {last_error}")


def check_deep_research_available(provider: str = "gemini") -> tuple[bool, str]:
    """
    Check if deep research is available for the given provider.

    Returns:
        Tuple of (is_available, message)
    """
    if provider == "perplexity":
        if not HAS_AIOHTTP:
            return False, "aiohttp not installed. Run: pip install aiohttp"
        client = PerplexityClient()
        if not client.api_key:
            return False, "Perplexity API key not found. Add to ~/.agent-core/config.json under 'perplexity.api_key'"
        return True, "Perplexity ready"

    elif provider == "gemini":
        if not HAS_GEMINI:
            return False, "google-genai not installed. Run: pip install google-genai"
        client = GeminiClient()
        if not client.api_key:
            return False, "Gemini API key not found. Set GOOGLE_API_KEY env var or add to ~/.agent-core/config.json under 'gemini.api_key'"
        return True, "Gemini ready"

    return False, f"Unknown provider: {provider}. Use 'gemini' or 'perplexity'"


def get_best_available_provider() -> tuple[Optional[str], str]:
    """
    Find the best available deep research provider.

    Returns:
        Tuple of (provider_name or None, status message)
    """
    # Check Gemini first (user's preference)
    available, msg = check_deep_research_available("gemini")
    if available:
        return "gemini", msg

    # Fall back to Perplexity
    available, msg = check_deep_research_available("perplexity")
    if available:
        return "perplexity", msg

    return None, "No deep research provider available. Configure Gemini or Perplexity API key."


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    # Parse args: python -m cpb.deep_research "query" [provider]
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the latest multi-agent orchestration patterns in 2025?"
    provider_arg = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Query: {query}")
    print("-" * 50)

    # Find best provider or use specified
    if provider_arg:
        provider = provider_arg
        available, msg = check_deep_research_available(provider)
    else:
        provider, msg = get_best_available_provider()
        available = provider is not None

    if not available or not provider:
        print(f"Deep research not available: {msg}")
        sys.exit(1)

    print(f"Using provider: {provider} ({msg})")

    async def test():
        result, search_results = await deep_research(query, provider=provider)
        print(f"\nProvider: {result.provider} ({result.model})")
        print(f"Time: {result.search_time_ms}ms")
        print(f"Cost: ${result.cost_usd:.4f}")
        print(f"Citations: {len(result.citations)}")
        print(f"\n--- Content ---\n{result.content[:1500]}...")
        print(f"\n--- Citations ---")
        for c in result.citations[:5]:
            print(f"  - {c.get('title', 'Source')}: {c['url']}")
        print(f"\n--- Search Results for Pipeline: {len(search_results)} ---")

    asyncio.run(test())
