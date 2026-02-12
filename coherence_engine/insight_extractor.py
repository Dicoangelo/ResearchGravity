"""
Coherence Engine — Insight Extractor

Given a coherence moment (two events that aligned across platforms),
fetches the full conversation context around each event and uses an LLM
to extract the ACTUAL insight — what was discovered, not just that
similarity was detected.

Designed for both real-time (post-moment-detection) and backfill
(enriching existing moments).
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger("coherence.insight_extractor")

# ── Configuration ────────────────────────────────────────────────────────────

# Context window: how many events before/after the target to include
CONTEXT_WINDOW = 5

# LLM provider: "anthropic" (Claude API) or "local" (ollama)
LLM_PROVIDER = os.environ.get("UCW_LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.environ.get("UCW_INSIGHT_MODEL", "claude-sonnet-4-5-20250929")
OLLAMA_MODEL = os.environ.get("UCW_OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


@dataclass
class InsightResult:
    """Extracted insight from a coherence moment."""
    moment_id: str
    summary: str          # What was actually discovered
    category: str         # crystallization, synthesis, convergence, etc.
    novelty: float        # 0-1: how novel is this insight
    raw_response: str     # Full LLM response for debugging


# ── Conversation Reconstruction ──────────────────────────────────────────────

async def get_conversation_context(
    pool: asyncpg.Pool,
    event_id: str,
    window: int = CONTEXT_WINDOW,
) -> List[Dict[str, Any]]:
    """
    Fetch the conversation context around an event.

    Returns up to `window` events before and after the target event
    from the same session, ordered chronologically.
    """
    async with pool.acquire() as conn:
        # First, get the target event and its session
        target = await conn.fetchrow(
            """SELECT event_id, session_id, timestamp_ns, platform,
                      direction, method,
                      data_layer->>'content' AS data_content,
                      light_layer->>'topic' AS light_topic,
                      light_layer->>'intent' AS light_intent,
                      light_layer->>'summary' AS light_summary,
                      light_layer->'concepts' AS light_concepts,
                      instinct_layer->>'gut_signal' AS instinct_gut_signal,
                      (instinct_layer->>'coherence_potential')::REAL AS instinct_coherence
               FROM cognitive_events
               WHERE event_id = $1""",
            event_id,
        )
        if not target:
            return []

        session_id = target["session_id"]
        ts = target["timestamp_ns"]

        if not session_id:
            # No session — return just the target event
            return [dict(target)]

        # Fetch N events before and after from the same session
        _EVENT_COLS = """event_id, session_id, timestamp_ns, platform,
                         direction, method,
                         data_layer->>'content' AS data_content,
                         light_layer->>'topic' AS light_topic,
                         light_layer->>'intent' AS light_intent,
                         light_layer->>'summary' AS light_summary,
                         light_layer->'concepts' AS light_concepts,
                         instinct_layer->>'gut_signal' AS instinct_gut_signal,
                         (instinct_layer->>'coherence_potential')::REAL AS instinct_coherence"""

        context = await conn.fetch(
            f"""(SELECT {_EVENT_COLS}
                 FROM cognitive_events
                 WHERE session_id = $1 AND timestamp_ns <= $2
                 ORDER BY timestamp_ns DESC LIMIT $3)
                UNION ALL
                (SELECT {_EVENT_COLS}
                 FROM cognitive_events
                 WHERE session_id = $1 AND timestamp_ns > $2
                 ORDER BY timestamp_ns ASC LIMIT $3)
                ORDER BY timestamp_ns ASC""",
            session_id, ts, window,
        )

        return [dict(r) for r in context]


def format_conversation_thread(events: List[Dict], highlight_event_id: str) -> str:
    """Format a list of events into a readable conversation thread."""
    lines = []
    for ev in events:
        marker = " <<<TARGET>>>" if ev["event_id"] == highlight_event_id else ""
        direction = ev.get("direction", "")
        arrow = "USER:" if direction == "out" else "ASSISTANT:"
        content = (ev.get("data_content") or ev.get("light_summary") or "")[:500]
        topic = ev.get("light_topic", "")
        lines.append(f"[{arrow} topic={topic}]{marker}\n{content}\n")
    return "\n".join(lines)


# ── LLM Synthesis ────────────────────────────────────────────────────────────

INSIGHT_PROMPT = """You are analyzing a coherence moment — two conversations on different AI platforms that aligned around the same intellectual thread.

## Event A ({platform_a})
### Conversation Context:
{context_a}

## Event B ({platform_b})
### Conversation Context:
{context_b}

## Coherence Detection
- Type: {coherence_type}
- Confidence: {confidence:.2f}
- Description: {description}

## Your Task

Analyze these two conversation threads and extract the REAL insight. Don't just say "these are similar" — explain:

1. **What question or problem was being explored?** (the underlying intellectual thread)
2. **What insight emerged?** (the actual discovery, realization, or synthesis)
3. **Why did it appear on both platforms?** (what drove the convergence)
4. **How novel is this?** (0.0 = trivial/obvious, 1.0 = genuine breakthrough)

## Output Format (JSON)

```json
{{
  "summary": "One paragraph describing the actual insight that emerged",
  "category": "one of: crystallization | synthesis | convergence | refinement | validation | dead_end",
  "novelty": 0.0-1.0,
  "question": "The underlying question being explored",
  "significance": "Why this matters in the broader context of the user's work"
}}
```

Categories:
- **crystallization**: A vague idea became concrete
- **synthesis**: Ideas from different domains merged into something new
- **convergence**: Multiple platforms arrived at the same conclusion independently
- **refinement**: An existing idea was significantly improved
- **validation**: A hypothesis was confirmed across platforms
- **dead_end**: The convergence revealed a dead end or contradiction
"""


async def synthesize_insight(
    moment_id: str,
    platform_a: str,
    platform_b: str,
    context_a: str,
    context_b: str,
    coherence_type: str,
    confidence: float,
    description: str,
) -> InsightResult:
    """Use an LLM to synthesize the actual insight from a coherence moment."""

    prompt = INSIGHT_PROMPT.format(
        platform_a=platform_a,
        platform_b=platform_b,
        context_a=context_a,
        context_b=context_b,
        coherence_type=coherence_type,
        confidence=confidence,
        description=description,
    )

    raw_response = ""

    if LLM_PROVIDER == "anthropic":
        raw_response = await _call_anthropic(prompt)
    else:
        raw_response = await _call_ollama(prompt)

    # Parse the JSON from the response
    try:
        # Extract JSON from markdown code block if present
        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        parsed = json.loads(json_str.strip())
        return InsightResult(
            moment_id=moment_id,
            summary=parsed.get("summary", "Failed to extract summary"),
            category=parsed.get("category", "unknown"),
            novelty=float(parsed.get("novelty", 0.3)),
            raw_response=raw_response,
        )
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        log.warning(f"Failed to parse insight JSON for {moment_id}: {e}")
        # Fallback: use the raw response as summary
        return InsightResult(
            moment_id=moment_id,
            summary=raw_response[:500] if raw_response else "Extraction failed",
            category="unknown",
            novelty=0.3,
            raw_response=raw_response,
        )


async def _call_anthropic(prompt: str) -> str:
    """Call Anthropic API for insight synthesis."""
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed — pip install anthropic")
        return ""

    client = anthropic.AsyncAnthropic()  # uses ANTHROPIC_API_KEY env var
    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        log.error(f"Anthropic API call failed: {e}")
        return ""


async def _call_ollama(prompt: str) -> str:
    """Call local Ollama for insight synthesis."""
    import asyncio
    try:
        import httpx
    except ImportError:
        log.error("httpx not installed — pip install httpx")
        return ""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
    except Exception as e:
        log.error(f"Ollama call failed: {e}")
        return ""


# ── Main Entry Points ────────────────────────────────────────────────────────

async def extract_insight_for_moment(
    pool: asyncpg.Pool,
    moment_id: str,
) -> Optional[InsightResult]:
    """
    Extract the actual insight from a coherence moment.

    1. Load the moment from DB
    2. Get conversation context for both events
    3. Synthesize via LLM
    4. Store the insight back on the moment
    """
    async with pool.acquire() as conn:
        moment = await conn.fetchrow(
            """SELECT moment_id, event_ids, platforms, coherence_type,
                      confidence, description
               FROM coherence_moments
               WHERE moment_id = $1""",
            moment_id,
        )

    if not moment:
        log.warning(f"Moment {moment_id} not found")
        return None

    event_ids = moment["event_ids"]
    platforms = moment["platforms"]

    if len(event_ids) < 2:
        log.warning(f"Moment {moment_id} has fewer than 2 events")
        return None

    # Get conversation context for both events
    context_a = await get_conversation_context(pool, event_ids[0])
    context_b = await get_conversation_context(pool, event_ids[1])

    thread_a = format_conversation_thread(context_a, event_ids[0])
    thread_b = format_conversation_thread(context_b, event_ids[1])

    # Synthesize
    result = await synthesize_insight(
        moment_id=moment_id,
        platform_a=platforms[0] if platforms else "unknown",
        platform_b=platforms[1] if len(platforms) > 1 else "unknown",
        context_a=thread_a,
        context_b=thread_b,
        coherence_type=moment["coherence_type"],
        confidence=moment["confidence"],
        description=moment["description"] or "",
    )

    # Store back on the moment
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE coherence_moments
               SET insight_summary = $2,
                   insight_category = $3,
                   insight_novelty = $4
               WHERE moment_id = $1""",
            moment_id,
            result.summary,
            result.category,
            result.novelty,
        )

    log.info(
        f"Insight extracted for {moment_id}: "
        f"category={result.category} novelty={result.novelty:.2f}"
    )
    return result


async def backfill_insights(
    pool: asyncpg.Pool,
    limit: int = 72,
    skip_existing: bool = True,
) -> List[InsightResult]:
    """
    Backfill insights on existing coherence moments.

    Args:
        pool: asyncpg connection pool
        limit: max moments to process
        skip_existing: skip moments that already have insight_summary
    """
    async with pool.acquire() as conn:
        where = "WHERE insight_summary IS NULL" if skip_existing else ""
        moments = await conn.fetch(
            f"""SELECT moment_id FROM coherence_moments
                {where}
                ORDER BY confidence DESC
                LIMIT $1""",
            limit,
        )

    results = []
    for i, row in enumerate(moments):
        log.info(f"Backfilling insight {i+1}/{len(moments)}: {row['moment_id']}")
        result = await extract_insight_for_moment(pool, row["moment_id"])
        if result:
            results.append(result)

    log.info(f"Backfill complete: {len(results)}/{len(moments)} insights extracted")
    return results
