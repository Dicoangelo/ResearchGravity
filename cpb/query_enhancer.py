#!/usr/bin/env python3
"""
CPB Query Enhancer - Transforms casual queries into research-grade prompts.

Uses a fast, cheap model (Haiku) to:
1. Expand vague queries into specific research questions
2. Add temporal context and scope
3. Identify key dimensions to explore
4. Suggest follow-up queries for deeper research

This runs BEFORE the 7-agent precision cascade, ensuring the expensive
Opus calls get optimized input.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .llm_client import complete


@dataclass
class EnhancedQuery:
    """Result of query enhancement."""
    original: str
    enhanced: str
    reasoning: str
    follow_ups: list[str]
    dimensions: list[str]  # Key aspects the enhanced query covers
    was_enhanced: bool  # False if query was already high-quality


ENHANCER_PROMPT = """You are a Research Query Architect. Your job is to transform casual or vague questions into precise, research-grade queries optimized for multi-agent AI systems.

## Your Task

Take the user's query and enhance it for maximum research depth. The enhanced query will be processed by a 7-agent ensemble including researchers, analysts, skeptics, and synthesizers.

## Enhancement Principles

1. **Add Specificity**: Replace vague terms with concrete dimensions
   - "good" → "performance benchmarks, cost efficiency, scalability"
   - "best" → "trade-offs between [specific options]"

2. **Add Temporal Context**: Include relevant timeframes
   - "current" → "as of 2025"
   - "recent" → "developments in 2024-2025"

3. **Expand Scope Intelligently**: Cover related aspects experts would consider
   - Technical implementation details
   - Trade-offs and limitations
   - Comparison dimensions
   - Practical deployment considerations

4. **Frame as Research Question**: Structure for analytical depth
   - "What are X?" → "What are the key characteristics, trade-offs, and implementation patterns of X?"

5. **Keep Intent**: Don't change what the user is actually asking about

## Output Format (JSON)

{
  "enhanced_query": "The refined, research-grade query",
  "reasoning": "Brief explanation of what was enhanced and why (1-2 sentences)",
  "dimensions": ["dim1", "dim2", "dim3"],  // Key aspects the enhanced query covers
  "follow_ups": [
    "Suggested follow-up query 1",
    "Suggested follow-up query 2",
    "Suggested follow-up query 3"
  ],
  "was_enhanced": true  // false if original was already research-grade
}

## Examples

**Input**: "what's good for multi-agent?"
**Output**:
{
  "enhanced_query": "What are the architectural patterns, performance trade-offs, and implementation frameworks for multi-agent AI orchestration in 2025, comparing voting-based consensus, auction mechanisms, and hierarchical coordination approaches?",
  "reasoning": "Expanded 'good' into specific evaluation dimensions and added concrete comparison targets for the vague 'multi-agent' reference.",
  "dimensions": ["architecture", "performance", "frameworks", "coordination patterns"],
  "follow_ups": [
    "What are the cost implications of 3-agent vs 7-agent ensembles for production use?",
    "How do CrewAI, AutoGen, and LangGraph compare for enterprise multi-agent deployment?",
    "What failure modes and recovery patterns exist for multi-agent systems?"
  ],
  "was_enhanced": true
}

**Input**: "What are the trade-offs between transformer-based and RNN-based architectures for sequence modeling in terms of computational efficiency, parallelization, and long-range dependency handling?"
**Output**:
{
  "enhanced_query": "What are the trade-offs between transformer-based and RNN-based architectures for sequence modeling in terms of computational efficiency, parallelization, and long-range dependency handling?",
  "reasoning": "Query is already research-grade with specific dimensions and clear scope.",
  "dimensions": ["computational efficiency", "parallelization", "long-range dependencies"],
  "follow_ups": [
    "How do state-space models like Mamba compare to transformers for long sequences?",
    "What are the memory-compute trade-offs for different attention mechanisms?",
    "Which architecture patterns work best for real-time inference applications?"
  ],
  "was_enhanced": false
}

Now enhance this query:
"""


async def enhance_query(
    query: str,
    context: Optional[str] = None,
    model: str = "haiku"
) -> EnhancedQuery:
    """
    Enhance a query into a research-grade prompt.

    Args:
        query: The original user query
        context: Optional additional context
        model: Model to use (default: haiku for speed/cost)

    Returns:
        EnhancedQuery with original, enhanced, and follow-ups
    """
    # Build the prompt
    prompt = ENHANCER_PROMPT + f"\n{query}"

    if context:
        prompt += f"\n\nAdditional context provided:\n{context}"

    try:
        # Call LLM (Haiku for speed)
        llm_response = await complete(
            system_prompt="You are a Research Query Architect. Output valid JSON only.",
            user_prompt=prompt,
            model=model,
            temperature=0.3,
            max_tokens=1000
        )
        response = llm_response.content

        # Parse JSON response
        import json

        # Handle potential markdown code blocks
        text = response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)

        return EnhancedQuery(
            original=query,
            enhanced=result.get("enhanced_query", query),
            reasoning=result.get("reasoning", ""),
            follow_ups=result.get("follow_ups", []),
            dimensions=result.get("dimensions", []),
            was_enhanced=result.get("was_enhanced", True)
        )

    except Exception as e:
        # On any error, return original query unchanged
        return EnhancedQuery(
            original=query,
            enhanced=query,
            reasoning=f"Enhancement failed: {str(e)[:50]}",
            follow_ups=[],
            dimensions=[],
            was_enhanced=False
        )


def enhance_query_sync(
    query: str,
    context: Optional[str] = None,
    model: str = "haiku"
) -> EnhancedQuery:
    """Synchronous wrapper for enhance_query."""
    return asyncio.run(enhance_query(query, context, model))


# Quick test
if __name__ == "__main__":
    import sys

    test_query = sys.argv[1] if len(sys.argv) > 1 else "what's good for multi-agent?"

    print(f"Original: {test_query}")
    print("-" * 50)

    result = enhance_query_sync(test_query)

    print(f"Enhanced: {result.enhanced}")
    print(f"\nReasoning: {result.reasoning}")
    print(f"\nDimensions: {', '.join(result.dimensions)}")
    print(f"\nFollow-ups:")
    for i, f in enumerate(result.follow_ups, 1):
        print(f"  {i}. {f}")
    print(f"\nWas enhanced: {result.was_enhanced}")
