#!/usr/bin/env python3
"""
Test semantic search once vectors are loaded.
Run this after backfill completes to see Cohere embeddings + reranking in action.
"""

import asyncio
import sys
from storage.qdrant_db import get_qdrant


async def test_search(query: str = "multi-agent consensus"):
    print(f"\n━━━ Semantic Search: '{query}' ━━━━━━━━━━━━━━━━━━")

    q = await get_qdrant()

    # Check collection status directly
    try:
        info = q.client.get_collection("findings")
        findings_count = info.points_count
        print(f"✓ {findings_count:,} findings available")
    except Exception as e:
        print(f"⚠️  Error checking collection: {e}")
        await q.close()
        return

    if findings_count == 0:
        print("\n⚠️  No vectors loaded yet. Backfill still in progress.")
        print("Check progress: ~/researchgravity/query_research.sh backfill")
        await q.close()
        return

    print(f"\n🔍 Searching findings (with Cohere reranking)...")
    try:
        results = await q.search_findings(
            query,
            limit=5,
            rerank=True,
            min_score=0.3
        )

        if results:
            for i, r in enumerate(results, 1):
                score = r.get('relevance_score', r.get('score', 0))
                content = r.get('content', '')[:150]
                session = r.get('session_id', 'unknown')[:35]
                finding_type = r.get('type', 'unknown')
                print(f"\n{i}. [Score: {score:.3f}] Type: {finding_type}")
                print(f"   Session: {session}")
                print(f"   {content}...")
        else:
            print("No results found")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    await q.close()


async def compare_search_modes(query: str = "agentic orchestration"):
    """Compare vector search vs reranked search."""
    print(f"\n━━━ Search Comparison: '{query}' ━━━━━━━━━━━━━━━")

    q = await get_qdrant()

    # Vector search only
    print("\n1️⃣  Vector Search Only (Cosine Similarity):")
    results_vector = await q.search_findings(query, limit=3, rerank=False)
    for i, r in enumerate(results_vector, 1):
        score = r.get('score', 0)
        content = r.get('content', '')[:80]
        print(f"   {i}. [{score:.3f}] {content}...")

    # With reranking
    print("\n2️⃣  Vector + Reranking (Cohere rerank-v3.5):")
    results_reranked = await q.search_findings(query, limit=3, rerank=True)
    for i, r in enumerate(results_reranked, 1):
        score = r.get('relevance_score', 0)
        content = r.get('content', '')[:80]
        print(f"   {i}. [{score:.3f}] {content}...")

    await q.close()


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "multi-agent consensus"

    if len(sys.argv) > 2 and sys.argv[2] == "--compare":
        asyncio.run(compare_search_modes(query))
    else:
        asyncio.run(test_search(query))
