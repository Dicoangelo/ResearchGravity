"""Generate the 3 remaining style variants: dark-premium, radial-flow, freeflow."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from visual.refined_pipeline import RefinedPipeline
from visual.config import get_visual_config

SOURCE_CONTEXT = """
ResearchGravity v6.1 — Metaventions AI Research Framework
Production Status | 6-Tier Sovereign Research Intelligence System

SYSTEM STATISTICS:
- 114+ archived sessions
- 2,530+ findings captured
- 8,935+ URLs tracked
- 27M+ tokens processed
- 11,579 graph nodes
- 13,744 graph edges
- 3 active Writer-Critics

CROSS-CUTTING: MCP SERVER (Left Side)
Claude Desktop Integration (21+ tools):
- get_session_context
- search_learnings
- visualize_research
- log_finding
- select_context_packs
- list_visual_profiles
- illustrate_finding

TIER 1: SIGNAL CAPTURE — Entry Layer / Raw Signals
Components:
- Interactive REPL (repl.py) — Rich terminal UI
- Auto-Capture V2 (auto_capture_v2.py) — From Claude sessions (+70% capture rate)
- File Watcher (watcher.py) — Background daemon
- Session Tracker — Tracks transcripts & URLs
- Manual Logging (log_url.py) — Tier 1/2/3 classification
Input sources: arXiv, HuggingFace, Google AI, Meta AI, Anthropic

TIER 2: INTELLIGENCE ENGINE — The Brain / Orchestration
Sub: The Brain / Orchestration
Models (CPB - Cognitive Precision Bridge):
- DIRECT (v0.2) — Direct routing
- RLM (R-2.0.5) — Reasoning Language Model
- ACE (8-5.6.7) — Adaptive Consensus Engine, 7-agent cascade
- HYBRID (v0.7) — Hybrid routing
- CASCADE (v0.7) — Cascade routing
Meta-Learning Engine: 666+ outcomes, 1,014 cognitive states
Intelligence CLI/API: predict, optimal-time, endpoints
DQ Scoring: Validity (40%) + Specificity (30%) + Correctness (30%)
Precision Mode v2.5: Tiered Search, Context Grounding, 7-Agent Cascade, MAR Consensus, Targeted Refinement, Editorial Frame

TIER 3: STORAGE TRIAD — Synchronized Persistence
Components:
- Qdrant — Vector Search (Cohere embed-v4.0, 1024d), hosted semantic search
- sqlite-vec — Local vector fallback (single-file, offline capable)
- SQLite — WAL mode, FTS5, full-text search, relational storage
Embedding List: Cohere v4 (1536) → Cohere v3 (1024d) → SBERT (384d)
Dual-Write Engine writes to Qdrant + sqlite-vec simultaneously

TIER 4: KNOWLEDGE GRAPH — Relationship Intelligence
Graph Intelligence: 11,579 nodes, 13,744 edges
Components:
- Concept Graph (concept_graph.py) — Node-edge concept relationships
- Lineage Tracking (lineage.py) — LineageNode, LineageEdge
- Unified Research Index (INDEX.md) — Papers/Topics/Session & Paper links
Data paths: Sessions → contains → Findings → cites → Papers

TIER 5: CRITIC VALIDATION — Quality Assurance
Writer-Critic System:
- Archive Critic (archive_critic.py) — Archive completeness
- Evidence Critic (evidence_critic.py) — Citation accuracy
- Pack Critic (pack_critic.py) — Pack relevance
Oracle Consensus: Multi-stream validation (Validity + Evidence + Accuracy)
MAR Consensus: Multi-Agent Review (arXiv:2512.20845)
Evidence Layer: Confidence scoring, Source validation

TIER 6: VISUAL INTELLIGENCE — Image Generation
Pipeline: Planner → Stylist → Visualizer → Critic (iterative)
Engines: Gemini Native ImageGen (1K-4K, multi-aspect, 4 quality profiles)
PaperBanana Adapter: arXiv:2601.23265 refined pipeline
Profiles: max, balanced, fast, budget

CROSS-CUTTING: REST API (Right Side)
FastAPI Server (Port 3847):
- JWT Auth, Rate Limiting, Dead Letter Queue
- 25 REST endpoints
- /api/v2/stats
- /api/v2/intelligence/*
- /api/search/semantic
- Structured Logging, Async Cohere

CROSS-CUTTING: UCW INTEGRATION (Bottom Layer)
Universal Cognitive Wallet — Sovereign data ownership
3 Semantic Layers (applied to every event):
- Instinct Layer — Coherence potential, flow state (163+ coherence moments)
- Light Layer — Intent, insights, key concepts (130,728 embeddings)
- Data Layer — Raw content, tokens, bytes (140,732 events)

INTELLIGENT DELEGATION v0.1.0 (arXiv:2602.11865):
- 11-dimensional TaskProfile
- Bayesian Beta trust scoring
- 4Ds Gates (Delegation, Description, Discernment, Diligence)
- Task decomposition, routing, verification
- Evolution engine with EMA learning
"""

VARIANTS = [
    {
        "name": "dark-premium",
        "filename": "researchgravity_v3_dark_premium_4K.png",
        "caption": (
            "Premium dark-themed architecture diagram of ResearchGravity v6.1 — Sovereign Research Intelligence. "
            "Deep dark background (#0D1117 or #1A1A2E) with glowing accent colors: cyan (#00D9FF) for data flow, "
            "purple (#7C3AED) for intelligence/AI, green (#10B981) for storage, orange (#F59E0B) for critics. "
            "Show all 6 tiers with neon-glow borders and subtle gradient backgrounds. "
            "Intelligence Engine (Tier 2) as the central, largest element with a glowing brain icon. "
            "3D database cylinders for Storage Triad with subtle reflections. "
            "Glowing arrows between tiers showing data flow. "
            "Stats displayed in pill-shaped badges with glow effects. "
            "UCW at bottom as a luminous foundation layer. "
            "Style: Dark mode premium — like a high-end SaaS product landing page or investor deck. "
            "Sophisticated, modern, the kind of diagram that makes people say 'this is a real product.'"
        ),
    },
    {
        "name": "radial-flow",
        "filename": "researchgravity_v3_radial_flow_4K.png",
        "caption": (
            "Radial/circular flow architecture diagram of ResearchGravity v6.1. "
            "Intelligence Engine (Tier 2 - The Brain) at the CENTER as a large circular hub with brain icon. "
            "Other 5 tiers arranged RADIALLY around it like spokes: "
            "Signal Capture (top-left), Storage Triad (top-right), Knowledge Graph (right), "
            "Critic Validation (bottom-right), Visual Intelligence (bottom-left). "
            "Curved data flow arrows flowing from outer tiers INTO the central Intelligence Engine and back out. "
            "MCP Server and REST API as thin arcs on the outer ring. "
            "UCW as the outermost ring wrapping everything — the sovereign foundation. "
            "Key stats in badges near each tier. "
            "Style: Organic, radial, hub-and-spoke — shows Intelligence Engine as the orchestrator. "
            "NOT a grid. Think: neural network visualization meets system architecture."
        ),
    },
    {
        "name": "freeflow",
        "filename": "researchgravity_v3_freeflow_4K.png",
        "caption": (
            "Create the most visually striking and creative architecture diagram you can for "
            "ResearchGravity v6.1 — a 6-tier Sovereign Research Intelligence system. "
            "You have FULL CREATIVE FREEDOM on layout, color palette, visual metaphors, and composition. "
            "The only constraints: (1) include all 6 tiers with their actual component names, "
            "(2) show data flow between tiers, (3) include the key statistics, "
            "(4) show MCP Server and REST API cross-cutting concerns, "
            "(5) UCW (Universal Cognitive Wallet) as a foundation layer. "
            "Push boundaries on visual innovation — try unconventional layouts, creative metaphors, "
            "artistic color choices. Make it the kind of diagram that would win a design award. "
            "Surprise us. Show what's possible when technical accuracy meets creative vision."
        ),
    },
]


async def run_one(variant, config):
    """Run a single variant with its own pipeline instance."""
    pipeline = RefinedPipeline(config=config)
    name = variant["name"]
    print(f"\n{'='*60}")
    print(f"  Generating: {name.upper()}")
    print(f"  Output: {variant['filename']}")
    print(f"{'='*60}\n", flush=True)

    result = await pipeline.generate(
        source_context=SOURCE_CONTEXT,
        caption=variant["caption"],
        resolution="4K",
        aspect_ratio="16:9",
        quality="max",
        iterations=2,
        output_dir="/Users/dicoangelo/Desktop",
        output_filename=variant["filename"],
    )

    if "error" in result:
        print(f"  ERROR ({name}): {result['error']}", flush=True)
        return name, False

    meta = result.get("metadata", {})
    print(f"  DONE ({name}): {result.get('png_path', '?')}", flush=True)
    print(f"  Cost: ${meta.get('estimated_cost_usd', 0):.4f} | Time: {meta.get('elapsed_seconds', 0):.1f}s", flush=True)
    return name, True


async def main():
    config = get_visual_config()
    config.apply_profile("max")
    config.cost_budget_per_session = 15.00

    # Run sequentially to avoid rate limits
    for variant in VARIANTS:
        await run_one(variant, config)

    print(f"\n{'='*60}")
    print(f"  ALL 3 VARIANTS COMPLETE")
    print(f"{'='*60}\n")

    for v in VARIANTS:
        path = f"/Users/dicoangelo/Desktop/{v['filename']}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  {v['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  {v['filename']} (MISSING)")


if __name__ == "__main__":
    asyncio.run(main())
