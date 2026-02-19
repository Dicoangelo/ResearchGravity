"""
Visual Intelligence Layer — triple-engine image generation for ResearchGravity.

Three image generation backends:

1. Refined Pipeline (recommended) — PaperBanana-style 5-agent architecture
   (Planner → Stylist → [Visualizer → Critic] × T rounds)
   Key: Critic refines TEXTUAL DESCRIPTION, each round regenerates from scratch.
   Best for: architecture diagrams, technical illustrations, any complex visualization

2. Gemini Native ImageGen — direct google-genai SDK (single-shot)
   (gemini-3-pro-image-preview, 1K-4K, aspect ratios, multi-image editing)
   Best for: brand assets, scene generation, image editing/composition, logos

3. PaperBanana Adapter — wraps PaperBanana's 5-agent pipeline with Qdrant refs
   Best for: methodology diagrams when PaperBanana package is installed

All engines share the same config system (profiles, cost tracking, UCW capture).
"""

from .config import get_visual_config, list_models, list_profiles, VisualConfig
from .gemini_native import GeminiImageGenerator
from .paperbanana_adapter import PaperBananaAdapter
from .refined_pipeline import RefinedPipeline

__all__ = [
    "RefinedPipeline",
    "PaperBananaAdapter",
    "GeminiImageGenerator",
    "get_visual_config",
    "list_profiles",
    "list_models",
    "VisualConfig",
]
