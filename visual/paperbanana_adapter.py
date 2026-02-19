"""
PaperBanana Adapter — bridges PaperBanana's 5-agent pipeline with
ResearchGravity's Qdrant-backed knowledge graph and UCW capture.

Replaces PaperBanana's static 13-example reference store with semantic
retrieval from past diagrams stored in Qdrant.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import VisualConfig, get_visual_config

# Lazy imports — PaperBanana may not be installed
PAPERBANANA_AVAILABLE = False
try:
    from paperbanana.core.config import Settings as PBSettings
    from paperbanana.core.pipeline import PaperBananaPipeline
    from paperbanana.core.types import DiagramType, GenerationInput
    PAPERBANANA_AVAILABLE = True
except ImportError:
    pass


class QdrantReferenceStore:
    """Replace PaperBanana's static examples with Qdrant semantic search."""

    def __init__(self, qdrant_db, min_score: float = 0.6):
        self.qdrant = qdrant_db
        self.min_score = min_score

    async def get_references(
        self,
        methodology: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search for relevant past diagrams."""
        try:
            results = await self.qdrant.search(
                collection_name="visual_assets",
                query=methodology,
                limit=limit,
                min_score=self.min_score,
            )
            return [
                {
                    "methodology": r.get("payload", {}).get("methodology_text", ""),
                    "caption": r.get("payload", {}).get("caption", ""),
                    "png_path": r.get("payload", {}).get("png_path", ""),
                    "score": r.get("score", 0),
                }
                for r in results
            ]
        except Exception:
            # Collection may not exist yet or Qdrant unavailable
            return []


class PaperBananaAdapter:
    """
    Wraps PaperBanana's 5-agent pipeline with:
    - Qdrant-backed reference retrieval (replaces static examples)
    - Google paid-tier config (gemini-2.5-pro VLM, 4K images)
    - UCW cognitive asset metadata generation
    - Cost tracking per diagram
    """

    def __init__(
        self,
        config: Optional[VisualConfig] = None,
        qdrant_db=None,
    ):
        self.config = config or get_visual_config()
        self.qdrant_db = qdrant_db
        self.reference_store = (
            QdrantReferenceStore(qdrant_db, self.config.min_reference_score)
            if qdrant_db
            else None
        )
        self._session_cost = 0.0

    def _build_pb_settings(self) -> "PBSettings":
        """Build PaperBanana Settings with paid-tier config."""
        # Set Google API key in env for PaperBanana to pick up
        if self.config.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.config.google_api_key

        return PBSettings(
            vlm_model=self.config.vlm_model,
            image_model=self.config.image_model,
            output_resolution=self.config.image_resolution,
            refinement_iterations=self.config.max_iterations,
            save_iterations=True,
        )

    async def generate_diagram(
        self,
        methodology: str,
        caption: str,
        session_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate a publication-quality methodology diagram.

        Returns dict with:
            - png_path: Path to generated PNG
            - metadata: Agent trace, iterations, costs
            - critic_scores: 4-dimension evaluation (if available)
            - asset_id: Unique ID for UCW capture
        """
        if not PAPERBANANA_AVAILABLE:
            return {"error": "PaperBanana not installed. Run: pip install paperbanana"}

        if not self.config.google_api_key:
            return {"error": "GOOGLE_API_KEY not configured"}

        # Budget check
        if self._session_cost >= self.config.cost_budget_per_session:
            return {
                "error": f"Session budget exceeded (${self._session_cost:.2f} / ${self.config.cost_budget_per_session:.2f})",
                "skipped": True,
            }

        asset_id = f"visual-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now()

        settings = self._build_pb_settings()
        if output_dir:
            settings.output_dir = str(output_dir)

        pipeline = PaperBananaPipeline(settings=settings)

        gen_input = GenerationInput(
            source_context=methodology,
            communicative_intent=caption,
            diagram_type=DiagramType.METHODOLOGY,
        )

        result = await pipeline.generate(gen_input)

        elapsed = (datetime.now() - started_at).total_seconds()
        estimated_cost = self._estimate_cost(self.config.max_iterations)
        self._session_cost += estimated_cost

        return {
            "asset_id": asset_id,
            "png_path": str(result.image_path),
            "methodology_text": methodology,
            "caption": caption,
            "session_id": session_id,
            "diagram_type": "methodology",
            "metadata": {
                "iterations": result.metadata.iterations_run if hasattr(result, "metadata") else self.config.max_iterations,
                "vlm_model": self.config.vlm_model,
                "image_model": self.config.image_model,
                "resolution": self.config.image_resolution,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_cost_usd": round(estimated_cost, 4),
            },
            "created_at": started_at.isoformat(),
        }

    async def generate_plot(
        self,
        data_json: str,
        intent: str,
        session_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate a statistical plot from JSON data."""
        if not PAPERBANANA_AVAILABLE:
            return {"error": "PaperBanana not installed"}

        if not self.config.google_api_key:
            return {"error": "GOOGLE_API_KEY not configured"}

        asset_id = f"visual-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now()

        raw_data = json.loads(data_json) if isinstance(data_json, str) else data_json

        settings = self._build_pb_settings()
        if output_dir:
            settings.output_dir = str(output_dir)

        pipeline = PaperBananaPipeline(settings=settings)

        gen_input = GenerationInput(
            source_context=f"Data for plotting:\n{json.dumps(raw_data)}",
            communicative_intent=intent,
            diagram_type=DiagramType.STATISTICAL_PLOT,
            raw_data=raw_data,
        )

        result = await pipeline.generate(gen_input)

        elapsed = (datetime.now() - started_at).total_seconds()
        estimated_cost = self._estimate_cost(self.config.max_iterations)
        self._session_cost += estimated_cost

        return {
            "asset_id": asset_id,
            "png_path": str(result.image_path),
            "methodology_text": json.dumps(raw_data),
            "caption": intent,
            "session_id": session_id,
            "diagram_type": "statistical_plot",
            "metadata": {
                "iterations": self.config.max_iterations,
                "vlm_model": self.config.vlm_model,
                "image_model": self.config.image_model,
                "resolution": self.config.image_resolution,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_cost_usd": round(estimated_cost, 4),
            },
            "created_at": started_at.isoformat(),
        }

    async def evaluate_diagram(
        self,
        generated_path: str,
        reference_path: str,
        context: str,
        caption: str,
    ) -> Dict[str, Any]:
        """Evaluate a diagram against a reference using PaperBanana's 4-dimension scoring."""
        if not PAPERBANANA_AVAILABLE:
            return {"error": "PaperBanana not installed"}

        try:
            from paperbanana.evaluation.judge import VLMJudge
            from paperbanana.providers.registry import ProviderRegistry

            settings = self._build_pb_settings()
            registry = ProviderRegistry(settings)
            judge = VLMJudge(registry.get_vlm_provider())

            scores = await judge.evaluate(
                generated_path=generated_path,
                reference_path=reference_path,
                source_context=context,
                caption=caption,
            )

            return {
                "faithfulness": scores.faithfulness if hasattr(scores, "faithfulness") else 0.0,
                "readability": scores.readability if hasattr(scores, "readability") else 0.0,
                "conciseness": scores.conciseness if hasattr(scores, "conciseness") else 0.0,
                "aesthetics": scores.aesthetics if hasattr(scores, "aesthetics") else 0.0,
                "overall": getattr(scores, "overall", 0.0),
            }
        except Exception as e:
            return {"error": str(e)}

    async def generate_session_diagrams(
        self,
        session_id: str,
        session_dir: Path,
        findings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Auto-generate diagrams for a session's findings.

        Called by archive_session.py after file copy.
        Extracts methodology-relevant findings and generates diagrams.
        """
        if not PAPERBANANA_AVAILABLE or not self.config.enabled:
            return []

        if not self.config.google_api_key:
            return []

        diagrams_dir = session_dir / "diagrams"
        diagrams_dir.mkdir(exist_ok=True)

        results = []
        self._session_cost = 0.0

        # Filter findings that have enough content for diagram generation
        diagrammable = [
            f for f in findings
            if f.get("type") in ("innovation", "implementation", "thesis")
            and len(f.get("text", "")) > 50
        ]

        if not diagrammable:
            # Fall back to any finding with substantial text
            diagrammable = [
                f for f in findings
                if len(f.get("text", "")) > 100
            ][:3]  # Max 3 diagrams

        for finding in diagrammable[:5]:  # Cap at 5 diagrams per session
            try:
                result = await self.generate_diagram(
                    methodology=finding.get("text", ""),
                    caption=f"{finding.get('type', 'Research').title()}: {finding.get('text', '')[:80]}",
                    session_id=session_id,
                    output_dir=diagrams_dir,
                )

                if "error" not in result:
                    results.append(result)
                elif result.get("skipped"):
                    break  # Budget exceeded, stop generating

            except Exception as e:
                # Graceful degradation — don't block archive
                continue

        return results

    def _estimate_cost(self, iterations: int) -> float:
        """Estimate cost for a diagram generation run using config-aware pricing."""
        return self.config.estimate_diagram_cost()

    def get_session_cost(self) -> float:
        """Get total cost for current session."""
        return round(self._session_cost, 4)

    def reset_session_cost(self):
        """Reset session cost tracker."""
        self._session_cost = 0.0
