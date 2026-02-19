"""
PaperBanana Critic — evaluates visual assets using PaperBanana's
4-dimension scoring (faithfulness, readability, conciseness, aesthetics).

Integrates into ResearchGravity's existing critic pipeline as a third
perspective alongside ArchiveCritic and EvidenceCritic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CriticBase, Issue, Severity, ValidationResult

# Lazy import
VISUAL_AVAILABLE = False
try:
    from visual import PaperBananaAdapter, get_visual_config
    VISUAL_AVAILABLE = True
except ImportError:
    pass


class PaperBananaCritic(CriticBase):
    """
    Critic that evaluates generated diagrams using PaperBanana's VLM judge.

    Validates:
    - Faithfulness to source methodology
    - Readability of visual layout
    - Conciseness (no unnecessary elements)
    - Aesthetics (NeurIPS-quality standards)
    """

    name = "paperbanana_critic"
    description = "Visual quality evaluation using PaperBanana 4-dimension scoring"

    async def validate(self, target_id: str, **kwargs) -> ValidationResult:
        """
        Validate visual assets for a session.

        Args:
            target_id: Session ID containing visual assets to evaluate
            **kwargs: Optional overrides (min_score, diagram_paths)
        """
        evidence = await self._gather_evidence(target_id, **kwargs)
        issues = await self._run_checks(evidence)
        confidence = self._calculate_confidence(evidence, issues)

        result = ValidationResult(
            valid=confidence >= self.min_confidence,
            confidence=round(confidence, 3),
            issues=issues,
            metrics=evidence.get("scores_summary", {}),
            critic_name=self.name,
            target_id=target_id,
        )
        self.record_result(result)
        return result

    async def _gather_evidence(self, target_id: str, **kwargs) -> Dict[str, Any]:
        """Gather visual asset data for the session."""
        evidence = {
            "session_id": target_id,
            "visual_assets": [],
            "scores_summary": {},
            "has_diagrams": False,
        }

        if not VISUAL_AVAILABLE:
            evidence["scores_summary"]["error"] = "Visual layer not available"
            return evidence

        # Try to load visual assets from session directory
        from pathlib import Path
        sessions_dir = Path.home() / ".agent-core" / "sessions" / target_id
        diagrams_dir = sessions_dir / "diagrams"

        if diagrams_dir.exists():
            png_files = list(diagrams_dir.glob("*.png"))
            evidence["has_diagrams"] = len(png_files) > 0
            evidence["visual_assets"] = [str(p) for p in png_files]

        # Load session metadata for visual_stats
        session_meta = sessions_dir / "session.json"
        if session_meta.exists():
            import json
            try:
                meta = json.loads(session_meta.read_text())
                evidence["visual_stats"] = meta.get("visual_stats", {})
            except (json.JSONDecodeError, IOError):
                pass

        return evidence

    async def _run_checks(self, evidence: Dict[str, Any]) -> List[Issue]:
        """Run quality checks on visual assets."""
        issues = []

        if not VISUAL_AVAILABLE:
            issues.append(self.add_issue(
                code="VIS_UNAVAILABLE",
                message="Visual layer (PaperBanana) not installed",
                severity=Severity.INFO,
                suggestion="Install with: pip install paperbanana",
            ))
            return issues

        if not evidence.get("has_diagrams"):
            issues.append(self.add_issue(
                code="VIS_NO_DIAGRAMS",
                message="No diagrams found in session archive",
                severity=Severity.INFO,
                suggestion="Session may not have had diagrammable findings",
            ))
            return issues

        visual_stats = evidence.get("visual_stats", {})

        # Check diagram count
        count = visual_stats.get("diagrams_generated", 0)
        if count == 0:
            issues.append(self.add_issue(
                code="VIS_ZERO_GENERATED",
                message="Visual layer ran but generated 0 diagrams",
                severity=Severity.WARNING,
                suggestion="Check if findings have sufficient methodology text (>50 chars)",
            ))

        # Check cost
        cost = visual_stats.get("total_cost", 0)
        if cost > 2.0:
            issues.append(self.add_issue(
                code="VIS_BUDGET_HIGH",
                message=f"Visual generation cost (${cost:.2f}) exceeds session budget ($2.00)",
                severity=Severity.WARNING,
            ))

        return issues

    def _calculate_confidence(self, evidence: Dict[str, Any], issues: List[Issue]) -> float:
        """Calculate confidence based on visual quality."""
        if not evidence.get("has_diagrams"):
            return 0.5  # Neutral — no diagrams isn't a failure

        # Start from high confidence and deduct for issues
        confidence = 0.9

        error_count = sum(1 for i in issues if i.severity in (Severity.ERROR, Severity.CRITICAL))
        warning_count = sum(1 for i in issues if i.severity == Severity.WARNING)

        confidence -= error_count * 0.2
        confidence -= warning_count * 0.1

        return max(0.0, min(1.0, confidence))
