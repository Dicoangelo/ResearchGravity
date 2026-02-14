"""
4Ds Framework — Anthropic's Responsible AI Delegation Gates

Implements Anthropic's 4Ds framework for human-AI collaboration:
- Delegation: What tasks should be delegated to AI?
- Description: How well are task requirements communicated?
- Discernment: How do we evaluate AI outputs?
- Diligence: What ethical/safety constraints apply?

Reference: Anthropic AI Fluency Framework
See: https://www.anthropic.com/research/ai-fluency

Each gate returns a tuple of (bool/float, str/List[str]) with decision and reasoning.
All evaluations are stored as DelegationEvent entries with gate_type field.

Usage:
    from delegation.four_ds import FourDsGate

    gate = FourDsGate()

    # Check if task should be delegated
    approved, reason = gate.delegation_gate(task_desc, profile)
    if not approved:
        print(f"Delegation blocked: {reason}")

    # Score description quality
    score, suggestions = gate.description_gate(task_desc)
    if score < 0.6:
        print(f"Improve description: {suggestions}")

    # Verify output quality
    quality, issues = gate.discernment_gate(output, expected, profile)
    if quality < 0.7:
        print(f"Output needs review: {issues}")

    # Check ethical constraints
    safe, warnings = gate.diligence_gate(task_desc, profile)
    if not safe:
        print(f"Safety warnings: {warnings}")
"""

import hashlib
import json
import re
import asyncio
import sqlite3
from typing import Tuple, List, Optional, Any
from pathlib import Path
from .models import TaskProfile, DelegationEvent
import time
import uuid

# Try to import LLM client for enhanced analysis
try:
    from cpb.llm_client import get_llm_client, LLMRequest
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False


class FourDsGate:
    """
    Anthropic's 4Ds Framework gates for responsible AI delegation.

    Implements four decision gates:
    1. Delegation: Is this appropriate for AI?
    2. Description: Is the task well-specified?
    3. Discernment: Is the output acceptable?
    4. Diligence: Are ethical constraints satisfied?
    """

    def __init__(self, db_path: str = ""):
        """
        Initialize 4Ds gate.

        Args:
            db_path: Path to SQLite database for event logging
        """
        if not db_path:
            db_path = str(Path.home() / ".agent-core" / "storage" / "delegation_events.db")
        self.db_path = db_path

    # ========================================================================
    # GATE 1: DELEGATION
    # ========================================================================

    def delegation_gate(
        self,
        task: str,
        profile: TaskProfile
    ) -> Tuple[bool, str]:
        """
        Gate 1: Should this task be delegated to AI?

        Blocks tasks that are:
        - High subjectivity (> 0.7) AND high criticality (> 0.8) AND low reversibility (< 0.2)
        - These combinations require human judgment

        Args:
            task: Task description
            profile: Task profile with 11-dimensional scores

        Returns:
            Tuple of (approved: bool, reason: str)
        """
        # High-risk combination: subjective + critical + irreversible
        high_risk = (
            profile.subjectivity > 0.7 and
            profile.criticality > 0.8 and
            profile.reversibility < 0.2
        )

        if high_risk:
            reason = (
                f"Task blocked: high subjectivity ({profile.subjectivity:.2f}) + "
                f"high criticality ({profile.criticality:.2f}) + "
                f"low reversibility ({profile.reversibility:.2f}) requires human judgment"
            )
            self._log_event(
                task_id=self._hash_task(task),
                event_type="delegation_gate",
                status="blocked",
                details={
                    "gate": "delegation",
                    "approved": False,
                    "reason": reason,
                    "subjectivity": profile.subjectivity,
                    "criticality": profile.criticality,
                    "reversibility": profile.reversibility
                }
            )
            return False, reason

        # Additional check: critical + low verifiability OR critical + irreversible
        if profile.criticality >= 0.8 and (profile.verifiability < 0.3 or profile.reversibility < 0.3):
            if profile.verifiability < 0.3:
                reason = (
                    f"Task blocked: high criticality ({profile.criticality:.2f}) + "
                    f"low verifiability ({profile.verifiability:.2f}) makes validation difficult"
                )
            else:
                reason = (
                    f"Task blocked: high criticality ({profile.criticality:.2f}) + "
                    f"low reversibility ({profile.reversibility:.2f}) makes errors costly"
                )
            self._log_event(
                task_id=self._hash_task(task),
                event_type="delegation_gate",
                status="blocked",
                details={
                    "gate": "delegation",
                    "approved": False,
                    "reason": reason,
                    "criticality": profile.criticality,
                    "verifiability": profile.verifiability
                }
            )
            return False, reason

        # Task approved for delegation
        reason = "Task approved: risk factors within acceptable bounds"
        self._log_event(
            task_id=self._hash_task(task),
            event_type="delegation_gate",
            status="approved",
            details={
                "gate": "delegation",
                "approved": True,
                "reason": reason
            }
        )
        return True, reason

    # ========================================================================
    # GATE 2: DESCRIPTION
    # ========================================================================

    def description_gate(
        self,
        task_description: str,
        use_llm: bool = True
    ) -> Tuple[float, str]:
        """
        Gate 2: How well is this task described?

        Scores description quality on:
        - Specificity: Clear, concrete requirements (40%)
        - Completeness: All necessary context provided (30%)
        - Constraint clarity: Explicit success criteria (30%)

        Returns score [0.0, 1.0] and enhancement suggestions.
        If score < 0.6, task description should be improved.

        Args:
            task_description: Task description text
            use_llm: Use LLM for enhanced analysis (default: True)

        Returns:
            Tuple of (score: float, suggestions: str)
        """
        if use_llm and HAS_LLM_CLIENT:
            try:
                score, suggestions = asyncio.run(
                    asyncio.wait_for(
                        self._llm_description_analysis(task_description),
                        timeout=3.0
                    )
                )
                self._log_event(
                    task_id=self._hash_task(task_description),
                    event_type="description_gate",
                    status="analyzed",
                    details={
                        "gate": "description",
                        "score": score,
                        "suggestions": suggestions,
                        "method": "llm"
                    }
                )
                return score, suggestions
            except Exception:
                pass  # Fall back to heuristic

        # Heuristic description scoring
        score, suggestions = self._heuristic_description_score(task_description)
        self._log_event(
            task_id=self._hash_task(task_description),
            event_type="description_gate",
            status="analyzed",
            details={
                "gate": "description",
                "score": score,
                "suggestions": suggestions,
                "method": "heuristic"
            }
        )
        return score, suggestions

    def _heuristic_description_score(self, description: str) -> Tuple[float, str]:
        """Heuristic scoring for task description quality."""
        suggestions = []
        scores = []

        # Specificity: Look for vague language
        vague_words = ["thing", "stuff", "something", "somehow", "figure out", "handle", "deal with"]
        has_vague = any(word in description.lower() for word in vague_words)

        # Look for specific terms
        specific_indicators = ["implement", "create", "build", "analyze", "verify", "test"]
        has_specific = any(word in description.lower() for word in specific_indicators)

        specificity = 0.3 if has_vague else (0.8 if has_specific else 0.5)
        scores.append(specificity * 0.4)  # 40% weight

        if has_vague:
            suggestions.append("Replace vague language with specific requirements")
        if not has_specific:
            suggestions.append("Add concrete action verbs (implement, create, analyze)")

        # Completeness: Check length and detail
        word_count = len(description.split())
        if word_count < 5:
            completeness = 0.2
            suggestions.append("Provide more context and details")
        elif word_count < 15:
            completeness = 0.5
            suggestions.append("Add more context about requirements and constraints")
        else:
            completeness = 0.8
        scores.append(completeness * 0.3)  # 30% weight

        # Constraint clarity: Look for success criteria
        has_criteria = any(word in description.lower() for word in ["should", "must", "verify", "test", "expect", "ensure", "include", "output"])
        has_metrics = any(char in description for char in ["<", ">", "=", "%"]) or any(word in description.lower() for word in ["at least", "minimum", "maximum"])

        constraint_clarity = 0.8 if (has_criteria and has_metrics) else (0.6 if has_criteria else 0.3)
        scores.append(constraint_clarity * 0.3)  # 30% weight

        if not has_criteria:
            suggestions.append("Define success criteria (what should the output satisfy?)")
        if not has_metrics:
            suggestions.append("Add measurable constraints where applicable")

        total_score = max(0.0, min(1.0, sum(scores)))

        if total_score >= 0.8:
            suggestion_text = "Description is clear and complete"
        elif total_score >= 0.6:
            suggestion_text = "Good description. Consider: " + "; ".join(suggestions)
        else:
            suggestion_text = "Improve description: " + "; ".join(suggestions)

        return total_score, suggestion_text

    async def _llm_description_analysis(self, description: str) -> Tuple[float, str]:
        """LLM-based description quality analysis."""
        client = await get_llm_client()

        request = LLMRequest(
            system_prompt="""You are a task description quality analyzer.
Score descriptions on specificity, completeness, and constraint clarity.
Output JSON only: {"score": 0.0-1.0, "suggestions": "string"}""",
            user_prompt=f"""Analyze this task description:

"{description}"

Score on:
- Specificity (40%): Concrete vs vague language
- Completeness (30%): Sufficient context
- Constraint clarity (30%): Clear success criteria

Return JSON: {{"score": 0.85, "suggestions": "suggestion text"}}""",
            temperature=0.3,
            model="haiku"
        )

        response = await client.generate(request)

        # Parse JSON from response
        json_match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            suggestions = data.get("suggestions", "")
            return score, suggestions

        return 0.5, "Unable to analyze"

    # ========================================================================
    # GATE 3: DISCERNMENT
    # ========================================================================

    def discernment_gate(
        self,
        output: str,
        expected: str,
        profile: TaskProfile
    ) -> Tuple[float, List[str]]:
        """
        Gate 3: Is this AI output acceptable?

        Scores output quality and identifies issues.
        Outputs scoring < 0.7 should be flagged for human review.

        Evaluation criteria:
        - Completeness: Addresses all requirements (40%)
        - Correctness: Free from obvious errors (30%)
        - Consistency: Matches expected format/style (30%)

        Args:
            output: AI-generated output
            expected: Expected output format/requirements
            profile: Task profile for context

        Returns:
            Tuple of (quality_score: float, issues: List[str])
        """
        issues = []
        scores = []

        # Completeness: Compare length and keywords
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())
        keyword_overlap = len(output_words & expected_words) / max(len(expected_words), 1)

        completeness = min(1.0, keyword_overlap + 0.3)  # Boost base score
        scores.append(completeness * 0.4)  # 40% weight

        if completeness < 0.5:
            issues.append(f"Low completeness ({completeness:.2f}): output may be missing key requirements")

        # Correctness: Look for error indicators
        error_indicators = ["error", "failed", "exception", "undefined", "null", "nan", "invalid"]
        has_errors = any(indicator in output.lower() for indicator in error_indicators)

        correctness = 0.3 if has_errors else 0.8
        scores.append(correctness * 0.3)  # 30% weight

        if has_errors:
            issues.append("Output contains error indicators")

        # Consistency: Check if output is too short or too long
        length_ratio = len(output) / max(len(expected), 1)
        if length_ratio < 0.3:
            consistency = 0.4
            issues.append("Output significantly shorter than expected")
        elif length_ratio > 3.0:
            consistency = 0.6
            issues.append("Output significantly longer than expected")
        else:
            consistency = 0.8
        scores.append(consistency * 0.3)  # 30% weight

        total_score = max(0.0, min(1.0, sum(scores)))

        # Flag for review if score < 0.7
        if total_score < 0.7:
            issues.insert(0, f"Quality score {total_score:.2f} < 0.7 threshold — flagged for human review")

        if not issues:
            issues.append("Output quality acceptable")

        self._log_event(
            task_id=self._hash_task(output[:100]),  # Use first 100 chars as task ID
            event_type="discernment_gate",
            status="reviewed" if total_score >= 0.7 else "flagged",
            details={
                "gate": "discernment",
                "quality_score": total_score,
                "issues": issues,
                "completeness": completeness,
                "correctness": correctness,
                "consistency": consistency
            }
        )

        return total_score, issues

    # ========================================================================
    # GATE 4: DILIGENCE
    # ========================================================================

    def diligence_gate(
        self,
        task: str,
        profile: TaskProfile
    ) -> Tuple[bool, List[str]]:
        """
        Gate 4: Are ethical and safety constraints satisfied?

        Checks for:
        - Data sensitivity: PII, credentials, secrets
        - Potential harm: Destructive actions, irreversible changes
        - Reversibility: Can mistakes be undone?

        Args:
            task: Task description
            profile: Task profile with risk scores

        Returns:
            Tuple of (safe: bool, warnings: List[str])
        """
        warnings = []

        # Check for data sensitivity
        sensitive_keywords = [
            "password", "credential", "secret", "api_key", "token", "private_key",
            "ssn", "credit_card", "personal", "pii", "confidential"
        ]
        has_sensitive_data = any(keyword in task.lower() for keyword in sensitive_keywords)

        if has_sensitive_data:
            warnings.append("Task involves sensitive data — ensure proper access controls")

        # Check for destructive operations
        destructive_keywords = [
            "delete", "drop", "remove", "destroy", "wipe", "erase",
            "truncate", "clear", "purge", "reset"
        ]
        is_destructive = any(keyword in task.lower() for keyword in destructive_keywords)

        if is_destructive and profile.reversibility < 0.5:
            warnings.append(f"Destructive operation with low reversibility ({profile.reversibility:.2f}) — high risk")

        # Check for irreversible high-criticality tasks
        if profile.criticality > 0.8 and profile.reversibility < 0.3:
            warnings.append(
                f"High criticality ({profile.criticality:.2f}) + "
                f"low reversibility ({profile.reversibility:.2f}) — consider human oversight"
            )

        # Check for deployment/production keywords
        production_keywords = ["deploy", "production", "release", "publish", "launch"]
        is_production = any(keyword in task.lower() for keyword in production_keywords)

        if is_production and profile.verifiability <= 0.6:
            warnings.append(
                f"Production deployment with low verifiability ({profile.verifiability:.2f}) — "
                "ensure thorough testing"
            )

        # Determine if task is safe
        # Block if: (sensitive + destructive + irreversible) OR (destructive keywords + very low reversibility)
        unsafe = (
            (has_sensitive_data and is_destructive and profile.reversibility < 0.2) or
            (is_destructive and profile.reversibility < 0.15)
        )

        if unsafe:
            if has_sensitive_data:
                warnings.insert(0, "BLOCKED: Sensitive + destructive + irreversible combination")
            else:
                warnings.insert(0, "BLOCKED: Destructive operation with critically low reversibility")
            safe = False
        else:
            safe = True
            if not warnings:
                warnings.append("No ethical or safety concerns detected")

        self._log_event(
            task_id=self._hash_task(task),
            event_type="diligence_gate",
            status="blocked" if not safe else ("warning" if len(warnings) > 1 else "safe"),
            details={
                "gate": "diligence",
                "safe": safe,
                "warnings": warnings,
                "has_sensitive_data": has_sensitive_data,
                "is_destructive": is_destructive,
                "is_production": is_production
            }
        )

        return safe, warnings

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _hash_task(self, task: str) -> str:
        """Generate a short hash for task identification."""
        return hashlib.md5(task.encode()).hexdigest()[:8]

    def _log_event(
        self,
        task_id: str,
        event_type: str,
        status: str,
        details: dict
    ):
        """Log a 4Ds gate event to database (sync — safe from any context)."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path, timeout=1.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS delegation_events (
                    event_id TEXT PRIMARY KEY,
                    delegation_id TEXT,
                    timestamp REAL,
                    event_type TEXT,
                    agent_id TEXT,
                    task_id TEXT,
                    status TEXT,
                    gate_type TEXT,
                    details TEXT
                )
            """)

            conn.execute("""
                INSERT INTO delegation_events (
                    event_id, delegation_id, timestamp, event_type,
                    agent_id, task_id, status, gate_type, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                uuid.uuid4().hex[:8],
                "4ds-gate",
                time.time(),
                event_type,
                "4ds-gate-system",
                task_id,
                status,
                details.get("gate", ""),
                json.dumps(details)
            ))

            conn.commit()
            conn.close()
        except Exception:
            pass  # Never block delegation on logging failure


# ============================================================================
# PUBLIC API
# ============================================================================

def delegation_gate(task: str, profile: TaskProfile) -> Tuple[bool, str]:
    """
    Gate 1: Is this task appropriate for AI delegation?

    Blocks high-risk tasks:
    - High subjectivity + high criticality + low reversibility
    - High criticality + low verifiability

    Args:
        task: Task description
        profile: 11-dimensional task profile

    Returns:
        Tuple of (approved: bool, reason: str)
    """
    gate = FourDsGate()
    return gate.delegation_gate(task, profile)


def description_gate(task_description: str, use_llm: bool = True) -> Tuple[float, str]:
    """
    Gate 2: How well is this task described?

    Scores description quality [0.0, 1.0] on:
    - Specificity (40%)
    - Completeness (30%)
    - Constraint clarity (30%)

    Args:
        task_description: Task description text
        use_llm: Use LLM for enhanced analysis

    Returns:
        Tuple of (score: float, suggestions: str)
    """
    gate = FourDsGate()
    return gate.description_gate(task_description, use_llm=use_llm)


def discernment_gate(
    output: str,
    expected: str,
    profile: TaskProfile
) -> Tuple[float, List[str]]:
    """
    Gate 3: Is this AI output acceptable?

    Scores output quality [0.0, 1.0] and flags issues.
    Outputs < 0.7 flagged for human review.

    Args:
        output: AI-generated output
        expected: Expected output format/requirements
        profile: Task profile for context

    Returns:
        Tuple of (quality_score: float, issues: List[str])
    """
    gate = FourDsGate()
    return gate.discernment_gate(output, expected, profile)


def diligence_gate(task: str, profile: TaskProfile) -> Tuple[bool, List[str]]:
    """
    Gate 4: Are ethical and safety constraints satisfied?

    Checks for:
    - Data sensitivity (PII, credentials)
    - Destructive operations
    - Reversibility of changes

    Args:
        task: Task description
        profile: Task profile with risk scores

    Returns:
        Tuple of (safe: bool, warnings: List[str])
    """
    gate = FourDsGate()
    return gate.diligence_gate(task, profile)
