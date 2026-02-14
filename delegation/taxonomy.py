"""
Task Taxonomy — 11-Dimensional Task Profiling

Implements the task taxonomy framework from arXiv:2602.11865 Section 3.1.

Provides automated task profiling across 11 dimensions:
- Complexity (computational/cognitive load)
- Criticality (impact of failure)
- Uncertainty (requirement ambiguity)
- Duration (time to complete)
- Cost (resource consumption)
- Resource Requirements (external dependencies)
- Constraints (hard constraints)
- Verifiability (ease of verification)
- Reversibility (ease of rollback)
- Contextuality (context dependence)
- Subjectivity (subjective judgment required)

Usage:
    from delegation.taxonomy import classify_task

    task_description = "Implement user authentication system"
    profile = classify_task(task_description)
    print(f"Complexity: {profile.complexity}")
    print(f"Risk Score: {profile.risk_score}")
    print(f"Delegation Overhead: {profile.delegation_overhead}")
"""

import asyncio
import json
import re
from typing import Dict, Optional

from .models import TaskProfile

# Import LLM client from cpb
try:
    from cpb.llm_client import get_llm_client
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False


# =============================================================================
# SCORING RUBRICS (for LLM and heuristics)
# =============================================================================

SCORING_RUBRICS = {
    "complexity": {
        "description": "Computational and cognitive complexity required",
        "scale": {
            0.0: "Trivial lookup or single-step operation",
            0.2: "Simple task with clear steps",
            0.4: "Multi-step task requiring planning",
            0.6: "Complex task with dependencies",
            0.8: "Highly complex requiring expertise",
            1.0: "Novel synthesis or groundbreaking work"
        }
    },
    "criticality": {
        "description": "Impact of failure on system/user",
        "scale": {
            0.0: "No impact if failed",
            0.2: "Minor inconvenience",
            0.4: "Noticeable but not critical",
            0.6: "Important feature affected",
            0.8: "Major system impact",
            1.0: "Mission-critical, catastrophic if failed"
        }
    },
    "uncertainty": {
        "description": "Ambiguity in requirements or approach",
        "scale": {
            0.0: "Completely specified, clear path",
            0.2: "Mostly clear with minor unknowns",
            0.4: "Some ambiguity in approach",
            0.6: "Significant uncertainty in requirements",
            0.8: "Highly ambiguous, exploratory",
            1.0: "Complete unknown, research required"
        }
    },
    "duration": {
        "description": "Estimated time to complete",
        "scale": {
            0.0: "Instant (<1 minute)",
            0.2: "Quick (1-15 minutes)",
            0.4: "Short (15-60 minutes)",
            0.6: "Medium (1-4 hours)",
            0.8: "Long (4-24 hours)",
            1.0: "Very long (>1 day)"
        }
    },
    "cost": {
        "description": "Resource consumption (compute, API calls, etc.)",
        "scale": {
            0.0: "Free, no resources",
            0.2: "Negligible cost",
            0.4: "Low cost",
            0.6: "Moderate cost",
            0.8: "High cost",
            1.0: "Very expensive"
        }
    },
    "resource_requirements": {
        "description": "External dependencies needed",
        "scale": {
            0.0: "No external dependencies",
            0.2: "Minimal dependencies (1-2)",
            0.4: "Some dependencies (3-5)",
            0.6: "Many dependencies (6-10)",
            0.8: "Complex dependency graph",
            1.0: "Extensive dependencies or rare resources"
        }
    },
    "constraints": {
        "description": "Hard constraints on solution approach",
        "scale": {
            0.0: "No constraints, full freedom",
            0.2: "Minimal constraints",
            0.4: "Some constraints",
            0.6: "Multiple constraints",
            0.8: "Heavily constrained",
            1.0: "Extremely constrained, narrow solution space"
        }
    },
    "verifiability": {
        "description": "Ease of verifying correctness (inverted: 0=hard, 1=easy)",
        "scale": {
            0.0: "Cannot verify or extremely difficult",
            0.2: "Very hard to verify",
            0.4: "Moderately difficult",
            0.6: "Can verify with effort",
            0.8: "Easy to verify",
            1.0: "Trivially verifiable (automated tests)"
        }
    },
    "reversibility": {
        "description": "Ease of undoing or rolling back (0=irreversible, 1=easy)",
        "scale": {
            0.0: "Completely irreversible",
            0.2: "Very difficult to reverse",
            0.4: "Difficult but possible",
            0.6: "Reversible with effort",
            0.8: "Easily reversible",
            1.0: "Trivially reversible (version control)"
        }
    },
    "contextuality": {
        "description": "Dependence on external context",
        "scale": {
            0.0: "Context-free, standalone",
            0.2: "Minimal context needed",
            0.4: "Some context required",
            0.6: "Significant context needed",
            0.8: "Highly contextual",
            1.0: "Completely context-dependent"
        }
    },
    "subjectivity": {
        "description": "Degree of subjective judgment required",
        "scale": {
            0.0: "Purely objective, deterministic",
            0.2: "Mostly objective",
            0.4: "Some subjective elements",
            0.6: "Balanced objective/subjective",
            0.8: "Highly subjective",
            1.0: "Purely subjective judgment"
        }
    }
}


# =============================================================================
# HEURISTIC SCORING
# =============================================================================

# Keyword patterns for heuristic scoring
COMPLEXITY_KEYWORDS = {
    "high": ["implement", "design", "architect", "optimize", "refactor", "research", "analyze", "synthesize"],
    "medium": ["update", "modify", "integrate", "configure", "debug", "test"],
    "low": ["read", "check", "view", "list", "display", "print", "get", "fetch"]
}

CRITICALITY_KEYWORDS = {
    "high": ["security", "authentication", "payment", "data loss", "crash", "production", "critical"],
    "medium": ["user experience", "performance", "feature", "important"],
    "low": ["cosmetic", "minor", "optional", "nice to have"]
}

UNCERTAINTY_KEYWORDS = {
    "high": ["explore", "investigate", "research", "unclear", "ambiguous", "unknown"],
    "medium": ["figure out", "decide", "choose", "determine"],
    "low": ["implement", "following spec", "as described", "specified"]
}


def _heuristic_score_dimension(task_description: str, dimension: str) -> float:
    """
    Score a single dimension using keyword heuristics.

    Args:
        task_description: Task description (lowercased)
        dimension: Dimension name

    Returns:
        Score in [0.0, 1.0]
    """
    desc_lower = task_description.lower()

    if dimension == "complexity":
        for keyword in COMPLEXITY_KEYWORDS["high"]:
            if keyword in desc_lower:
                return 0.7
        for keyword in COMPLEXITY_KEYWORDS["medium"]:
            if keyword in desc_lower:
                return 0.5
        for keyword in COMPLEXITY_KEYWORDS["low"]:
            if keyword in desc_lower:
                return 0.2
        return 0.5  # Default medium

    elif dimension == "criticality":
        for keyword in CRITICALITY_KEYWORDS["high"]:
            if keyword in desc_lower:
                return 0.8
        for keyword in CRITICALITY_KEYWORDS["medium"]:
            if keyword in desc_lower:
                return 0.5
        for keyword in CRITICALITY_KEYWORDS["low"]:
            if keyword in desc_lower:
                return 0.2
        return 0.4  # Default low-medium

    elif dimension == "uncertainty":
        for keyword in UNCERTAINTY_KEYWORDS["high"]:
            if keyword in desc_lower:
                return 0.8
        for keyword in UNCERTAINTY_KEYWORDS["medium"]:
            if keyword in desc_lower:
                return 0.5
        for keyword in UNCERTAINTY_KEYWORDS["low"]:
            if keyword in desc_lower:
                return 0.2
        return 0.5  # Default medium

    elif dimension == "verifiability":
        # High verifiability if mentions tests
        if any(kw in desc_lower for kw in ["test", "verify", "check", "validate"]):
            return 0.8
        # Lower for subjective tasks
        if any(kw in desc_lower for kw in ["design", "choose", "decide"]):
            return 0.4
        return 0.6  # Default medium-high

    elif dimension == "reversibility":
        # Low reversibility for destructive operations
        if any(kw in desc_lower for kw in ["delete", "drop", "remove", "deploy", "publish"]):
            return 0.3
        # High for code changes (version control)
        if any(kw in desc_lower for kw in ["code", "implement", "refactor", "update"]):
            return 0.8
        return 0.6  # Default medium-high

    # Default values for remaining dimensions
    elif dimension == "duration":
        # Infer from complexity
        complexity = _heuristic_score_dimension(desc_lower, "complexity")
        return min(1.0, complexity + 0.1)

    elif dimension == "cost":
        # Look for cost indicators
        if any(kw in desc_lower for kw in ["api", "llm", "model", "compute"]):
            return 0.6
        return 0.3  # Default low

    elif dimension == "resource_requirements":
        # Count implied dependencies
        if any(kw in desc_lower for kw in ["integrate", "connect", "api", "database", "service"]):
            return 0.6
        return 0.4  # Default low-medium

    elif dimension == "constraints":
        # Look for constraint keywords
        if any(kw in desc_lower for kw in ["must", "required", "constraint", "limitation"]):
            return 0.6
        return 0.3  # Default low

    elif dimension == "contextuality":
        # High if mentions existing system
        if any(kw in desc_lower for kw in ["existing", "current", "integrate with", "based on"]):
            return 0.7
        return 0.4  # Default low-medium

    elif dimension == "subjectivity":
        # High for design/UX tasks
        if any(kw in desc_lower for kw in ["design", "ux", "ui", "choose", "aesthetic"]):
            return 0.7
        # Low for technical tasks
        if any(kw in desc_lower for kw in ["implement", "algorithm", "optimize", "test"]):
            return 0.3
        return 0.5  # Default medium

    return 0.5  # Fallback


def _heuristic_classify(task_description: str, context: Optional[Dict] = None) -> TaskProfile:
    """
    Classify task using keyword heuristics (fallback when LLM unavailable).

    Args:
        task_description: Task description
        context: Optional context dict

    Returns:
        TaskProfile with heuristically scored dimensions
    """
    # Score all dimensions
    scores = {}
    for dim in SCORING_RUBRICS.keys():
        scores[dim] = _heuristic_score_dimension(task_description, dim)

    # Adjust based on context if provided
    if context:
        if context.get("is_critical"):
            scores["criticality"] = max(scores["criticality"], 0.7)
        if context.get("time_sensitive"):
            scores["duration"] = max(scores["duration"], 0.6)
        if context.get("high_stakes"):
            scores["reversibility"] = min(scores["reversibility"], 0.4)

    return TaskProfile(**scores)


# =============================================================================
# LLM-BASED CLASSIFICATION
# =============================================================================

def _build_classification_prompt(task_description: str, context: Optional[Dict] = None) -> Dict[str, str]:
    """
    Build LLM prompt for task classification.

    Args:
        task_description: Task description
        context: Optional context

    Returns:
        Dict with 'system_prompt' and 'user_prompt'
    """
    context_str = ""
    if context:
        context_str = f"\n\n**Additional Context:**\n{json.dumps(context, indent=2)}"

    system_prompt = """You are a task classification expert specializing in delegation taxonomy.

Your job is to analyze tasks and score them across 11 dimensions on a 0.0-1.0 scale.

CRITICAL: You must return ONLY a valid JSON object with exactly these 11 keys:
- complexity
- criticality
- uncertainty
- duration
- cost
- resource_requirements
- constraints
- verifiability
- reversibility
- contextuality
- subjectivity

Each value must be a float between 0.0 and 1.0.

Example output format:
{
  "complexity": 0.7,
  "criticality": 0.5,
  "uncertainty": 0.3,
  "duration": 0.6,
  "cost": 0.4,
  "resource_requirements": 0.5,
  "constraints": 0.4,
  "verifiability": 0.8,
  "reversibility": 0.7,
  "contextuality": 0.6,
  "subjectivity": 0.3
}

Do NOT include any explanation, markdown formatting, or additional text. Output ONLY the JSON object."""

    # Build detailed rubric for user prompt
    rubric_str = ""
    for dim, rubric in SCORING_RUBRICS.items():
        rubric_str += f"\n**{dim.replace('_', ' ').title()}** — {rubric['description']}\n"
        for score, desc in sorted(rubric['scale'].items()):
            rubric_str += f"  - {score}: {desc}\n"

    user_prompt = f"""## Task to Classify

{task_description}{context_str}

## Scoring Rubrics
{rubric_str}

## Your Task

Score the task across all 11 dimensions. Consider:
1. **Task complexity** — Is this trivial, or does it require deep expertise?
2. **Impact** — What happens if this fails?
3. **Clarity** — Are requirements clear or ambiguous?
4. **Time** — How long will this realistically take?
5. **Resources** — What external dependencies are needed?
6. **Constraints** — How constrained is the solution space?
7. **Verification** — How easy is it to verify correctness?
8. **Reversibility** — Can this be undone if it goes wrong?
9. **Context** — How much context does this need?
10. **Subjectivity** — Is this objective or subjective?

Output ONLY the JSON object with scores. No other text."""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


async def _llm_classify(task_description: str, context: Optional[Dict] = None) -> TaskProfile:
    """
    Classify task using LLM.

    Args:
        task_description: Task description
        context: Optional context

    Returns:
        TaskProfile with LLM-scored dimensions

    Raises:
        RuntimeError: If LLM classification fails
    """
    if not HAS_LLM_CLIENT:
        raise RuntimeError("LLM client not available")

    # Build prompt
    prompts = _build_classification_prompt(task_description, context)

    # Call LLM
    llm_client = get_llm_client()
    response = await llm_client.complete(
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
        model="haiku",  # Use fast model for classification
        max_tokens=512,
        temperature=0.3  # Low temperature for consistent scoring
    )

    # Parse JSON response
    content = response.content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        # Extract JSON from code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
        else:
            # Try removing just the backticks
            content = content.replace("```json", "").replace("```", "").strip()

    try:
        scores = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nResponse: {content}")

    # Validate all dimensions present
    missing = set(SCORING_RUBRICS.keys()) - set(scores.keys())
    if missing:
        raise RuntimeError(f"LLM response missing dimensions: {missing}")

    # Clamp scores to [0.0, 1.0]
    for dim in scores:
        scores[dim] = max(0.0, min(1.0, float(scores[dim])))

    return TaskProfile(**scores)


# =============================================================================
# PUBLIC API
# =============================================================================

def classify_task(
    description: str,
    context: Optional[Dict] = None,
    use_llm: bool = True,
    timeout: float = 3.0
) -> TaskProfile:
    """
    Classify a task across 11 dimensions for intelligent delegation.

    This is the main entry point for task taxonomy. It will:
    1. Attempt LLM-based classification (if use_llm=True and available)
    2. Fall back to heuristic classification if LLM unavailable or times out

    Args:
        description: Task description to classify
        context: Optional context dict with additional info
        use_llm: Whether to use LLM (True) or force heuristics (False)
        timeout: Timeout for LLM call in seconds (default: 3.0)

    Returns:
        TaskProfile with all dimensions scored and computed properties

    Examples:
        >>> profile = classify_task("Implement authentication system")
        >>> print(f"Complexity: {profile.complexity}")
        >>> print(f"Risk: {profile.risk_score}")
        >>>
        >>> # With context
        >>> profile = classify_task(
        ...     "Add new API endpoint",
        ...     context={"is_critical": True, "time_sensitive": True}
        ... )
        >>>
        >>> # Force heuristics
        >>> profile = classify_task("Debug login bug", use_llm=False)
    """
    if not description or not description.strip():
        raise ValueError("Task description cannot be empty")

    # Try LLM first if requested and available
    if use_llm and HAS_LLM_CLIENT:
        try:
            # Run with timeout
            profile = asyncio.wait_for(
                _llm_classify(description, context),
                timeout=timeout
            )
            return asyncio.run(profile)
        except (asyncio.TimeoutError, RuntimeError, Exception):
            # Fall through to heuristic
            pass

    # Fallback to heuristic
    return _heuristic_classify(description, context)


# =============================================================================
# TASK PROFILE EXTENSIONS
# =============================================================================

def compute_delegation_overhead(profile: TaskProfile) -> float:
    """
    Compute delegation overhead score.

    Tasks with complexity < 0.2 should bypass delegation entirely.

    Args:
        profile: TaskProfile to analyze

    Returns:
        Delegation overhead score [0.0, 1.0]
        - < 0.2: Don't delegate (overhead exceeds value)
        - 0.2-0.5: Consider delegating for parallel execution
        - > 0.5: Strong delegation candidate
    """
    # Simple heuristic: overhead inversely proportional to complexity
    # But also consider duration and cost
    if profile.complexity < 0.2:
        return 0.1  # Too simple to delegate

    # Overhead is lower for complex, long-duration, high-cost tasks
    overhead = 1.0 - (
        profile.complexity * 0.5 +
        profile.duration * 0.3 +
        profile.cost * 0.2
    )

    return max(0.0, min(1.0, overhead))


def compute_risk_score(profile: TaskProfile) -> float:
    """
    Compute risk score for the task.

    Risk = criticality * (1 - reversibility) * uncertainty

    High risk tasks need more careful handling and verification.

    Args:
        profile: TaskProfile to analyze

    Returns:
        Risk score [0.0, 1.0]
    """
    # Weighted combination emphasizing criticality
    # Use multiplication for risk (all factors must be high for high risk)
    risk = (
        profile.criticality * 0.5 +
        (1.0 - profile.reversibility) * 0.3 +
        profile.uncertainty * 0.2
    )

    return max(0.0, min(1.0, risk))


# Extend TaskProfile with computed properties
TaskProfile.delegation_overhead = property(lambda self: compute_delegation_overhead(self))
TaskProfile.risk_score = property(lambda self: compute_risk_score(self))
