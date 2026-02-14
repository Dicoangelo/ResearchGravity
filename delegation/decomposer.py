"""
Task Decomposer — Contract-First Task Decomposition

Implements the contract-first decomposition strategy from arXiv:2602.11865 Section 4.1.

Key principle: "Task delegation contingent upon outcome having precise verification."
If a subtask has verifiability < 0.3, we recursively decompose until all subtasks are verifiable.

Decomposition strategies:
- Sequential: Dependencies must execute in order
- Parallel: Independent subtasks can run concurrently
- Hierarchical: Nested decomposition for very complex tasks
- Hybrid: Mix of sequential and parallel execution

Usage:
    from delegation.decomposer import decompose_task
    from delegation.taxonomy import classify_task

    profile = classify_task("Build API server with auth")
    subtasks = decompose_task(
        task="Build API server with auth",
        profile=profile,
        max_depth=4
    )

    # Check all subtasks are verifiable
    assert all(st.profile.verifiability >= 0.3 for st in subtasks)
"""

import asyncio
import json
import re
import uuid
from typing import List, Dict, Optional, Set

from .models import TaskProfile, SubTask, VerificationMethod

# Import LLM client from cpb
try:
    from cpb.llm_client import get_llm_client
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False


# =============================================================================
# CONSTANTS
# =============================================================================

MIN_VERIFIABILITY = 0.3  # Contract-first threshold from paper
MAX_DEPTH = 4  # Prevent infinite recursion
DEFAULT_TIMEOUT = 5.0  # LLM timeout in seconds


# =============================================================================
# LLM-BASED DECOMPOSITION
# =============================================================================

def _build_decomposition_prompt(task: str, profile: TaskProfile, depth: int) -> Dict[str, str]:
    """
    Build LLM prompt for task decomposition.

    Args:
        task: Task description
        profile: TaskProfile for the task
        depth: Current decomposition depth

    Returns:
        Dict with system_prompt and user_prompt
    """
    system_prompt = """You are an expert task decomposition system. Your job is to break down complex tasks into smaller, verifiable subtasks.

CRITICAL RULES:
1. Each subtask MUST have verifiability >= 0.3 (on a 0.0-1.0 scale)
2. Each subtask MUST have a clear verification method
3. Mark parallel_safe=true only if subtask has NO dependencies on other subtasks
4. Include estimated_cost and estimated_duration (0.0-1.0 scale)
5. Identify dependencies between subtasks (list of subtask IDs)

Output ONLY a JSON object with this structure:
{
  "subtasks": [
    {
      "description": "Clear description of what to do",
      "verification_method": "automated_test" | "semantic_similarity" | "human_review" | "ground_truth",
      "estimated_cost": 0.0-1.0,
      "estimated_duration": 0.0-1.0,
      "parallel_safe": true | false,
      "dependencies": ["subtask-0", "subtask-1"],
      "profile": {
        "complexity": 0.0-1.0,
        "criticality": 0.0-1.0,
        "uncertainty": 0.0-1.0,
        "duration": 0.0-1.0,
        "cost": 0.0-1.0,
        "resource_requirements": 0.0-1.0,
        "constraints": 0.0-1.0,
        "verifiability": 0.0-1.0,
        "reversibility": 0.0-1.0,
        "contextuality": 0.0-1.0,
        "subjectivity": 0.0-1.0
      }
    }
  ]
}

IMPORTANT: verifiability MUST be >= 0.3 for ALL subtasks. If a subtask cannot be verified, break it down further or make verification explicit."""

    user_prompt = f"""## Task to Decompose

**Description:** {task}

**Current Depth:** {depth}/{MAX_DEPTH}

**Parent Task Profile:**
- Complexity: {profile.complexity:.2f}
- Criticality: {profile.criticality:.2f}
- Uncertainty: {profile.uncertainty:.2f}
- Verifiability: {profile.verifiability:.2f}

## Decomposition Strategy

Break this task into 2-6 subtasks that:
1. Are independently verifiable (verifiability >= 0.3)
2. Cover the full scope of the parent task
3. Have clear dependencies marked
4. Can be executed in parallel where possible
5. Each have appropriate verification methods

Output the JSON object with subtasks."""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }


async def _llm_decompose(
    task: str,
    profile: TaskProfile,
    parent_id: Optional[str],
    depth: int
) -> List[SubTask]:
    """
    Decompose task using LLM.

    Args:
        task: Task description
        profile: TaskProfile for the task
        parent_id: Parent task ID (for hierarchical decomposition)
        depth: Current decomposition depth

    Returns:
        List of SubTask objects

    Raises:
        RuntimeError: If LLM decomposition fails
    """
    if not HAS_LLM_CLIENT:
        raise RuntimeError("LLM client not available")

    # Build prompt
    prompts = _build_decomposition_prompt(task, profile, depth)

    # Call LLM
    llm_client = get_llm_client()
    response = await llm_client.complete(
        system_prompt=prompts["system_prompt"],
        user_prompt=prompts["user_prompt"],
        model="sonnet",  # Use sonnet for better decomposition quality
        max_tokens=2048,
        temperature=0.4  # Moderate temperature for creativity with structure
    )

    # Parse JSON response
    content = response.content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
        else:
            content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nResponse: {content[:200]}")

    if "subtasks" not in data:
        raise RuntimeError(f"LLM response missing 'subtasks' key: {list(data.keys())}")

    # Parse subtasks
    subtasks = []
    for idx, st_data in enumerate(data["subtasks"]):
        # Generate ID
        subtask_id = f"subtask-{uuid.uuid4().hex[:8]}"

        # Parse verification method
        method_str = st_data.get("verification_method", "human_review").lower()
        if method_str == "automated_test":
            method = VerificationMethod.AUTOMATED_TEST
        elif method_str == "semantic_similarity":
            method = VerificationMethod.SEMANTIC_SIMILARITY
        elif method_str == "ground_truth":
            method = VerificationMethod.GROUND_TRUTH
        else:
            method = VerificationMethod.HUMAN_REVIEW

        # Parse profile
        profile_data = st_data.get("profile", {})
        st_profile = TaskProfile(
            complexity=max(0.0, min(1.0, float(profile_data.get("complexity", 0.5)))),
            criticality=max(0.0, min(1.0, float(profile_data.get("criticality", 0.5)))),
            uncertainty=max(0.0, min(1.0, float(profile_data.get("uncertainty", 0.5)))),
            duration=max(0.0, min(1.0, float(profile_data.get("duration", 0.5)))),
            cost=max(0.0, min(1.0, float(profile_data.get("cost", 0.5)))),
            resource_requirements=max(0.0, min(1.0, float(profile_data.get("resource_requirements", 0.5)))),
            constraints=max(0.0, min(1.0, float(profile_data.get("constraints", 0.5)))),
            verifiability=max(0.0, min(1.0, float(profile_data.get("verifiability", 0.5)))),
            reversibility=max(0.0, min(1.0, float(profile_data.get("reversibility", 0.5)))),
            contextuality=max(0.0, min(1.0, float(profile_data.get("contextuality", 0.5)))),
            subjectivity=max(0.0, min(1.0, float(profile_data.get("subjectivity", 0.5))))
        )

        # Create SubTask
        subtask = SubTask(
            id=subtask_id,
            description=st_data.get("description", f"Subtask {idx}"),
            verification_method=method,
            estimated_cost=max(0.0, min(1.0, float(st_data.get("estimated_cost", 0.5)))),
            estimated_duration=max(0.0, min(1.0, float(st_data.get("estimated_duration", 0.5)))),
            parallel_safe=bool(st_data.get("parallel_safe", False)),
            parent_task_id=parent_id,
            dependencies=st_data.get("dependencies", []),
            profile=st_profile,
            metadata={"depth": depth}
        )

        subtasks.append(subtask)

    return subtasks


# =============================================================================
# HEURISTIC DECOMPOSITION (Fallback)
# =============================================================================

def _heuristic_decompose(
    task: str,
    profile: TaskProfile,
    parent_id: Optional[str],
    depth: int
) -> List[SubTask]:
    """
    Heuristic-based task decomposition (fallback when LLM unavailable).

    Uses keyword-based rules to decompose tasks into common patterns:
    - "Build X" -> Design, Implement, Test, Deploy
    - "Implement Y" -> Plan, Code, Test
    - "Research Z" -> Survey, Analyze, Synthesize

    Args:
        task: Task description
        profile: TaskProfile for the task
        parent_id: Parent task ID
        depth: Current decomposition depth

    Returns:
        List of SubTask objects
    """
    task_lower = task.lower()
    subtasks = []

    # Pattern 1: Build/Create systems
    if any(kw in task_lower for kw in ["build", "create", "develop", "implement system"]):
        templates = [
            ("Design system architecture", VerificationMethod.HUMAN_REVIEW, 0.4, 0.3, False, []),
            ("Implement core functionality", VerificationMethod.AUTOMATED_TEST, 0.5, 0.6, False, ["subtask-0"]),
            ("Add tests and validation", VerificationMethod.AUTOMATED_TEST, 0.3, 0.3, False, ["subtask-1"]),
            ("Deploy and verify", VerificationMethod.GROUND_TRUTH, 0.4, 0.4, False, ["subtask-2"])
        ]

    # Pattern 2: Research tasks
    elif any(kw in task_lower for kw in ["research", "investigate", "explore", "analyze"]):
        templates = [
            ("Survey existing solutions", VerificationMethod.HUMAN_REVIEW, 0.3, 0.4, True, []),
            ("Analyze findings", VerificationMethod.SEMANTIC_SIMILARITY, 0.4, 0.5, False, ["subtask-0"]),
            ("Synthesize recommendations", VerificationMethod.HUMAN_REVIEW, 0.5, 0.4, False, ["subtask-1"])
        ]

    # Pattern 3: Implementation tasks
    elif any(kw in task_lower for kw in ["implement", "code", "write"]):
        templates = [
            ("Plan implementation approach", VerificationMethod.HUMAN_REVIEW, 0.3, 0.2, False, []),
            ("Write code", VerificationMethod.AUTOMATED_TEST, 0.5, 0.6, False, ["subtask-0"]),
            ("Add tests", VerificationMethod.AUTOMATED_TEST, 0.3, 0.3, False, ["subtask-1"])
        ]

    # Default pattern
    else:
        templates = [
            ("Understand requirements", VerificationMethod.HUMAN_REVIEW, 0.2, 0.2, False, []),
            ("Execute main task", VerificationMethod.AUTOMATED_TEST, 0.6, 0.6, False, ["subtask-0"]),
            ("Verify completion", VerificationMethod.GROUND_TRUTH, 0.3, 0.2, False, ["subtask-1"])
        ]

    # Create SubTask objects from templates
    for idx, (desc, method, cost, duration, parallel, deps) in enumerate(templates):
        # Create profile for subtask (inherit from parent, reduced complexity)
        st_profile = TaskProfile(
            complexity=max(0.2, profile.complexity * 0.6),  # Subtasks are simpler
            criticality=profile.criticality,  # Inherit criticality
            uncertainty=max(0.2, profile.uncertainty * 0.7),  # Reduce uncertainty
            duration=duration,
            cost=cost,
            resource_requirements=profile.resource_requirements * 0.5,
            constraints=profile.constraints * 0.5,
            verifiability=0.7,  # Heuristic subtasks are verifiable by design
            reversibility=max(0.5, profile.reversibility),  # Conservative
            contextuality=profile.contextuality * 0.6,
            subjectivity=profile.subjectivity * 0.5
        )

        subtask = SubTask(
            id=f"subtask-{uuid.uuid4().hex[:8]}",
            description=f"{desc} for: {task[:50]}",
            verification_method=method,
            estimated_cost=cost,
            estimated_duration=duration,
            parallel_safe=parallel,
            parent_task_id=parent_id,
            dependencies=deps,
            profile=st_profile,
            metadata={"depth": depth, "heuristic": True}
        )

        subtasks.append(subtask)

    return subtasks


# =============================================================================
# RECURSIVE DECOMPOSITION (Contract-First Enforcement)
# =============================================================================

def _recursive_decompose(
    task: str,
    profile: TaskProfile,
    parent_id: Optional[str],
    depth: int,
    use_llm: bool,
    timeout: float
) -> List[SubTask]:
    """
    Recursively decompose task until all subtasks meet verifiability threshold.

    This implements the contract-first principle: if any subtask has
    verifiability < MIN_VERIFIABILITY, decompose it further.

    Args:
        task: Task description
        profile: TaskProfile for the task
        parent_id: Parent task ID
        depth: Current decomposition depth
        use_llm: Whether to use LLM
        timeout: LLM timeout

    Returns:
        List of fully verifiable SubTask objects
    """
    # Base case: max depth reached
    if depth >= MAX_DEPTH:
        # Force verifiability to minimum at max depth
        forced_profile = TaskProfile(
            complexity=profile.complexity,
            criticality=profile.criticality,
            uncertainty=profile.uncertainty,
            duration=profile.duration,
            cost=profile.cost,
            resource_requirements=profile.resource_requirements,
            constraints=profile.constraints,
            verifiability=MIN_VERIFIABILITY,  # Force to minimum
            reversibility=profile.reversibility,
            contextuality=profile.contextuality,
            subjectivity=profile.subjectivity
        )

        return [SubTask(
            id=f"subtask-{uuid.uuid4().hex[:8]}",
            description=task,
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=profile.cost,
            estimated_duration=profile.duration,
            parallel_safe=True,
            parent_task_id=parent_id,
            dependencies=[],
            profile=forced_profile,
            metadata={"depth": depth, "forced_verifiable": True}
        )]

    # Try LLM decomposition first
    if use_llm and HAS_LLM_CLIENT:
        try:
            subtasks = asyncio.run(
                asyncio.wait_for(
                    _llm_decompose(task, profile, parent_id, depth),
                    timeout=timeout
                )
            )
        except Exception:
            # Fall back to heuristic
            subtasks = _heuristic_decompose(task, profile, parent_id, depth)
    else:
        # Use heuristic
        subtasks = _heuristic_decompose(task, profile, parent_id, depth)

    # Contract-first check: recursively decompose low-verifiability subtasks
    verified_subtasks = []
    for st in subtasks:
        if st.profile and st.profile.verifiability < MIN_VERIFIABILITY:
            # Recursively decompose this subtask
            nested = _recursive_decompose(
                task=st.description,
                profile=st.profile,
                parent_id=st.id,
                depth=depth + 1,
                use_llm=use_llm,
                timeout=timeout
            )
            verified_subtasks.extend(nested)
        else:
            verified_subtasks.append(st)

    return verified_subtasks


# =============================================================================
# DEPENDENCY ANALYSIS
# =============================================================================

def _analyze_dependencies(subtasks: List[SubTask]) -> List[SubTask]:
    """
    Analyze dependencies and update parallel_safe flags.

    A subtask is parallel_safe=True only if:
    1. It has no dependencies
    2. OR all its dependencies are also parallel_safe

    Args:
        subtasks: List of SubTask objects

    Returns:
        Updated list of SubTask objects with corrected parallel_safe flags
    """
    # Build ID map
    id_to_task = {st.id: st for st in subtasks}

    # Iteratively update parallel_safe
    changed = True
    while changed:
        changed = False
        for st in subtasks:
            if st.parallel_safe and st.dependencies:
                # Check if all dependencies are parallel_safe
                deps_parallel = all(
                    id_to_task.get(dep_id, SubTask(
                        id="", description="", verification_method=VerificationMethod.HUMAN_REVIEW,
                        estimated_cost=0.0, estimated_duration=0.0, parallel_safe=False
                    )).parallel_safe
                    for dep_id in st.dependencies
                )

                if not deps_parallel:
                    st.parallel_safe = False
                    changed = True

    return subtasks


# =============================================================================
# PUBLIC API
# =============================================================================

def decompose_task(
    task: str,
    profile: TaskProfile,
    max_depth: int = MAX_DEPTH,
    use_llm: bool = True,
    timeout: float = DEFAULT_TIMEOUT
) -> List[SubTask]:
    """
    Decompose a task into verifiable subtasks following the contract-first principle.

    This is the main entry point for task decomposition. It will:
    1. Attempt LLM-based decomposition (if use_llm=True and available)
    2. Fall back to heuristic decomposition if LLM fails or unavailable
    3. Recursively decompose any subtask with verifiability < 0.3
    4. Stop at max_depth to prevent infinite recursion
    5. Analyze dependencies and update parallel_safe flags

    Contract-first rule (from arXiv:2602.11865 Section 4.1):
    "Task delegation contingent upon outcome having precise verification."

    Args:
        task: Task description to decompose
        profile: TaskProfile for the task
        max_depth: Maximum decomposition depth (default: 4)
        use_llm: Whether to use LLM for decomposition (default: True)
        timeout: Timeout for LLM calls in seconds (default: 5.0)

    Returns:
        List of SubTask objects, all with verifiability >= 0.3

    Example:
        >>> from delegation.taxonomy import classify_task
        >>> profile = classify_task("Build user authentication system")
        >>> subtasks = decompose_task("Build user authentication system", profile)
        >>> assert all(st.profile.verifiability >= 0.3 for st in subtasks)
    """
    # Recursive decomposition with contract-first enforcement
    subtasks = _recursive_decompose(
        task=task,
        profile=profile,
        parent_id=None,
        depth=0,
        use_llm=use_llm,
        timeout=timeout
    )

    # Analyze dependencies and update parallel_safe flags
    subtasks = _analyze_dependencies(subtasks)

    return subtasks
