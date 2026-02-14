"""
Test suite for delegation.decomposer module

Tests contract-first task decomposition with both LLM and heuristic paths.
"""

import pytest
from delegation.decomposer import (
    decompose_task,
    _heuristic_decompose,
    _analyze_dependencies,
    MIN_VERIFIABILITY,
)
from delegation.models import TaskProfile, SubTask, VerificationMethod


class TestHeuristicDecomposition:
    """Test heuristic-based decomposition patterns"""

    def test_build_system_pattern(self):
        """Build/create tasks decompose into Design->Implement->Test->Deploy"""
        profile = TaskProfile(complexity=0.7, criticality=0.8)
        subtasks = _heuristic_decompose("Build authentication system", profile, None, 0)

        assert len(subtasks) == 4, "Build pattern should create 4 subtasks"
        assert "Design" in subtasks[0].description
        assert "Implement" in subtasks[1].description
        assert "tests" in subtasks[2].description.lower()
        assert "Deploy" in subtasks[3].description

    def test_research_pattern(self):
        """Research tasks decompose into Survey->Analyze->Synthesize"""
        profile = TaskProfile(complexity=0.6, uncertainty=0.7)
        subtasks = _heuristic_decompose("Research multi-agent coordination", profile, None, 0)

        assert len(subtasks) == 3, "Research pattern should create 3 subtasks"
        assert "Survey" in subtasks[0].description
        assert "Analyze" in subtasks[1].description
        assert "Synthesize" in subtasks[2].description

    def test_implementation_pattern(self):
        """Implementation tasks decompose into Plan->Code->Test"""
        profile = TaskProfile(complexity=0.5, criticality=0.6)
        subtasks = _heuristic_decompose("Implement user login", profile, None, 0)

        assert len(subtasks) == 3, "Implementation pattern should create 3 subtasks"
        assert "Plan" in subtasks[0].description
        assert "code" in subtasks[1].description.lower()
        assert "test" in subtasks[2].description.lower()

    def test_default_pattern(self):
        """Unknown tasks decompose into Understand->Execute->Verify"""
        profile = TaskProfile(complexity=0.5)
        subtasks = _heuristic_decompose("Do something unclear", profile, None, 0)

        assert len(subtasks) == 3, "Default pattern should create 3 subtasks"
        assert "Understand" in subtasks[0].description
        assert "Execute" in subtasks[1].description
        assert "Verify" in subtasks[2].description

    def test_heuristic_verifiability(self):
        """Heuristic subtasks all have verifiability >= 0.3"""
        profile = TaskProfile(complexity=0.8, verifiability=0.2)
        subtasks = _heuristic_decompose("Build complex system", profile, None, 0)

        for st in subtasks:
            assert st.profile.verifiability >= MIN_VERIFIABILITY, \
                f"Subtask '{st.description}' has verifiability {st.profile.verifiability} < {MIN_VERIFIABILITY}"

    def test_complexity_reduction(self):
        """Subtasks have reduced complexity compared to parent"""
        profile = TaskProfile(complexity=0.9)
        subtasks = _heuristic_decompose("Build highly complex system", profile, None, 0)

        for st in subtasks:
            assert st.profile.complexity < profile.complexity, \
                f"Subtask complexity {st.profile.complexity} should be less than parent {profile.complexity}"

    def test_criticality_inheritance(self):
        """Subtasks inherit criticality from parent"""
        profile = TaskProfile(criticality=0.9)
        subtasks = _heuristic_decompose("Build mission-critical system", profile, None, 0)

        for st in subtasks:
            assert st.profile.criticality == profile.criticality, \
                f"Subtask should inherit criticality {profile.criticality}"


class TestDependencyAnalysis:
    """Test dependency analysis and parallel_safe flag updates"""

    def test_no_dependencies_parallel_safe(self):
        """Subtasks with no dependencies stay parallel_safe"""
        st1 = SubTask(
            id="st-1", description="Task 1", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=True, dependencies=[]
        )
        st2 = SubTask(
            id="st-2", description="Task 2", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=True, dependencies=[]
        )

        result = _analyze_dependencies([st1, st2])

        assert result[0].parallel_safe is True
        assert result[1].parallel_safe is True

    def test_dependencies_not_parallel_safe(self):
        """Subtasks with dependencies lose parallel_safe flag"""
        st1 = SubTask(
            id="st-1", description="Task 1", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=False, dependencies=[]
        )
        st2 = SubTask(
            id="st-2", description="Task 2", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=True, dependencies=["st-1"]
        )

        result = _analyze_dependencies([st1, st2])

        assert result[0].parallel_safe is False
        assert result[1].parallel_safe is False, "st-2 depends on non-parallel st-1"

    def test_transitive_dependencies(self):
        """Transitive dependencies propagate parallel_safe=False"""
        st1 = SubTask(
            id="st-1", description="Task 1", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=False, dependencies=[]
        )
        st2 = SubTask(
            id="st-2", description="Task 2", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=True, dependencies=["st-1"]
        )
        st3 = SubTask(
            id="st-3", description="Task 3", verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3, estimated_duration=0.3, parallel_safe=True, dependencies=["st-2"]
        )

        result = _analyze_dependencies([st1, st2, st3])

        assert result[0].parallel_safe is False
        assert result[1].parallel_safe is False
        assert result[2].parallel_safe is False, "st-3 transitively depends on non-parallel st-1"


class TestContractFirstDecomposition:
    """Test contract-first recursive decomposition"""

    def test_decompose_complex_task(self):
        """Complex task decomposes into multiple subtasks"""
        profile = TaskProfile(complexity=0.8, criticality=0.9, verifiability=0.5)
        subtasks = decompose_task("Build user authentication system", profile, use_llm=False)

        assert len(subtasks) >= 2, "Complex task should decompose into multiple subtasks"
        assert all(isinstance(st, SubTask) for st in subtasks)

    def test_all_subtasks_verifiable(self):
        """All subtasks meet minimum verifiability threshold"""
        profile = TaskProfile(complexity=0.9, verifiability=0.2)  # Low verifiability parent
        subtasks = decompose_task("Build complex system with unclear verification", profile, use_llm=False)

        for st in subtasks:
            assert st.profile is not None, f"Subtask '{st.description}' missing profile"
            assert st.profile.verifiability >= MIN_VERIFIABILITY, \
                f"Subtask '{st.description}' has verifiability {st.profile.verifiability} < {MIN_VERIFIABILITY}"

    def test_max_depth_constraint(self):
        """Decomposition stops at max_depth"""
        # Create a task that would decompose infinitely if not for max_depth
        profile = TaskProfile(complexity=0.9, verifiability=0.1)  # Very low verifiability
        subtasks = decompose_task("Extremely complex unverifiable task", profile, max_depth=2, use_llm=False)

        # Should have created subtasks, but stopped at depth 2
        assert len(subtasks) >= 1
        # At max depth, verifiability is forced to MIN_VERIFIABILITY
        for st in subtasks:
            assert st.metadata.get("depth", 0) <= 2, f"Subtask exceeded max_depth: {st.metadata}"

    def test_subtask_ids_unique(self):
        """All subtasks have unique IDs"""
        profile = TaskProfile(complexity=0.7)
        subtasks = decompose_task("Build API server", profile, use_llm=False)

        ids = [st.id for st in subtasks]
        assert len(ids) == len(set(ids)), "Subtask IDs are not unique"

    def test_verification_methods_assigned(self):
        """All subtasks have verification methods"""
        profile = TaskProfile(complexity=0.6)
        subtasks = decompose_task("Implement feature X", profile, use_llm=False)

        for st in subtasks:
            assert st.verification_method in [
                VerificationMethod.AUTOMATED_TEST,
                VerificationMethod.SEMANTIC_SIMILARITY,
                VerificationMethod.HUMAN_REVIEW,
                VerificationMethod.GROUND_TRUTH
            ], f"Invalid verification method: {st.verification_method}"

    def test_estimated_cost_duration_valid(self):
        """All subtasks have valid estimated_cost and estimated_duration"""
        profile = TaskProfile(complexity=0.7)
        subtasks = decompose_task("Build dashboard", profile, use_llm=False)

        for st in subtasks:
            assert 0.0 <= st.estimated_cost <= 1.0, \
                f"Invalid estimated_cost: {st.estimated_cost}"
            assert 0.0 <= st.estimated_duration <= 1.0, \
                f"Invalid estimated_duration: {st.estimated_duration}"


class TestDecompositionAPI:
    """Test public API behavior"""

    def test_decompose_returns_list(self):
        """decompose_task returns a list of SubTask objects"""
        profile = TaskProfile(complexity=0.5)
        result = decompose_task("Simple task", profile, use_llm=False)

        assert isinstance(result, list)
        assert all(isinstance(st, SubTask) for st in result)

    def test_decompose_with_max_depth(self):
        """max_depth parameter limits decomposition depth"""
        profile = TaskProfile(complexity=0.9, verifiability=0.1)
        subtasks = decompose_task("Complex task", profile, max_depth=1, use_llm=False)

        # Should decompose once but stop at depth 1
        assert len(subtasks) >= 1
        for st in subtasks:
            assert st.metadata.get("depth", 0) <= 1

    def test_decompose_use_llm_false(self):
        """use_llm=False forces heuristic decomposition"""
        profile = TaskProfile(complexity=0.6)
        subtasks = decompose_task("Build system", profile, use_llm=False)

        # Heuristic should mark metadata
        assert any(st.metadata.get("heuristic") for st in subtasks), \
            "Heuristic decomposition should mark metadata"


class TestDiverseTaskProfiles:
    """Test decomposition with diverse task profiles"""

    def test_high_complexity_task(self):
        """High complexity tasks decompose appropriately"""
        profile = TaskProfile(
            complexity=0.9,
            criticality=0.8,
            uncertainty=0.7,
            verifiability=0.4
        )
        subtasks = decompose_task("Architect distributed system", profile, use_llm=False)

        assert len(subtasks) >= 3, "High complexity should create multiple subtasks"
        assert all(st.profile.verifiability >= MIN_VERIFIABILITY for st in subtasks)

    def test_low_complexity_task(self):
        """Low complexity tasks still decompose reasonably"""
        profile = TaskProfile(
            complexity=0.2,
            criticality=0.3,
            uncertainty=0.2,
            verifiability=0.8
        )
        subtasks = decompose_task("Update README", profile, use_llm=False)

        assert len(subtasks) >= 1
        for st in subtasks:
            assert st.profile.complexity <= profile.complexity

    def test_critical_task(self):
        """Critical tasks preserve criticality in subtasks"""
        profile = TaskProfile(criticality=0.95, complexity=0.7)
        subtasks = decompose_task("Fix security vulnerability", profile, use_llm=False)

        for st in subtasks:
            assert st.profile.criticality >= 0.8, \
                "Critical parent should have critical subtasks"

    def test_uncertain_task(self):
        """Uncertain tasks decompose to reduce uncertainty"""
        profile = TaskProfile(uncertainty=0.9, complexity=0.6)
        subtasks = decompose_task("Explore unknown solution space", profile, use_llm=False)

        # Subtasks should have lower uncertainty than parent
        for st in subtasks:
            assert st.profile.uncertainty < profile.uncertainty, \
                "Decomposition should reduce uncertainty"


class TestPerformance:
    """Test decomposition performance"""

    def test_heuristic_performance(self):
        """Heuristic decomposition is fast (<100ms)"""
        import time

        profile = TaskProfile(complexity=0.7)
        start = time.time()
        subtasks = decompose_task("Build API", profile, use_llm=False)
        duration = time.time() - start

        assert duration < 0.1, f"Heuristic decomposition took {duration:.3f}s (expected < 0.1s)"
        assert len(subtasks) >= 2

    def test_multiple_decompositions(self):
        """Multiple decompositions in sequence (<1s for 5)"""
        import time

        tasks = [
            "Build authentication",
            "Research framework",
            "Implement feature",
            "Deploy service",
            "Analyze data"
        ]

        start = time.time()
        for task in tasks:
            profile = TaskProfile(complexity=0.6)
            subtasks = decompose_task(task, profile, use_llm=False)
            assert len(subtasks) >= 2
        duration = time.time() - start

        assert duration < 1.0, f"5 decompositions took {duration:.3f}s (expected < 1.0s)"
