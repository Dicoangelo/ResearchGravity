#!/usr/bin/env python3
"""
Test delegation data models

Verifies that all dataclasses can be instantiated and validates constraints.
"""

import pytest
import time
from delegation.models import (
    TaskProfile,
    SubTask,
    Assignment,
    TrustEntry,
    DelegationEvent,
    VerificationResult,
    VerificationMethod,
)


class TestTaskProfile:
    """Test TaskProfile dataclass"""

    def test_default_instantiation(self):
        """Test creating TaskProfile with defaults"""
        profile = TaskProfile()
        assert profile.complexity == 0.5
        assert profile.criticality == 0.5
        assert profile.uncertainty == 0.5
        assert profile.duration == 0.5
        assert profile.cost == 0.5
        assert profile.resource_requirements == 0.5
        assert profile.constraints == 0.5
        assert profile.verifiability == 0.5
        assert profile.reversibility == 0.5
        assert profile.contextuality == 0.5
        assert profile.subjectivity == 0.5

    def test_custom_values(self):
        """Test creating TaskProfile with custom values"""
        profile = TaskProfile(
            complexity=0.8,
            criticality=0.9,
            uncertainty=0.3,
            duration=0.6,
            cost=0.4,
            resource_requirements=0.7,
            constraints=0.5,
            verifiability=0.8,
            reversibility=0.9,
            contextuality=0.4,
            subjectivity=0.2,
        )
        assert profile.complexity == 0.8
        assert profile.criticality == 0.9
        assert profile.uncertainty == 0.3

    def test_validation_rejects_out_of_range(self):
        """Test that values outside [0.0, 1.0] are rejected"""
        with pytest.raises(ValueError, match="complexity must be in"):
            TaskProfile(complexity=1.5)

        with pytest.raises(ValueError, match="criticality must be in"):
            TaskProfile(criticality=-0.1)

        with pytest.raises(ValueError, match="subjectivity must be in"):
            TaskProfile(subjectivity=2.0)


class TestSubTask:
    """Test SubTask dataclass"""

    def test_basic_instantiation(self):
        """Test creating SubTask with required fields"""
        subtask = SubTask(
            id="task-1",
            description="Write unit tests",
            verification_method=VerificationMethod.AUTOMATED_TEST,
            estimated_cost=0.3,
            estimated_duration=0.4,
            parallel_safe=True,
        )
        assert subtask.id == "task-1"
        assert subtask.description == "Write unit tests"
        assert subtask.verification_method == VerificationMethod.AUTOMATED_TEST
        assert subtask.estimated_cost == 0.3
        assert subtask.estimated_duration == 0.4
        assert subtask.parallel_safe is True
        assert subtask.parent_task_id is None
        assert subtask.dependencies == []

    def test_with_dependencies(self):
        """Test SubTask with dependencies"""
        subtask = SubTask(
            id="task-2",
            description="Deploy to production",
            verification_method=VerificationMethod.HUMAN_REVIEW,
            estimated_cost=0.6,
            estimated_duration=0.5,
            parallel_safe=False,
            dependencies=["task-1"],
        )
        assert subtask.dependencies == ["task-1"]
        assert subtask.parallel_safe is False

    def test_validation_rejects_invalid_cost(self):
        """Test that invalid cost is rejected"""
        with pytest.raises(ValueError, match="estimated_cost must be in"):
            SubTask(
                id="task-x",
                description="Test",
                verification_method=VerificationMethod.AUTOMATED_TEST,
                estimated_cost=1.5,
                estimated_duration=0.5,
                parallel_safe=True,
            )

    def test_validation_rejects_invalid_duration(self):
        """Test that invalid duration is rejected"""
        with pytest.raises(ValueError, match="estimated_duration must be in"):
            SubTask(
                id="task-x",
                description="Test",
                verification_method=VerificationMethod.AUTOMATED_TEST,
                estimated_cost=0.5,
                estimated_duration=-0.1,
                parallel_safe=True,
            )


class TestAssignment:
    """Test Assignment dataclass"""

    def test_basic_instantiation(self):
        """Test creating Assignment"""
        now = time.time()
        assignment = Assignment(
            subtask_id="task-1",
            agent_id="agent-1",
            trust_score=0.75,
            capability_match=0.85,
            timestamp=now,
            assignment_reasoning="Highest trust agent for this task type",
        )
        assert assignment.subtask_id == "task-1"
        assert assignment.agent_id == "agent-1"
        assert assignment.trust_score == 0.75
        assert assignment.capability_match == 0.85
        assert assignment.timestamp == now
        assert "Highest trust" in assignment.assignment_reasoning

    def test_validation_rejects_invalid_trust_score(self):
        """Test that invalid trust_score is rejected"""
        with pytest.raises(ValueError, match="trust_score must be in"):
            Assignment(
                subtask_id="task-1",
                agent_id="agent-1",
                trust_score=1.5,
                capability_match=0.8,
                timestamp=time.time(),
            )

    def test_validation_rejects_invalid_capability_match(self):
        """Test that invalid capability_match is rejected"""
        with pytest.raises(ValueError, match="capability_match must be in"):
            Assignment(
                subtask_id="task-1",
                agent_id="agent-1",
                trust_score=0.8,
                capability_match=-0.2,
                timestamp=time.time(),
            )


class TestTrustEntry:
    """Test TrustEntry dataclass"""

    def test_basic_instantiation(self):
        """Test creating TrustEntry"""
        now = time.time()
        entry = TrustEntry(
            agent_id="agent-1",
            task_id="task-1",
            timestamp=now,
            success=True,
            quality_score=0.9,
            trust_delta=0.05,
            updated_trust_score=0.80,
            notes="Excellent performance",
        )
        assert entry.agent_id == "agent-1"
        assert entry.task_id == "task-1"
        assert entry.success is True
        assert entry.quality_score == 0.9
        assert entry.trust_delta == 0.05
        assert entry.updated_trust_score == 0.80

    def test_validation_rejects_invalid_quality_score(self):
        """Test that invalid quality_score is rejected"""
        with pytest.raises(ValueError, match="quality_score must be in"):
            TrustEntry(
                agent_id="agent-1",
                task_id="task-1",
                timestamp=time.time(),
                success=True,
                quality_score=1.2,
                trust_delta=0.05,
                updated_trust_score=0.8,
            )

    def test_validation_rejects_invalid_trust_delta(self):
        """Test that invalid trust_delta is rejected"""
        with pytest.raises(ValueError, match="trust_delta must be in"):
            TrustEntry(
                agent_id="agent-1",
                task_id="task-1",
                timestamp=time.time(),
                success=True,
                quality_score=0.9,
                trust_delta=1.5,
                updated_trust_score=0.8,
            )


class TestDelegationEvent:
    """Test DelegationEvent dataclass"""

    def test_basic_instantiation(self):
        """Test creating DelegationEvent"""
        now = time.time()
        event = DelegationEvent(
            event_id="evt-1",
            delegation_id="del-1",
            timestamp=now,
            event_type="assigned",
            agent_id="agent-1",
            task_id="task-1",
            status="in_progress",
            details={"note": "Task started"},
        )
        assert event.event_id == "evt-1"
        assert event.delegation_id == "del-1"
        assert event.event_type == "assigned"
        assert event.agent_id == "agent-1"
        assert event.task_id == "task-1"
        assert event.status == "in_progress"
        assert event.details == {"note": "Task started"}


class TestVerificationResult:
    """Test VerificationResult dataclass"""

    def test_basic_instantiation(self):
        """Test creating VerificationResult"""
        now = time.time()
        result = VerificationResult(
            subtask_id="task-1",
            timestamp=now,
            method=VerificationMethod.AUTOMATED_TEST,
            passed=True,
            quality_score=0.95,
            feedback="All tests passed",
            evidence={"test_count": 42, "passed": 42, "failed": 0},
        )
        assert result.subtask_id == "task-1"
        assert result.method == VerificationMethod.AUTOMATED_TEST
        assert result.passed is True
        assert result.quality_score == 0.95
        assert "All tests passed" in result.feedback

    def test_validation_rejects_invalid_quality_score(self):
        """Test that invalid quality_score is rejected"""
        with pytest.raises(ValueError, match="quality_score must be in"):
            VerificationResult(
                subtask_id="task-1",
                timestamp=time.time(),
                method=VerificationMethod.GROUND_TRUTH,
                passed=True,
                quality_score=1.1,
            )


class TestVerificationMethod:
    """Test VerificationMethod enum"""

    def test_enum_values(self):
        """Test that all enum values exist"""
        assert VerificationMethod.AUTOMATED_TEST == "automated_test"
        assert VerificationMethod.SEMANTIC_SIMILARITY == "semantic_similarity"
        assert VerificationMethod.HUMAN_REVIEW == "human_review"
        assert VerificationMethod.GROUND_TRUTH == "ground_truth"
