"""
Tests for 4Ds Framework — Anthropic's Responsible AI Gates

Tests all four gates:
1. Delegation: Should this be delegated to AI?
2. Description: Is the task well-described?
3. Discernment: Is the output acceptable?
4. Diligence: Are ethical constraints satisfied?
"""

import pytest
from delegation.four_ds import (
    FourDsGate,
    delegation_gate,
    description_gate,
    discernment_gate,
    diligence_gate,
)
from delegation.models import TaskProfile


class TestDelegationGate:
    """Test Gate 1: Delegation decision"""

    def test_blocks_high_risk_tasks(self):
        """High subjectivity + high criticality + low reversibility should be blocked"""
        profile = TaskProfile(
            subjectivity=0.8,
            criticality=0.9,
            reversibility=0.1
        )
        approved, reason = delegation_gate("Make strategic hiring decision", profile)
        assert not approved
        assert "blocked" in reason.lower()
        assert "subjectivity" in reason.lower()

    def test_blocks_critical_unverifiable(self):
        """High criticality + low verifiability should be blocked"""
        profile = TaskProfile(
            criticality=0.9,
            verifiability=0.2,
            subjectivity=0.5,
            reversibility=0.5
        )
        approved, reason = delegation_gate("Deploy to production", profile)
        assert not approved
        assert "verifiability" in reason.lower()

    def test_approves_safe_tasks(self):
        """Tasks within risk bounds should be approved"""
        profile = TaskProfile(
            subjectivity=0.3,
            criticality=0.5,
            reversibility=0.8,
            verifiability=0.7
        )
        approved, reason = delegation_gate("Format code files", profile)
        assert approved
        assert "approved" in reason.lower()

    def test_edge_case_exact_thresholds(self):
        """Test exact threshold boundaries"""
        # At threshold for criticality+reversibility (>= 0.8 and < 0.3, should fail)
        profile = TaskProfile(
            subjectivity=0.6,
            criticality=0.8,  # >= 0.8 triggers block
            reversibility=0.29  # < 0.3 triggers block
        )
        approved, reason = delegation_gate("Task at threshold", profile)
        assert not approved  # At >= threshold, so fails

        # Just below threshold (should pass)
        profile_pass = TaskProfile(
            subjectivity=0.6,
            criticality=0.79,  # < 0.8, won't trigger
            reversibility=0.29
        )
        approved_pass, reason_pass = delegation_gate("Task below threshold", profile_pass)
        assert approved_pass


class TestDescriptionGate:
    """Test Gate 2: Description quality"""

    def test_rejects_vague_descriptions(self):
        """Vague descriptions ('do the thing') should score low"""
        score, suggestions = description_gate("do the thing", use_llm=False)
        assert score < 0.6
        assert "vague" in suggestions.lower() or "improve" in suggestions.lower()

    def test_accepts_specific_descriptions(self):
        """Specific descriptions with criteria should score high"""
        desc = "Implement authentication system with JWT tokens. Must support login, logout, and token refresh. Verify tokens expire after 1 hour."
        score, suggestions = description_gate(desc, use_llm=False)
        assert score >= 0.6
        # Should have some positive feedback
        assert "clear" in suggestions.lower() or "good" in suggestions.lower() or "complete" in suggestions.lower()

    def test_penalizes_short_descriptions(self):
        """Very short descriptions should score low"""
        score, suggestions = description_gate("fix bug", use_llm=False)
        assert score < 0.5
        assert "context" in suggestions.lower() or "details" in suggestions.lower()

    def test_rewards_measurable_criteria(self):
        """Descriptions with measurable criteria should score higher"""
        # Make descriptions longer so completeness doesn't penalize both equally
        desc_without = "Build a fast API endpoint for user authentication that handles login and logout"
        desc_with = "Build API endpoint that responds in < 100ms with > 99% uptime and handles at least 1000 requests per second"

        score_without, _ = description_gate(desc_without, use_llm=False)
        score_with, _ = description_gate(desc_with, use_llm=False)

        assert score_with > score_without

    def test_heuristic_vs_llm_modes(self):
        """Both modes should return valid scores"""
        desc = "Implement user registration with email validation"

        # Heuristic mode
        score_heuristic, sugg_heuristic = description_gate(desc, use_llm=False)
        assert 0.0 <= score_heuristic <= 1.0
        assert isinstance(sugg_heuristic, str)

        # LLM mode (may fall back to heuristic if unavailable)
        score_llm, sugg_llm = description_gate(desc, use_llm=True)
        assert 0.0 <= score_llm <= 1.0
        assert isinstance(sugg_llm, str)


class TestDiscernmentGate:
    """Test Gate 3: Output quality assessment"""

    def test_flags_low_quality_output(self):
        """Output with score < 0.7 should be flagged for review"""
        output = "error failed"
        expected = "Successfully implemented authentication with JWT tokens and refresh mechanism"
        profile = TaskProfile()

        score, issues = discernment_gate(output, expected, profile)
        assert score < 0.7
        assert any("flagged" in issue.lower() for issue in issues)

    def test_accepts_high_quality_output(self):
        """Output matching expected format should score well"""
        output = "Successfully implemented authentication with JWT tokens, login and logout endpoints, token refresh mechanism working"
        expected = "Implement authentication with JWT tokens, login, logout, refresh"
        profile = TaskProfile()

        score, issues = discernment_gate(output, expected, profile)
        assert score >= 0.6  # Should have decent overlap

    def test_detects_error_indicators(self):
        """Output with error terms should be flagged"""
        output = "Exception occurred: undefined variable in authentication module"
        expected = "Authentication module implementation"
        profile = TaskProfile()

        score, issues = discernment_gate(output, expected, profile)
        assert any("error" in issue.lower() for issue in issues)

    def test_length_consistency_check(self):
        """Significantly different lengths should raise issues"""
        expected = "A reasonably long description of expected output format and content"

        # Too short
        output_short = "done"
        score_short, issues_short = discernment_gate(output_short, expected, TaskProfile())
        assert any("shorter" in issue.lower() for issue in issues_short)

        # Too long (3x+)
        output_long = expected * 5
        score_long, issues_long = discernment_gate(output_long, expected, TaskProfile())
        assert any("longer" in issue.lower() for issue in issues_long)

    def test_completeness_scoring(self):
        """Keyword overlap should affect completeness score"""
        expected = "authentication login logout tokens refresh user"

        # Good overlap
        output_good = "implemented authentication with login logout and token refresh for users"
        score_good, _ = discernment_gate(output_good, expected, TaskProfile())

        # Poor overlap
        output_poor = "did some work on the project"
        score_poor, _ = discernment_gate(output_poor, expected, TaskProfile())

        assert score_good > score_poor


class TestDiligenceGate:
    """Test Gate 4: Ethical and safety constraints"""

    def test_warns_sensitive_data(self):
        """Tasks with sensitive data should trigger warnings"""
        profile = TaskProfile(reversibility=0.8)
        safe, warnings = diligence_gate("Process user passwords and API keys", profile)

        assert safe  # Not blocked, but warned
        assert any("sensitive" in w.lower() for w in warnings)

    def test_warns_destructive_operations(self):
        """Destructive operations with low reversibility should warn"""
        profile = TaskProfile(reversibility=0.3)
        safe, warnings = diligence_gate("Delete all user records from database", profile)

        assert any("destructive" in w.lower() or "reversibility" in w.lower() for w in warnings)

    def test_blocks_unsafe_combination(self):
        """Sensitive + destructive + irreversible should be blocked"""
        profile = TaskProfile(reversibility=0.1)
        safe, warnings = diligence_gate("Delete all API keys and credentials permanently", profile)

        assert not safe
        assert any("BLOCKED" in w.upper() for w in warnings)

    def test_warns_production_deployment(self):
        """Production deployment with low verifiability should warn"""
        profile = TaskProfile(verifiability=0.4)
        safe, warnings = diligence_gate("Deploy new release to production", profile)

        assert safe  # Warned but not blocked
        assert any("production" in w.lower() or "verifiability" in w.lower() for w in warnings)

    def test_approves_safe_tasks(self):
        """Safe tasks should have no warnings"""
        profile = TaskProfile(
            reversibility=0.9,
            verifiability=0.8,
            criticality=0.3
        )
        safe, warnings = diligence_gate("Format markdown documentation", profile)

        assert safe
        assert any("no" in w.lower() and "concern" in w.lower() for w in warnings)

    def test_critical_irreversible_warning(self):
        """High criticality + low reversibility should warn"""
        profile = TaskProfile(
            criticality=0.9,
            reversibility=0.2,
            verifiability=0.8
        )
        safe, warnings = diligence_gate("Update production database schema", profile)

        assert any("criticality" in w.lower() or "oversight" in w.lower() for w in warnings)


class TestFourDsGateClass:
    """Test FourDsGate class methods"""

    def test_gate_initialization(self):
        """Gate should initialize with default or custom db path"""
        gate_default = FourDsGate()
        assert gate_default.db_path.endswith("delegation_events.db")

        gate_custom = FourDsGate(db_path="/tmp/test.db")
        assert gate_custom.db_path == "/tmp/test.db"

    def test_all_gates_return_correct_types(self):
        """All gates should return correct tuple types"""
        gate = FourDsGate()
        profile = TaskProfile()

        # Gate 1: (bool, str)
        approved, reason = gate.delegation_gate("task", profile)
        assert isinstance(approved, bool)
        assert isinstance(reason, str)

        # Gate 2: (float, str)
        score, suggestions = gate.description_gate("task description", use_llm=False)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(suggestions, str)

        # Gate 3: (float, List[str])
        quality, issues = gate.discernment_gate("output", "expected", profile)
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
        assert isinstance(issues, list)
        assert all(isinstance(issue, str) for issue in issues)

        # Gate 4: (bool, List[str])
        safe, warnings = gate.diligence_gate("task", profile)
        assert isinstance(safe, bool)
        assert isinstance(warnings, list)
        assert all(isinstance(w, str) for w in warnings)


class TestPublicAPI:
    """Test public API functions"""

    def test_all_public_functions_importable(self):
        """All four gate functions should be importable"""
        from delegation import (
            delegation_gate,
            description_gate,
            discernment_gate,
            diligence_gate,
            FourDsGate,
        )
        # If import works, test passes

    def test_functions_work_standalone(self):
        """Public API functions should work without creating gate instance"""
        profile = TaskProfile(complexity=0.5)

        # These should all work
        approved, reason = delegation_gate("simple task", profile)
        score, suggestions = description_gate("implement feature X", use_llm=False)
        quality, issues = discernment_gate("output", "expected output", profile)
        safe, warnings = diligence_gate("safe task", profile)

        # Basic assertions
        assert isinstance(approved, bool)
        assert isinstance(score, float)
        assert isinstance(quality, float)
        assert isinstance(safe, bool)


class TestIntegrationScenarios:
    """Test realistic end-to-end scenarios"""

    def test_research_task_flow(self):
        """Research task should pass all gates"""
        profile = TaskProfile(
            complexity=0.6,
            criticality=0.4,
            subjectivity=0.5,
            reversibility=0.9,
            verifiability=0.7
        )

        # Gate 1: Should be approved for delegation
        approved, _ = delegation_gate("Research best practices for API design", profile)
        assert approved

        # Gate 2: Description quality
        desc = "Research and document best practices for REST API design. Include examples of versioning, error handling, and authentication patterns. Output should be markdown with at least 5 references."
        score, _ = description_gate(desc, use_llm=False)
        assert score >= 0.6

        # Gate 4: Should be safe
        safe, warnings = diligence_gate("Research API design patterns", profile)
        assert safe

    def test_production_deployment_flow(self):
        """Production deployment should trigger appropriate warnings"""
        profile = TaskProfile(
            complexity=0.8,
            criticality=0.9,
            subjectivity=0.3,
            reversibility=0.4,
            verifiability=0.6
        )

        # Gate 1: May be approved but with concerns
        approved, reason = delegation_gate("Deploy v2.0 to production", profile)
        # Could go either way depending on exact thresholds

        # Gate 4: Should warn about production + verifiability
        safe, warnings = diligence_gate("Deploy release v2.0 to production servers", profile)
        assert any("production" in w.lower() or "verifiability" in w.lower() for w in warnings)

    def test_data_deletion_flow(self):
        """Sensitive data deletion should be heavily scrutinized"""
        profile = TaskProfile(
            complexity=0.5,
            criticality=0.9,
            reversibility=0.1,
            subjectivity=0.4,
            verifiability=0.6
        )

        # Gate 1: Should be blocked (critical + irreversible)
        approved, reason = delegation_gate("Delete user data permanently", profile)
        assert not approved

        # Gate 4: Should be blocked (sensitive + destructive + irreversible)
        safe, warnings = diligence_gate("Delete all user API keys permanently", profile)
        assert not safe
        assert any("BLOCKED" in w.upper() for w in warnings)
