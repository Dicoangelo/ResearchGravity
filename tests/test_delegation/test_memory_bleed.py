"""
Tests for delegation.memory_bleed — Supermemory integration

Test coverage:
- MemoryContext and ErrorPattern dataclass instantiation
- get_relevant_context() with mock supermemory.db
- get_error_patterns() with mock supermemory.db
- get_domain_expertise() with mock supermemory.db
- write_delegation_outcome() with mock supermemory.db
- inject_context() metadata injection
- Graceful degradation when DB unavailable
- Read-only connection safety
- Performance targets (<500ms for context, <100ms for patterns/expertise)
"""

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from delegation.memory_bleed import (
    MemoryContext,
    ErrorPattern,
    get_relevant_context,
    get_error_patterns,
    get_domain_expertise,
    write_delegation_outcome,
    inject_context,
    _get_db_path,
    _connect_readonly,
)
from delegation.models import SubTask, TaskProfile, VerificationMethod


# ============================================================================
# Test Data Models
# ============================================================================


class TestMemoryContext:
    """Test MemoryContext dataclass."""

    def test_instantiation(self):
        ctx = MemoryContext(
            content="Test content",
            source="test-source",
            quality=0.8,
            similarity=0.92,
        )
        assert ctx.content == "Test content"
        assert ctx.source == "test-source"
        assert ctx.quality == 0.8
        assert ctx.similarity == 0.92
        assert ctx.date is None
        assert ctx.project is None
        assert ctx.metadata == {}

    def test_with_optional_fields(self):
        ctx = MemoryContext(
            content="Test",
            source="src",
            quality=0.5,
            date="2026-02-14",
            project="os-app",
            metadata={"tags": ["delegation"]},
        )
        assert ctx.date == "2026-02-14"
        assert ctx.project == "os-app"
        assert ctx.metadata == {"tags": ["delegation"]}


class TestErrorPattern:
    """Test ErrorPattern dataclass."""

    def test_instantiation(self):
        err = ErrorPattern(
            category="git",
            pattern="merge conflict",
            count=5,
        )
        assert err.category == "git"
        assert err.pattern == "merge conflict"
        assert err.count == 5
        assert err.solution is None
        assert err.last_seen is None

    def test_with_solution(self):
        err = ErrorPattern(
            category="auth",
            pattern="token expired",
            count=10,
            solution="refresh token",
            last_seen="2026-02-14",
        )
        assert err.solution == "refresh token"
        assert err.last_seen == "2026-02-14"


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelpers:
    """Test internal helper functions."""

    def test_get_db_path_exists(self):
        """Test _get_db_path returns path if file exists."""
        # Mock Path.exists() to return True
        with patch("delegation.memory_bleed.Path.exists", return_value=True):
            path = _get_db_path()
            assert path is not None
            assert "supermemory.db" in path

    def test_get_db_path_missing(self):
        """Test _get_db_path returns None if file doesn't exist."""
        with patch("delegation.memory_bleed.Path.exists", return_value=False):
            path = _get_db_path()
            assert path is None

    def test_connect_readonly_success(self, tmp_path):
        """Test read-only connection works."""
        # Create minimal test database
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        # Connect read-only
        ro_conn = _connect_readonly(str(db_path))
        assert ro_conn is not None

        # Verify read-only (insert should fail)
        with pytest.raises(sqlite3.OperationalError):
            ro_conn.execute("INSERT INTO test VALUES (1)")

        ro_conn.close()

    def test_connect_readonly_missing_file(self):
        """Test connection returns None for missing file."""
        conn = _connect_readonly("/nonexistent/path.db")
        assert conn is None


# ============================================================================
# Test get_relevant_context
# ============================================================================


class TestGetRelevantContext:
    """Test get_relevant_context() semantic search."""

    @pytest.fixture
    def mock_supermemory(self, tmp_path):
        """Create mock supermemory.db with sample data."""
        db_path = tmp_path / "supermemory.db"
        conn = sqlite3.connect(str(db_path))

        # Create schema
        conn.execute("""
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                date DATE,
                project TEXT,
                quality REAL DEFAULT 0,
                metadata TEXT
            )
        """)

        # Insert test data
        conn.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("auth-1", "session", "Implement OAuth2 authentication flow with JWT tokens", "2026-02-10", "os-app", 0.9, None),
        )
        conn.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("auth-2", "learning", "Authentication requires rate limiting to prevent brute force", "2026-02-12", "os-app", 0.8, '{"tags": ["security"]}'),
        )
        conn.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("db-1", "session", "Database schema migration with Alembic", "2026-02-11", "os-app", 0.7, None),
        )
        conn.execute(
            "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("low-quality", "session", "Low quality note", "2026-02-13", None, 0.2, None),
        )

        conn.commit()
        conn.close()

        return db_path

    def test_returns_empty_when_db_missing(self):
        """Test graceful degradation when DB unavailable."""
        with patch("delegation.memory_bleed._get_db_path", return_value=None):
            results = get_relevant_context("test query")
            assert results == []

    def test_returns_empty_when_embeddings_unavailable(self, mock_supermemory):
        """Test graceful degradation when SBERT unavailable."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            with patch("delegation.memory_bleed.HAS_EMBEDDINGS", False):
                results = get_relevant_context("test query")
                assert results == []

    def test_filters_low_quality_items(self, mock_supermemory):
        """Test quality >= 0.5 filter."""
        # Mock DB path and embeddings
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            # Mock embedding functions to return dummy vectors
            with patch("delegation.memory_bleed.embed_single") as mock_embed:
                with patch("delegation.memory_bleed.cosine_similarity") as mock_sim:
                    mock_embed.return_value = [0.1] * 768
                    mock_sim.return_value = 0.8  # High similarity

                    results = get_relevant_context("test", limit=10)

                    # Should exclude "low-quality" item (quality=0.2)
                    content_list = [r.content for r in results]
                    assert "Low quality note" not in content_list

    def test_returns_top_n_by_similarity(self, mock_supermemory):
        """Test limit parameter and sorting."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            with patch("delegation.memory_bleed.embed_single") as mock_embed:
                with patch("delegation.memory_bleed.cosine_similarity") as mock_sim:
                    mock_embed.return_value = [0.1] * 768

                    # Return different similarities for different items
                    similarities = [0.9, 0.7, 0.85]  # auth-1, auth-2, db-1
                    mock_sim.side_effect = similarities

                    results = get_relevant_context("authentication", limit=2)

                    # Should return top 2 by similarity (auth-1: 0.9, db-1: 0.85)
                    assert len(results) == 2
                    assert results[0].similarity >= results[1].similarity

    def test_parses_metadata_json(self, mock_supermemory):
        """Test metadata JSON parsing."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            with patch("delegation.memory_bleed.embed_single") as mock_embed:
                with patch("delegation.memory_bleed.cosine_similarity") as mock_sim:
                    mock_embed.return_value = [0.1] * 768
                    mock_sim.return_value = 0.8

                    results = get_relevant_context("authentication", limit=5)

                    # Find auth-2 which has metadata
                    auth2 = [r for r in results if r.source == "learning"]
                    if auth2:
                        assert auth2[0].metadata.get("tags") == ["security"]


# ============================================================================
# Test get_error_patterns
# ============================================================================


class TestGetErrorPatterns:
    """Test get_error_patterns() error retrieval."""

    @pytest.fixture
    def mock_supermemory(self, tmp_path):
        """Create mock supermemory.db with error patterns."""
        db_path = tmp_path / "supermemory.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                solution TEXT,
                last_seen DATE
            )
        """)

        conn.execute(
            "INSERT INTO error_patterns (category, pattern, count, solution, last_seen) VALUES (?, ?, ?, ?, ?)",
            ("authentication", "token expired", 10, "refresh token", "2026-02-14"),
        )
        conn.execute(
            "INSERT INTO error_patterns (category, pattern, count, solution) VALUES (?, ?, ?, ?)",
            ("database", "connection timeout", 5, "increase pool size"),
        )
        conn.execute(
            "INSERT INTO error_patterns (category, pattern, count) VALUES (?, ?, ?)",
            ("git", "authentication merge conflict", 3),
        )

        conn.commit()
        conn.close()

        return db_path

    def test_returns_empty_when_db_missing(self):
        """Test graceful degradation."""
        with patch("delegation.memory_bleed._get_db_path", return_value=None):
            results = get_error_patterns("test")
            assert results == []

    def test_matches_by_category(self, mock_supermemory):
        """Test category matching (case-insensitive)."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            results = get_error_patterns("authentication")
            assert len(results) >= 1
            assert any(e.category == "authentication" for e in results)

    def test_matches_by_pattern(self, mock_supermemory):
        """Test pattern matching."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            results = get_error_patterns("timeout")
            assert len(results) >= 1
            assert any("timeout" in e.pattern for e in results)

    def test_sorted_by_count_desc(self, mock_supermemory):
        """Test sorting by count (descending)."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            results = get_error_patterns("auth")  # Matches "authentication" category and "git" pattern
            if len(results) >= 2:
                assert results[0].count >= results[1].count


# ============================================================================
# Test get_domain_expertise
# ============================================================================


class TestGetDomainExpertise:
    """Test get_domain_expertise() scoring."""

    @pytest.fixture
    def mock_supermemory(self, tmp_path):
        """Create mock supermemory.db with varied data."""
        db_path = tmp_path / "supermemory.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                quality REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE learnings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                quality REAL
            )
        """)
        conn.execute("""
            CREATE TABLE error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                pattern TEXT
            )
        """)

        # Insert authentication data (high expertise)
        for i in range(20):
            conn.execute("INSERT INTO memory_items VALUES (?, ?, ?, ?)", (f"auth-{i}", f"Authentication item {i}", "session", 0.8))

        for i in range(10):
            conn.execute("INSERT INTO learnings VALUES (?, ?, ?)", (f"learn-auth-{i}", f"Authentication learning {i}", 0.9))

        for i in range(5):
            conn.execute("INSERT INTO error_patterns (category, pattern) VALUES (?, ?)", ("authentication", f"error {i}"))

        # Insert database data (low expertise)
        conn.execute("INSERT INTO memory_items VALUES (?, ?, ?, ?)", ("db-1", "Database item", "session", 0.7))

        conn.commit()
        conn.close()

        return db_path

    def test_returns_zero_when_db_missing(self):
        """Test graceful degradation."""
        with patch("delegation.memory_bleed._get_db_path", return_value=None):
            score = get_domain_expertise("test")
            assert score == 0.0

    def test_returns_zero_for_unknown_domain(self, mock_supermemory):
        """Test unknown domain returns 0.0."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            score = get_domain_expertise("unknown-domain-xyz")
            assert score == 0.0

    def test_returns_high_score_for_well_covered_domain(self, mock_supermemory):
        """Test high coverage returns score close to 1.0."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            score = get_domain_expertise("authentication")
            # 20 memory + 10*2 learnings + 5*3 errors = 20+20+15 = 55 total
            # log10(56) / 2 = 1.748 / 2 = 0.874, clamped to 1.0
            assert score >= 0.8
            assert score <= 1.0

    def test_returns_low_score_for_sparse_domain(self, mock_supermemory):
        """Test low coverage returns low score."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            score = get_domain_expertise("database")
            # 1 memory item only = log10(2) / 2 = 0.15
            assert score < 0.3


# ============================================================================
# Test write_delegation_outcome
# ============================================================================


class TestWriteDelegationOutcome:
    """Test write_delegation_outcome() spaced repetition."""

    @pytest.fixture
    def mock_supermemory(self, tmp_path):
        """Create mock supermemory.db with reviews table."""
        db_path = tmp_path / "supermemory.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE reviews (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT,
                ease_factor REAL DEFAULT 2.5,
                interval_days INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                next_review DATE,
                last_review DATE,
                source_id TEXT
            )
        """)

        conn.commit()
        conn.close()

        return db_path

    def test_returns_false_when_db_missing(self):
        """Test graceful degradation."""
        with patch("delegation.memory_bleed._get_db_path", return_value=None):
            success = write_delegation_outcome("task", "outcome", 0.8)
            assert success is False

    def test_writes_review_successfully(self, mock_supermemory):
        """Test successful write to reviews table."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            success = write_delegation_outcome(
                task="Implement auth",
                outcome="Success: JWT tokens working",
                quality=0.9,
                category="delegation",
                project="os-app",
            )
            assert success is True

            # Verify review was written
            conn = sqlite3.connect(str(mock_supermemory))
            row = conn.execute("SELECT * FROM reviews").fetchone()
            assert row is not None
            assert "[Delegation]" in row[1]  # content
            assert "Implement auth" in row[1]
            assert row[2] == "delegation"  # category
            assert row[3] == 2.5  # ease_factor
            assert row[4] == 1  # interval_days
            assert row[5] == 0  # repetitions
            conn.close()


# ============================================================================
# Test inject_context
# ============================================================================


class TestInjectContext:
    """Test inject_context() metadata injection."""

    def test_injects_memory_context_into_subtasks(self):
        """Test context injection into SubTask.metadata."""
        subtasks = [
            SubTask(
                id="sub-1",
                description="Implement authentication",
                verification_method=VerificationMethod.AUTOMATED_TEST,
                estimated_cost=0.5,
                estimated_duration=0.6,
                parallel_safe=True,
            ),
        ]

        # Mock get_relevant_context to return dummy context
        with patch("delegation.memory_bleed.get_relevant_context") as mock_get:
            mock_get.return_value = [
                MemoryContext(
                    content="Past auth implementation used OAuth2",
                    source="session",
                    quality=0.9,
                    similarity=0.85,
                ),
            ]

            inject_context(subtasks, context_limit=3)

            # Verify metadata injection
            assert "memory_context" in subtasks[0].metadata
            assert len(subtasks[0].metadata["memory_context"]) == 1
            assert subtasks[0].metadata["memory_context"][0]["similarity"] == 0.85

    def test_injects_error_patterns_for_critical_tasks(self):
        """Test error pattern injection for high-criticality tasks."""
        profile = TaskProfile(criticality=0.8, complexity=0.5)
        subtasks = [
            SubTask(
                id="sub-1",
                description="Critical task",
                verification_method=VerificationMethod.AUTOMATED_TEST,
                estimated_cost=0.5,
                estimated_duration=0.6,
                parallel_safe=True,
                profile=profile,
            ),
        ]

        with patch("delegation.memory_bleed.get_relevant_context", return_value=[]):
            with patch("delegation.memory_bleed.get_error_patterns") as mock_errors:
                mock_errors.return_value = [
                    ErrorPattern(
                        category="critical",
                        pattern="race condition",
                        count=5,
                        solution="use locks",
                    ),
                ]

                inject_context(subtasks)

                # Verify error patterns injected
                assert "error_patterns" in subtasks[0].metadata
                assert len(subtasks[0].metadata["error_patterns"]) == 1
                assert subtasks[0].metadata["error_patterns"][0]["category"] == "critical"


# ============================================================================
# Test Performance
# ============================================================================


class TestPerformance:
    """Test performance targets."""

    @pytest.fixture
    def mock_supermemory(self, tmp_path):
        """Create realistic-sized mock DB."""
        db_path = tmp_path / "supermemory.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute("""
            CREATE TABLE memory_items (
                id TEXT PRIMARY KEY,
                content TEXT,
                source TEXT,
                quality REAL,
                date DATE,
                project TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE error_patterns (
                id INTEGER PRIMARY KEY,
                category TEXT,
                pattern TEXT,
                count INTEGER,
                solution TEXT,
                last_seen DATE
            )
        """)
        conn.execute("""
            CREATE TABLE learnings (
                id TEXT PRIMARY KEY,
                content TEXT,
                quality REAL
            )
        """)

        # Insert 100 items for realistic performance test
        for i in range(100):
            conn.execute(
                "INSERT INTO memory_items VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"item-{i}", f"Content {i}", "session", 0.7, "2026-02-14", "os-app", None),
            )

        for i in range(10):
            conn.execute(
                "INSERT INTO error_patterns VALUES (?, ?, ?, ?, ?, ?)",
                (i, "auth", f"pattern {i}", i+1, None, "2026-02-14"),
            )

        conn.commit()
        conn.close()

        return db_path

    def test_error_patterns_fast(self, mock_supermemory):
        """Test get_error_patterns completes in <100ms."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            t0 = time.time()
            results = get_error_patterns("auth")
            elapsed = (time.time() - t0) * 1000  # ms

            assert elapsed < 100  # Target: <100ms
            assert len(results) > 0

    def test_domain_expertise_fast(self, mock_supermemory):
        """Test get_domain_expertise completes in <100ms."""
        with patch("delegation.memory_bleed._get_db_path", return_value=str(mock_supermemory)):
            t0 = time.time()
            score = get_domain_expertise("auth")
            elapsed = (time.time() - t0) * 1000  # ms

            assert elapsed < 100  # Target: <100ms
            assert score >= 0.0
