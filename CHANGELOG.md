# Changelog

All notable changes to ResearchGravity are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2026-03-17]
### Added
- Hardening — decay engine, adaptive ReACT refinement

### Fixed
- Patch rollup to >=4.59.0 (CVE path traversal)
- Add asyncpg to test dependencies for CI

### Changed
- Upgrade GitHub Actions for Node.js 24 compatibility

## [2026-03-15]
### Added
- MiroFish integration — temporal graph, ReACT synthesis, ontology, personas, oracle

## [2026-03-13]
### Fixed
- Phase 1-3: fix test infrastructure, ruff auto-fix and formatting, test suite (356 passed, 0 failed)

## [2026-02-25]
### Added
- HuggingFace daily papers importer, refined pipeline improvements

## [2026-02-19]
### Added
- Ecosystem hardening — delegation engine, MCP expansion, dashboard components, CI/CD

### Fixed
- Make ruff lint step non-blocking in CI

## [2026-02-16]
### Added
- Interactive proof deck with real pipeline demo
- Architecture diagrams: dark + light mode with auto-switch

### Changed
- Reorganize repo structure: 122 root items to 35 (71% reduction)

### Fixed
- CI paths and README paths after repo reorganization
- UCW metrics and Qdrant sync

## [2026-02-14]
### Added
- Intelligent Delegation Module v0.1.0: multi-agent task decomposition, trust, and verification
- Evolution engine: EMA-based learning from delegation outcomes
- X/Twitter trust bridge: sync author trust into delegation ledger
- Wire live execution, history, auto-delegate monitors, thread publishing

## [2026-02-12]
### Added
- Deep Cognitive Substrate + Chrome Extension (Phase 8 complete)
- UCW Phase 8.1-8.4: emergence listener, session-level coherence, insight backfill, breakthrough detection
- FSRS insight notifier, daemon KG extraction, review-due CLI
- Webhook capture server and cognitive profile

### Fixed
- MCP server name collision and PostgreSQL session key conflicts
- NotebookLM MCP server transport API mismatch
- NotebookLM cookie persistence and context pack null-safety

### Changed
- Drop duplicate HNSW index (595MB), sync schema with production
- Wire arc detection into daemon as periodic background task

## [2026-02-09]
### Changed
- Update Metaventions AI URLs to metaventionsai.com

## [2026-02-08]
### Added
- Phase 2 coherence engine — realtime mode, knowledge graph, embedding upgrade

## [2026-02-07]
### Added
- UCW Raw MCP server with 14 tools and cognitive capture
- Embedding pipeline and PostgreSQL migration
- Coherence engine — real-time detection daemon
- Real-time auto-embedding + pgvector-native similarity search
- Multi-platform capture adapters: ChatGPT, Cursor, Grok/X, Claude Code CLI, CCC
- Capture infrastructure: dedup engine, capture manager, CLI
- CLI transcript importer, coherence analysis
- Metaventions whitepaper v1.0 — Sovereign Substrate for Cognitive Equity

### Fixed
- Batch embed infinite loop + cross-platform coherence pipeline
- Lazy DB init + handshake guard for CLI MCP connection
- Synchronous stdout writes in transport layer
- Handle notifications/initialized method from Claude Desktop

## [2026-02-01]
### Fixed
- Sanitize FTS queries to handle hyphens correctly

## [2026-01-31]
### Added
- v6.1: Security, performance, and reliability upgrades
- v6.0: Interactive Research Platform
- Sentence-transformers offline fallback for embeddings
- Upgrade to Cohere embed-v4 with Matryoshka dimensions

### Fixed
- Rename 'validate' field to avoid Pydantic BaseModel shadow
- Robust health_check handles missing backends gracefully

## [2026-01-28]
### Added
- `--transcripts` flag for YouTube channel scraper

## [2026-01-26]
### Added
- Meta-Learning Engine with predictive session intelligence
- Qdrant vector storage and semantic search (100% complete)

## [2026-01-21]
### Added
- Complete MCP Integration (v3.7)
- CPB v2.5: comprehensive hardening (caching, retry, fallback, cost tracking)
- CPB Precision Mode v2.4: Pioneer Mode, Trust Context, Deep Research

### Changed
- Migrate to google-genai SDK and install aiohttp

## [2026-01-20]
### Added
- CPB Precision Mode v2.0-v2.3: tiered search, MAR consensus, ground truth validation, query enhancement, run logging
- Writer-Critic System + Graph Intelligence (Phases 4 & 6)
- Phase 3a Storage Triad
- Chief of Staff infrastructure — Evidence Layer and API

### Fixed
- Improve arXiv search with keyword extraction and category filtering

## [2026-01-19]
### Changed
- README v5.0: Chief of Staff architecture, Graph Intelligence, Writer-Critic

## [2026-01-18]
### Added
- MCP Integration v4.0: Universal Context Access Protocol
- Context Packs V2: 7-layer context management system
- Autonomous CLI routing system with research integration
- Pattern-aware context loading for co-evolution system

## [2026-01-16]
### Changed
- Identity Unveiling: Rebrand to Dicoangelo
- Add Context Prefetcher for memory injection

## [2026-01-15]
### Added
- YouTube research capability (v3.3.0)

## [2026-01-14]
### Added
- Signal Verification Protocol (SVP) method
- Agent Core v3.2 — Auto-Capture, Lineage Tracking, Project Registry

## [2026-01-12]
### Changed
- Agent Core v3.0 — Metaventions AI
- Professionalize project structure with GitHub community features

## [2026-01-09]
### Added
- Agent-core v2.0 with fixed scripts
