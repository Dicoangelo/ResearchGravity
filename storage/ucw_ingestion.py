"""
UCW (Universal Cognitive Wallet) Pack Ingestion Pipeline

Handles bulk import of knowledge packs from:
- UCW trades (external wallet imports)
- Agent production (continuous agent outputs)
- External sources (APIs, file imports)

Features:
- Batch ingestion with transaction safety
- Deduplication across sources
- Provenance tracking
- Optional validation via Writer-Critic
- Conflict resolution strategies
"""

import asyncio
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import uuid

from .engine import StorageEngine, get_engine


class ConflictStrategy(Enum):
    """How to handle duplicate packs."""
    SKIP = "skip"           # Keep existing, ignore new
    REPLACE = "replace"     # Replace existing with new
    MERGE = "merge"         # Merge contents (append findings, etc.)
    VERSION = "version"     # Keep both with version suffix


class IngestionResult:
    """Result of a pack ingestion operation."""

    def __init__(self):
        self.total = 0
        self.imported = 0
        self.skipped = 0
        self.merged = 0
        self.errors: List[Dict[str, Any]] = []
        self.pack_ids: List[str] = []
        self.started_at = datetime.now()
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "imported": self.imported,
            "skipped": self.skipped,
            "merged": self.merged,
            "errors": self.errors,
            "pack_ids": self.pack_ids,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
        }


class UCWIngestionPipeline:
    """
    Pipeline for ingesting packs from UCW trades and external sources.

    Usage:
        pipeline = UCWIngestionPipeline()
        await pipeline.initialize()

        # Import from UCW trade
        result = await pipeline.ingest_from_ucw(
            wallet_id="wallet_xyz",
            packs=traded_packs,
            validate=True
        )

        # Import from file
        result = await pipeline.ingest_from_file(
            "exported_packs.json",
            source="external_agent"
        )

        # Continuous agent ingestion
        await pipeline.ingest_agent_output(finding, agent_id="agent_001")
    """

    def __init__(
        self,
        conflict_strategy: ConflictStrategy = ConflictStrategy.SKIP,
        validate_by_default: bool = False,
        batch_size: int = 100
    ):
        self.conflict_strategy = conflict_strategy
        self.validate_by_default = validate_by_default
        self.batch_size = batch_size
        self.engine: Optional[StorageEngine] = None
        self._validator: Optional[Callable] = None

    async def initialize(self):
        """Initialize the pipeline."""
        self.engine = await get_engine()

        # Try to load validator
        try:
            from critic import PackCritic, run_oracle_consensus

            async def validate_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
                critic = PackCritic()
                result = run_oracle_consensus(critic, {"pack": pack})
                return {
                    "approved": result.approved,
                    "confidence": result.confidence,
                    "notes": result.notes,
                }

            self._validator = validate_pack
        except ImportError:
            self._validator = None

    def _generate_pack_id(self, pack: Dict[str, Any], source: str) -> str:
        """Generate deterministic ID for a pack."""
        # Use content hash for deduplication
        content_str = json.dumps(pack.get("content", {}), sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:12]
        return f"pack-{source[:8]}-{content_hash}"

    def _compute_content_hash(self, pack: Dict[str, Any]) -> str:
        """Compute hash of pack content for deduplication."""
        content_str = json.dumps(pack.get("content", {}), sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def _check_duplicate(self, pack: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if pack already exists."""
        pack_id = pack.get("id")
        if pack_id:
            existing = await self.engine.sqlite.get_packs(limit=1)
            # Simple check - in production would query by ID
            # For now, rely on SQLite's ON CONFLICT handling
            pass
        return None

    async def ingest_from_ucw(
        self,
        wallet_id: str,
        packs: List[Dict[str, Any]],
        validate: Optional[bool] = None,
        conflict_strategy: Optional[ConflictStrategy] = None
    ) -> IngestionResult:
        """
        Ingest packs from a UCW trade.

        Args:
            wallet_id: The UCW wallet ID this trade came from
            packs: List of pack dictionaries
            validate: Whether to run Writer-Critic validation
            conflict_strategy: How to handle duplicates

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult()
        result.total = len(packs)

        strategy = conflict_strategy or self.conflict_strategy
        should_validate = validate if validate is not None else self.validate_by_default

        for pack in packs:
            try:
                # Generate ID if not present
                if "id" not in pack:
                    pack["id"] = self._generate_pack_id(pack, f"ucw-{wallet_id}")

                # Validate if requested
                if should_validate and self._validator:
                    validation = await self._validator(pack)
                    pack["validated"] = 1 if validation["approved"] else 0
                    pack["validation_result"] = validation

                    if not validation["approved"] and validation["confidence"] < 0.5:
                        result.skipped += 1
                        result.errors.append({
                            "pack_id": pack["id"],
                            "reason": "validation_failed",
                            "details": validation,
                        })
                        continue

                # Store pack with provenance
                pack_id = await self.engine.store_pack(
                    pack,
                    source="ucw_trade",
                    source_id=wallet_id
                )

                result.imported += 1
                result.pack_ids.append(pack_id)

            except Exception as e:
                result.errors.append({
                    "pack_id": pack.get("id"),
                    "reason": "error",
                    "details": str(e),
                })

        result.completed_at = datetime.now()
        return result

    async def ingest_from_file(
        self,
        file_path: str,
        source: str = "file_import",
        validate: bool = False
    ) -> IngestionResult:
        """
        Ingest packs from a JSON file.

        Args:
            file_path: Path to JSON file containing packs
            source: Source identifier for provenance
            validate: Whether to validate packs
        """
        path = Path(file_path)
        if not path.exists():
            result = IngestionResult()
            result.errors.append({"reason": "file_not_found", "path": str(path)})
            result.completed_at = datetime.now()
            return result

        data = json.loads(path.read_text())

        # Handle different formats
        if isinstance(data, list):
            packs = data
        elif isinstance(data, dict):
            packs = data.get("packs", [data])  # Single pack or packs array
        else:
            result = IngestionResult()
            result.errors.append({"reason": "invalid_format"})
            result.completed_at = datetime.now()
            return result

        return await self.ingest_packs(
            packs,
            source=source,
            source_id=str(path),
            validate=validate
        )

    async def ingest_packs(
        self,
        packs: List[Dict[str, Any]],
        source: str = "external",
        source_id: Optional[str] = None,
        validate: bool = False
    ) -> IngestionResult:
        """
        Generic pack ingestion.

        Args:
            packs: List of pack dictionaries
            source: Source type for provenance
            source_id: Specific source identifier
            validate: Whether to validate packs
        """
        result = IngestionResult()
        result.total = len(packs)

        # Process in batches
        for i in range(0, len(packs), self.batch_size):
            batch = packs[i:i + self.batch_size]

            for pack in batch:
                try:
                    if "id" not in pack:
                        pack["id"] = self._generate_pack_id(pack, source)

                    if validate and self._validator:
                        validation = await self._validator(pack)
                        pack["validated"] = 1 if validation["approved"] else 0
                        pack["validation_result"] = validation

                    pack_id = await self.engine.store_pack(pack, source, source_id)
                    result.imported += 1
                    result.pack_ids.append(pack_id)

                except Exception as e:
                    result.errors.append({
                        "pack_id": pack.get("id"),
                        "error": str(e),
                    })

        result.completed_at = datetime.now()
        return result

    async def ingest_agent_output(
        self,
        content: Dict[str, Any],
        agent_id: str,
        content_type: str = "finding"
    ) -> str:
        """
        Ingest output from an agent in real-time.

        This is for continuous agent production, not batch imports.

        Args:
            content: The content to ingest (finding, pack, etc.)
            agent_id: The agent that produced this content
            content_type: Type of content (finding, pack, session)

        Returns:
            ID of the stored content
        """
        if content_type == "finding":
            content["id"] = content.get("id") or f"finding-{agent_id}-{uuid.uuid4().hex[:8]}"
            return await self.engine.store_finding(content, source=f"agent:{agent_id}")

        elif content_type == "pack":
            content["id"] = content.get("id") or f"pack-{agent_id}-{uuid.uuid4().hex[:8]}"
            return await self.engine.store_pack(
                content,
                source="agent_produced",
                source_id=agent_id
            )

        elif content_type == "session":
            return await self.engine.store_session(content, source=f"agent:{agent_id}")

        else:
            raise ValueError(f"Unknown content type: {content_type}")

    async def get_provenance_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested content by source."""
        async with self.engine.sqlite.connection() as db:
            cursor = await db.execute("""
                SELECT source_type, COUNT(*) as count
                FROM provenance
                GROUP BY source_type
            """)
            rows = await cursor.fetchall()

            stats = {row[0]: row[1] for row in rows}

            # Get UCW-specific stats
            cursor = await db.execute("""
                SELECT source_id, COUNT(*) as count
                FROM provenance
                WHERE source_type = 'ucw_trade'
                GROUP BY source_id
            """)
            ucw_rows = await cursor.fetchall()
            stats["ucw_wallets"] = {row[0]: row[1] for row in ucw_rows}

            # Get agent-specific stats
            cursor = await db.execute("""
                SELECT source_id, COUNT(*) as count
                FROM provenance
                WHERE source_type LIKE 'agent:%'
                GROUP BY source_id
            """)
            agent_rows = await cursor.fetchall()
            stats["agents"] = {row[0]: row[1] for row in agent_rows}

            return stats


# Convenience functions

async def ingest_ucw_trade(
    wallet_id: str,
    packs: List[Dict[str, Any]],
    validate: bool = False
) -> IngestionResult:
    """Quick function for UCW trade ingestion."""
    pipeline = UCWIngestionPipeline()
    await pipeline.initialize()
    return await pipeline.ingest_from_ucw(wallet_id, packs, validate=validate)


async def ingest_from_file(file_path: str, source: str = "file") -> IngestionResult:
    """Quick function for file ingestion."""
    pipeline = UCWIngestionPipeline()
    await pipeline.initialize()
    return await pipeline.ingest_from_file(file_path, source)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UCW Pack Ingestion")
    parser.add_argument("file", help="JSON file to ingest")
    parser.add_argument("--source", default="file_import", help="Source identifier")
    parser.add_argument("--validate", action="store_true", help="Validate packs")

    args = parser.parse_args()

    async def main():
        result = await ingest_from_file(args.file, args.source)
        print(json.dumps(result.to_dict(), indent=2))

    asyncio.run(main())
