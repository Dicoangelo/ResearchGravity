"""
Writer-Critic Validation System

Implements dual-agent validation for high-stakes outputs:
- Archive synthesis (completeness)
- Evidence citations (accuracy)
- Context packs (relevance)

Usage:
    from critic import ArchiveCritic, EvidenceCritic, PackCritic

    critic = ArchiveCritic()
    result = await critic.validate(session_id)

    if result.confidence < 0.7:
        print(f"Issues found: {result.issues}")
"""

from .base import CriticBase, ValidationResult
from .archive_critic import ArchiveCritic
from .evidence_critic import EvidenceCritic
from .pack_critic import PackCritic

__all__ = [
    'CriticBase',
    'ValidationResult',
    'ArchiveCritic',
    'EvidenceCritic',
    'PackCritic',
]
