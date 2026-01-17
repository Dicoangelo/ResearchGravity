"""
Universal Cognitive Wallet (UCW)
Portable, appreciating, tradeable AI memory that users own.

Usage:
    from ucw import CognitiveWallet, export_wallet, calculate_value

    wallet = CognitiveWallet.from_agent_core()
    value = calculate_value(wallet)
    export_wallet(wallet, "my-wallet.ucw.json")
"""

from .schema import (
    CognitiveWallet,
    Concept,
    Session,
    Connection,
    URL,
    ValueMetrics,
)
from .export import export_wallet, build_wallet_from_agent_core
from .value import CognitiveAppreciationEngine, calculate_value
from .history import load_history, record_snapshot, get_value_delta

__version__ = "0.1.0"
__all__ = [
    "CognitiveWallet",
    "Concept",
    "Session",
    "Connection",
    "URL",
    "ValueMetrics",
    "export_wallet",
    "build_wallet_from_agent_core",
    "CognitiveAppreciationEngine",
    "calculate_value",
    "load_history",
    "record_snapshot",
    "get_value_delta",
]
