"""
Chapter 56: Nyströmformer for Trading

This module provides a complete implementation of Nyströmformer
for financial time series analysis and trading applications.

Nyströmformer approximates self-attention with O(n) complexity
instead of O(n²), making it ideal for processing long sequences
like tick data and order book snapshots.

Main components:
- NystromAttention: The core Nyström-approximated attention mechanism
- NystromformerTrading: Complete model for trading applications
- Data utilities for Bybit and other exchanges
- Backtesting framework for strategy evaluation
"""

from .model import (
    NystromAttention,
    NystromEncoderLayer,
    NystromformerTrading,
)

from .data import (
    load_bybit_data,
    prepare_long_sequence_data,
    calculate_features,
)

from .strategy import (
    NystromBacktester,
    BacktestConfig,
    generate_signals,
)

__version__ = "0.1.0"
__all__ = [
    "NystromAttention",
    "NystromEncoderLayer",
    "NystromformerTrading",
    "load_bybit_data",
    "prepare_long_sequence_data",
    "calculate_features",
    "NystromBacktester",
    "BacktestConfig",
    "generate_signals",
]
