# Chapter 56: Nyströmformer for Trading

This chapter explores **Nyströmformer**, an efficient transformer architecture that uses the Nyström method to approximate self-attention with linear O(n) complexity instead of the standard quadratic O(n²). This makes it ideal for processing long financial time series such as tick data, order book snapshots, and extended historical sequences.

<p align="center">
<img src="https://i.imgur.com/8KqPL4v.png" width="70%">
</p>

## Contents

1. [Introduction to Nyströmformer](#introduction-to-nyströmformer)
    * [The Quadratic Attention Problem](#the-quadratic-attention-problem)
    * [Nyström Method Overview](#nyström-method-overview)
    * [Key Advantages for Trading](#key-advantages-for-trading)
2. [Mathematical Foundation](#mathematical-foundation)
    * [Standard Self-Attention](#standard-self-attention)
    * [Nyström Approximation](#nyström-approximation)
    * [Landmark Selection](#landmark-selection)
    * [Iterative Pseudoinverse](#iterative-pseudoinverse)
3. [Nyströmformer Architecture](#nyströmformer-architecture)
    * [Segment-Means for Landmarks](#segment-means-for-landmarks)
    * [Three-Matrix Decomposition](#three-matrix-decomposition)
    * [Complexity Analysis](#complexity-analysis)
4. [Trading Applications](#trading-applications)
    * [Long Sequence Price Prediction](#long-sequence-price-prediction)
    * [Order Book Analysis](#order-book-analysis)
    * [Multi-Timeframe Fusion](#multi-timeframe-fusion)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Nyströmformer Model](#02-nyströmformer-model)
    * [03: Training Pipeline](#03-training-pipeline)
    * [04: Backtesting Strategy](#04-backtesting-strategy)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Nyströmformer

### The Quadratic Attention Problem

Standard transformer self-attention computes attention scores between all pairs of tokens, resulting in O(n²) time and memory complexity. For a sequence of length n:

```
Standard Attention Cost:
- Sequence length 512:    262,144 operations
- Sequence length 1024:   1,048,576 operations
- Sequence length 4096:   16,777,216 operations
- Sequence length 8192:   67,108,864 operations

The cost grows quadratically!
```

In trading, we often need to process:
- **Tick data**: Thousands of price updates per minute
- **Order book snapshots**: Deep history of bid/ask levels
- **Multi-asset correlations**: Long lookback windows across many instruments
- **High-frequency features**: Microsecond-level data streams

Standard transformers become prohibitively expensive for these use cases.

### Nyström Method Overview

The **Nyström method** is a classical technique from numerical linear algebra for approximating large matrices using a subset of their columns and rows. Originally developed for kernel methods, it works by:

1. **Selecting landmarks**: Choose m representative points from n total points (m << n)
2. **Computing interactions**: Calculate full attention only for landmark-to-all relationships
3. **Reconstructing the matrix**: Approximate the full n×n attention matrix from the smaller computations

```
┌─────────────────────────────────────────────────────────────────┐
│                    NYSTRÖM APPROXIMATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Standard Attention (n×n):      Nyström Approximation:          │
│                                                                  │
│  ┌─────────────────┐           ┌───┬───────────┐               │
│  │█████████████████│           │ A │     B     │  A: m×m       │
│  │█████████████████│    →      ├───┼───────────┤  B: m×(n-m)   │
│  │█████████████████│           │ C │  B·A⁺·C   │  C: (n-m)×m   │
│  │█████████████████│           └───┴───────────┘               │
│  └─────────────────┘                                            │
│                                                                  │
│  Cost: O(n²)                   Cost: O(n·m)                     │
│                                                                  │
│  With m=64 and n=4096:  64× speedup!                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Advantages for Trading

| Feature | Standard Attention | Nyströmformer | Trading Benefit |
|---------|-------------------|---------------|-----------------|
| Complexity | O(n²) | O(n) | Process longer history |
| Memory | O(n²) | O(n) | Larger batch sizes |
| Sequence Length | ~512-2048 | 4096-8192+ | More context for predictions |
| Inference Speed | Slow for long seq | Fast | Real-time processing |
| Accuracy | Exact | Near-exact | Minimal quality loss |

## Mathematical Foundation

### Standard Self-Attention

The standard self-attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Where:
- **Q** (Query): n × d matrix
- **K** (Key): n × d matrix
- **V** (Value): n × d matrix
- **d**: Head dimension
- **n**: Sequence length

The softmax is applied row-wise to the n×n matrix QK^T, which is the computational bottleneck.

### Nyström Approximation

Nyströmformer approximates the softmax attention matrix **S** = softmax(QK^T/√d) as:

```
Ŝ = F̃ · Ã⁺ · B̃

Where:
- F̃ = softmax(Q · K̃^T / √d)     ∈ ℝ^(n×m)  — Full queries to landmark keys
- Ã = softmax(Q̃ · K̃^T / √d)     ∈ ℝ^(m×m)  — Landmarks to landmarks
- B̃ = softmax(Q̃ · K^T / √d)     ∈ ℝ^(m×n)  — Landmark queries to full keys
- Ã⁺ is the Moore-Penrose pseudoinverse of Ã
```

The final attention output becomes:
```
Output = Ŝ · V = F̃ · Ã⁺ · B̃ · V
```

### Landmark Selection

**Segment-Means Method** (used in Nyströmformer):

Given n tokens, divide them into m segments of size l = n/m:

```python
# For queries Q ∈ ℝ^(n×d)
Q̃[i] = mean(Q[i*l : (i+1)*l])  for i = 0, 1, ..., m-1

# For keys K ∈ ℝ^(n×d)
K̃[i] = mean(K[i*l : (i+1)*l])  for i = 0, 1, ..., m-1
```

This creates m landmark queries Q̃ and m landmark keys K̃, each representing a segment of the input sequence.

```
Input sequence (n=8 tokens):
[t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈]

With m=2 landmarks:
Segment 1: [t₁, t₂, t₃, t₄] → Landmark L₁ = mean([t₁, t₂, t₃, t₄])
Segment 2: [t₅, t₆, t₇, t₈] → Landmark L₂ = mean([t₅, t₆, t₇, t₈])
```

### Iterative Pseudoinverse

Computing the exact pseudoinverse via SVD is expensive on GPUs. Nyströmformer uses an iterative approximation that converges in ~6 iterations:

```python
def iterative_pinv(A, num_iter=6):
    """
    Iterative Moore-Penrose pseudoinverse approximation.
    Based on Newton-Schulz iteration.
    """
    # Initialize
    Z = A.transpose(-1, -2) / (torch.norm(A) ** 2)
    I = torch.eye(A.shape[-1], device=A.device)

    for _ in range(num_iter):
        Z = 2 * Z - Z @ A @ Z

    return Z
```

### Complexity Analysis

| Component | Standard | Nyströmformer |
|-----------|----------|---------------|
| QK^T computation | O(n²d) | O(nmd) |
| Softmax | O(n²) | O(nm + m²) |
| Pseudoinverse | N/A | O(m³) |
| Attention × V | O(n²d) | O(nmd) |
| **Total** | **O(n²d)** | **O(nmd + m³)** |

With m fixed (e.g., m=64), Nyströmformer achieves **linear complexity O(n)** in sequence length.

## Nyströmformer Architecture

### Segment-Means for Landmarks

```python
class NystromAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_landmarks=64, seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Segment size for landmark computation
        self.segment_size = seq_len // num_landmarks

    def compute_landmarks(self, x):
        """
        Compute landmark points using segment-means.
        x: [batch, n_heads, seq_len, head_dim]
        returns: [batch, n_heads, num_landmarks, head_dim]
        """
        batch, n_heads, seq_len, head_dim = x.shape

        # Reshape to segments
        x = x.reshape(
            batch, n_heads,
            self.num_landmarks,
            self.segment_size,
            head_dim
        )

        # Take mean of each segment
        landmarks = x.mean(dim=3)

        return landmarks
```

### Three-Matrix Decomposition

```python
def nystrom_attention(self, Q, K, V):
    """
    Compute Nyström-approximated attention.

    Q, K, V: [batch, n_heads, seq_len, head_dim]
    """
    # Compute landmark queries and keys
    Q_landmarks = self.compute_landmarks(Q)  # [batch, n_heads, m, head_dim]
    K_landmarks = self.compute_landmarks(K)  # [batch, n_heads, m, head_dim]

    scale = 1.0 / math.sqrt(self.head_dim)

    # Matrix F̃: Full queries to landmark keys
    # [batch, n_heads, n, m]
    kernel_1 = F.softmax(
        torch.matmul(Q, K_landmarks.transpose(-1, -2)) * scale,
        dim=-1
    )

    # Matrix Ã: Landmarks to landmarks
    # [batch, n_heads, m, m]
    kernel_2 = F.softmax(
        torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)) * scale,
        dim=-1
    )

    # Matrix B̃: Landmark queries to full keys
    # [batch, n_heads, m, n]
    kernel_3 = F.softmax(
        torch.matmul(Q_landmarks, K.transpose(-1, -2)) * scale,
        dim=-1
    )

    # Compute pseudoinverse of kernel_2
    kernel_2_inv = self.iterative_pinv(kernel_2)

    # Approximate attention: Ŝ = F̃ · Ã⁺ · B̃
    # First: Ã⁺ · B̃ → [batch, n_heads, m, n]
    attn_intermediate = torch.matmul(kernel_2_inv, kernel_3)

    # Then: F̃ · (Ã⁺ · B̃) → [batch, n_heads, n, n] (never materialized!)
    # Instead compute: F̃ · (Ã⁺ · B̃ · V) → [batch, n_heads, n, head_dim]

    # B̃ · V → [batch, n_heads, m, head_dim]
    context_landmarks = torch.matmul(kernel_3, V)

    # Ã⁺ · (B̃ · V) → [batch, n_heads, m, head_dim]
    context_landmarks = torch.matmul(kernel_2_inv, context_landmarks)

    # F̃ · result → [batch, n_heads, n, head_dim]
    output = torch.matmul(kernel_1, context_landmarks)

    return output
```

### Complexity Analysis

```
Memory and Compute Comparison for seq_len=4096, num_landmarks=64:

Standard Attention:
- QK^T: 4096 × 4096 = 16,777,216 elements
- Memory: ~64MB per attention head (fp32)
- Compute: O(n²) = O(16.7M)

Nyströmformer:
- F̃: 4096 × 64 = 262,144 elements
- Ã: 64 × 64 = 4,096 elements
- B̃: 64 × 4096 = 262,144 elements
- Total: ~528,384 elements
- Memory: ~2MB per attention head (fp32)
- Compute: O(n·m) = O(262K)

Reduction: ~32× less memory, ~64× less compute
```

## Trading Applications

### Long Sequence Price Prediction

Nyströmformer excels at processing extended price histories:

```python
class NystromPricePredictor(nn.Module):
    """
    Predicts future price movements using long historical sequences.

    Use cases:
    - Intraday trading with minute-level data (4096 min = ~68 hours)
    - Tick-by-tick analysis (8192 ticks = extended market depth)
    - Multi-day patterns with hourly data
    """
    def __init__(
        self,
        input_dim: int = 6,        # OHLCV + features
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        num_landmarks: int = 64,
        seq_len: int = 4096,
        pred_horizon: int = 24,    # Predict 24 steps ahead
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                num_landmarks=num_landmarks,
                seq_len=seq_len,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.pred_head = nn.Linear(d_model, pred_horizon)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        # Use last position for prediction
        output = self.pred_head(x[:, -1, :])

        return output
```

### Order Book Analysis

Process deep order book history efficiently:

```python
class NystromLOBModel(nn.Module):
    """
    Limit Order Book (LOB) analysis using Nyströmformer.

    Handles long sequences of order book snapshots with:
    - Multiple price levels (10-50 levels per side)
    - Temporal dependencies across thousands of snapshots
    - Cross-level attention patterns
    """
    def __init__(
        self,
        n_levels: int = 20,        # Price levels per side
        d_model: int = 128,
        n_heads: int = 4,
        num_landmarks: int = 32,
        seq_len: int = 2048,       # Order book snapshots
    ):
        super().__init__()

        # Each snapshot: [bid_prices, bid_volumes, ask_prices, ask_volumes]
        input_dim = n_levels * 4

        self.input_proj = nn.Linear(input_dim, d_model)

        self.nystrom_attention = NystromAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_landmarks=num_landmarks,
            seq_len=seq_len
        )

        # Predict: mid-price direction, spread change, volume imbalance
        self.output_head = nn.Linear(d_model, 3)
```

### Multi-Timeframe Fusion

Combine signals from multiple timeframes efficiently:

```python
class MultiTimeframeNystrom(nn.Module):
    """
    Fuse information from multiple timeframes using Nyströmformer.

    Example configuration:
    - 1-minute bars: 1440 samples (1 day)
    - 5-minute bars: 288 samples (1 day)
    - 1-hour bars: 168 samples (1 week)

    Total: 1896 tokens processed efficiently with Nyström attention.
    """
    def __init__(self, d_model=256, num_landmarks=64):
        super().__init__()

        self.timeframe_embeds = nn.ModuleDict({
            '1m': nn.Linear(5, d_model),   # OHLCV
            '5m': nn.Linear(5, d_model),
            '1h': nn.Linear(5, d_model),
        })

        self.timeframe_tokens = nn.ParameterDict({
            '1m': nn.Parameter(torch.randn(1, 1, d_model)),
            '5m': nn.Parameter(torch.randn(1, 1, d_model)),
            '1h': nn.Parameter(torch.randn(1, 1, d_model)),
        })

        # Single Nyströmformer processes all timeframes
        self.nystrom_encoder = NystromEncoder(
            d_model=d_model,
            n_heads=8,
            n_layers=4,
            num_landmarks=num_landmarks,
            seq_len=2048  # Accommodates all timeframes
        )

    def forward(self, data_1m, data_5m, data_1h):
        # Embed each timeframe
        x_1m = self.timeframe_embeds['1m'](data_1m) + self.timeframe_tokens['1m']
        x_5m = self.timeframe_embeds['5m'](data_5m) + self.timeframe_tokens['5m']
        x_1h = self.timeframe_embeds['1h'](data_1h) + self.timeframe_tokens['1h']

        # Concatenate along sequence dimension
        x = torch.cat([x_1m, x_5m, x_1h], dim=1)

        # Process with Nyström attention
        output = self.nystrom_encoder(x)

        return output
```

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 4096,      # Extended lookback for Nyströmformer
    horizon: int = 24,
    source: str = 'bybit'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare long sequence data for Nyströmformer.

    Args:
        symbols: Trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Historical sequence length (can be very long!)
        horizon: Prediction horizon
        source: Data source ('bybit', 'binance', 'yahoo')

    Returns:
        X: Features [n_samples, lookback, n_features]
        y: Targets [n_samples, horizon]
    """
    all_features = []

    for symbol in symbols:
        # Load data from source
        if source == 'bybit':
            df = load_bybit_data(symbol, interval='1m')
        elif source == 'binance':
            df = load_binance_data(symbol, interval='1m')
        else:
            df = load_yahoo_data(symbol)

        # Calculate features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(20).std()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(200).mean()
        df['rsi'] = calculate_rsi(df['close'], period=14)
        df['atr'] = calculate_atr(df, period=14)

        all_features.append(df)

    # Combine features
    features = pd.concat(all_features, axis=1, keys=symbols)
    features = features.dropna()

    # Create sequences
    X, y = [], []
    feature_cols = ['log_return', 'volatility', 'volume_ma_ratio',
                    'price_ma_ratio', 'rsi', 'atr']

    for i in range(lookback, len(features) - horizon):
        # Input: [lookback, n_features * n_symbols]
        x_seq = features.iloc[i-lookback:i][
            [(s, f) for s in symbols for f in feature_cols]
        ].values
        X.append(x_seq)

        # Target: future returns for primary symbol
        y_seq = features.iloc[i:i+horizon][(symbols[0], 'log_return')].values
        y.append(y_seq)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_bybit_data(symbol: str, interval: str = '1m') -> pd.DataFrame:
    """Load historical data from Bybit."""
    import requests

    url = f"https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval_to_bybit(interval),
        'limit': 10000
    }

    response = requests.get(url, params=params)
    data = response.json()['result']['list']

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df
```

### 02: Nyströmformer Model

```python
# python/02_nystromformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NystromAttention(nn.Module):
    """
    Nyström-approximated multi-head self-attention.

    Complexity: O(n·m) instead of O(n²) where m = num_landmarks << n
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_landmarks: int = 64,
        seq_len: int = 512,
        pinv_iterations: int = 6,
        residual_conv: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert seq_len % num_landmarks == 0, "seq_len must be divisible by num_landmarks"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len
        self.segment_size = seq_len // num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Optional: Residual convolution for values (skip connection)
        self.residual_conv = residual_conv
        if residual_conv:
            self.conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=3,
                padding=1,
                groups=d_model  # Depthwise conv
            )

        self.dropout = nn.Dropout(dropout)

    def compute_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute landmark points using segment-means.

        Args:
            x: [batch, n_heads, seq_len, head_dim]

        Returns:
            landmarks: [batch, n_heads, num_landmarks, head_dim]
        """
        batch, n_heads, seq_len, head_dim = x.shape

        # Reshape: [batch, n_heads, num_landmarks, segment_size, head_dim]
        x_segments = x.reshape(
            batch, n_heads,
            self.num_landmarks,
            self.segment_size,
            head_dim
        )

        # Mean over segment dimension
        landmarks = x_segments.mean(dim=3)

        return landmarks

    def iterative_pinv(self, A: torch.Tensor) -> torch.Tensor:
        """
        Iterative Newton-Schulz approximation of Moore-Penrose pseudoinverse.

        More efficient than SVD on GPUs, converges in ~6 iterations.

        Args:
            A: [batch, n_heads, m, m] - landmark attention matrix

        Returns:
            A_pinv: Approximate pseudoinverse of A
        """
        # Initial approximation
        A_T = A.transpose(-1, -2)

        # Normalize for numerical stability
        norm_A = torch.norm(A, dim=(-2, -1), keepdim=True)
        A_normalized = A / (norm_A + 1e-6)
        A_T_normalized = A_T / (norm_A + 1e-6)

        Z = A_T_normalized

        I = torch.eye(
            A.shape[-1],
            device=A.device,
            dtype=A.dtype
        ).unsqueeze(0).unsqueeze(0)

        for _ in range(self.pinv_iterations):
            Z = 0.5 * Z @ (3 * I - A_normalized @ Z)

        # Adjust for normalization
        Z = Z / (norm_A + 1e-6)

        return Z

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Nyström attention forward pass.

        Args:
            x: [batch, seq_len, d_model]
            attention_mask: Optional mask
            return_attention: Whether to return approximate attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: Optional approximate attention
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute landmarks
        Q_landmarks = self.compute_landmarks(Q)  # [batch, n_heads, m, head_dim]
        K_landmarks = self.compute_landmarks(K)  # [batch, n_heads, m, head_dim]

        # Kernel 1: F̃ = softmax(Q @ K̃^T / √d)
        # [batch, n_heads, n, m]
        kernel_1 = F.softmax(
            torch.matmul(Q, K_landmarks.transpose(-1, -2)) * self.scale,
            dim=-1
        )

        # Kernel 2: Ã = softmax(Q̃ @ K̃^T / √d)
        # [batch, n_heads, m, m]
        kernel_2 = F.softmax(
            torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)) * self.scale,
            dim=-1
        )

        # Kernel 3: B̃ = softmax(Q̃ @ K^T / √d)
        # [batch, n_heads, m, n]
        kernel_3 = F.softmax(
            torch.matmul(Q_landmarks, K.transpose(-1, -2)) * self.scale,
            dim=-1
        )

        # Compute pseudoinverse of kernel_2
        kernel_2_inv = self.iterative_pinv(kernel_2)

        # Efficient computation: Never materialize n×n matrix
        # Step 1: B̃ @ V → [batch, n_heads, m, head_dim]
        context_1 = torch.matmul(kernel_3, V)

        # Step 2: Ã⁺ @ (B̃ @ V) → [batch, n_heads, m, head_dim]
        context_2 = torch.matmul(kernel_2_inv, context_1)

        # Step 3: F̃ @ (Ã⁺ @ B̃ @ V) → [batch, n_heads, n, head_dim]
        output = torch.matmul(kernel_1, context_2)

        # Optional: Add residual convolution on values
        if self.residual_conv:
            V_residual = V.transpose(1, 2).reshape(batch, seq_len, -1)
            V_residual = self.conv(V_residual.transpose(1, 2)).transpose(1, 2)
            V_residual = V_residual.view(batch, seq_len, self.n_heads, self.head_dim)
            V_residual = V_residual.transpose(1, 2)
            output = output + V_residual

        # Reshape back
        output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(output)
        output = self.dropout(output)

        # Optionally return attention approximation
        attention_weights = None
        if return_attention:
            # Approximate attention: F̃ @ Ã⁺ @ B̃ (compute for visualization only)
            attention_weights = torch.matmul(
                kernel_1,
                torch.matmul(kernel_2_inv, kernel_3)
            )

        return output, attention_weights


class NystromEncoderLayer(nn.Module):
    """Single Nyströmformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_landmarks: int = 64,
        seq_len: int = 512,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = NystromAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_landmarks=num_landmarks,
            seq_len=seq_len,
            dropout=dropout
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class NystromformerTrading(nn.Module):
    """
    Complete Nyströmformer for trading applications.
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        num_landmarks: int = 64,
        seq_len: int = 4096,
        output_type: str = 'regression',  # 'regression', 'classification', 'allocation'
        pred_horizon: int = 24,
        n_classes: int = 3,  # For classification: down, neutral, up
        dropout: float = 0.1
    ):
        super().__init__()

        self.output_type = output_type
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = self._create_pos_encoding(seq_len, d_model)

        # Encoder layers
        self.layers = nn.ModuleList([
            NystromEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                num_landmarks=num_landmarks,
                seq_len=seq_len,
                dim_feedforward=d_model * 4,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output head based on task type
        if output_type == 'regression':
            self.head = nn.Linear(d_model, pred_horizon)
        elif output_type == 'classification':
            self.head = nn.Linear(d_model, n_classes)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, pred_horizon),
                nn.Tanh()  # Bound allocations to [-1, 1]
            )

    def _create_pos_encoding(self, seq_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            output: Predictions based on output_type
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Encode
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Use last position for prediction
        x = x[:, -1, :]

        # Output head
        output = self.head(x)

        return output
```

### 03: Training Pipeline

```python
# python/03_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NystromTrainer:
    """Training pipeline for Nyströmformer trading model."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )

        # Loss based on output type
        if model.output_type == 'regression':
            self.criterion = nn.MSELoss()
        elif model.output_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif model.output_type == 'allocation':
            self.criterion = self._sharpe_loss

    def _sharpe_loss(self, allocations: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Sharpe ratio loss for portfolio allocation.

        Args:
            allocations: Model predicted allocations [batch, horizon]
            returns: Actual returns [batch, horizon]

        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Portfolio returns
        portfolio_returns = allocations * returns

        # Sharpe ratio (negative for minimization)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() + 1e-8
        sharpe = mean_return / std_return

        return -sharpe

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            total_loss += loss.item()
            all_preds.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        metrics = {
            'loss': total_loss / len(dataloader),
        }

        if self.model.output_type == 'regression':
            mse = ((all_preds - all_targets) ** 2).mean().item()
            mae = (all_preds - all_targets).abs().mean().item()
            metrics['mse'] = mse
            metrics['mae'] = mae

            # Directional accuracy
            pred_direction = (all_preds[:, 0] > 0).float()
            true_direction = (all_targets[:, 0] > 0).float()
            metrics['direction_accuracy'] = (pred_direction == true_direction).float().mean().item()

        elif self.model.output_type == 'classification':
            pred_classes = all_preds.argmax(dim=1)
            metrics['accuracy'] = (pred_classes == all_targets).float().mean().item()

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict:
        """Full training loop with early stopping."""

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val Metrics: {val_metrics}"
            )

            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history


def main():
    """Example training script."""
    from data import prepare_long_sequence_data

    # Prepare data
    logger.info("Loading data...")
    X, y = prepare_long_sequence_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        lookback=4096,
        horizon=24,
        source='bybit'
    )

    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    model = NystromformerTrading(
        input_dim=X.shape[-1],
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_landmarks=64,
        seq_len=4096,
        output_type='regression',
        pred_horizon=24
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = NystromTrainer(model)
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=100,
        patience=15,
        save_path='checkpoints/nystromformer_best.pt'
    )

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
```

### 04: Backtesting Strategy

```python
# python/04_backtest.py

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    risk_per_trade: float = 0.02  # 2% risk per trade


class NystromBacktester:
    """Backtesting engine for Nyströmformer trading strategy."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: BacktestConfig = BacktestConfig()
    ):
        self.model = model
        self.config = config
        self.model.eval()

    @torch.no_grad()
    def generate_signals(
        self,
        data: np.ndarray,
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        Generate trading signals from model predictions.

        Args:
            data: [n_samples, seq_len, n_features]
            threshold: Minimum predicted return to trigger signal

        Returns:
            signals: [n_samples] with values in {-1, 0, 1}
        """
        self.model.eval()

        # Get predictions
        x = torch.tensor(data, dtype=torch.float32)
        predictions = self.model(x).numpy()

        # Use first prediction horizon step
        pred_returns = predictions[:, 0] if len(predictions.shape) > 1 else predictions

        # Generate signals
        signals = np.zeros_like(pred_returns)
        signals[pred_returns > threshold] = 1   # Long signal
        signals[pred_returns < -threshold] = -1  # Short signal

        return signals

    def run_backtest(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict:
        """
        Run full backtest simulation.

        Args:
            data: Feature data [n_samples, seq_len, n_features]
            prices: Price series aligned with predictions
            timestamps: Timestamps for the backtest period

        Returns:
            Dictionary with backtest results
        """
        signals = self.generate_signals(data)

        # Initialize tracking
        capital = self.config.initial_capital
        position = 0.0
        position_price = 0.0

        # Results tracking
        equity_curve = [capital]
        positions = [0.0]
        returns = []
        trades = []

        for i in range(len(signals)):
            current_price = prices[i]
            signal = signals[i]

            # Calculate position change
            target_position = signal * self.config.max_position_size
            position_change = target_position - position

            if abs(position_change) > 0.01:  # Minimum position change
                # Calculate trade costs
                trade_value = abs(position_change) * capital
                costs = trade_value * (self.config.transaction_cost + self.config.slippage)

                # Execute trade
                if position_change > 0:  # Buying
                    capital -= costs
                else:  # Selling
                    capital -= costs

                trades.append({
                    'timestamp': timestamps[i],
                    'price': current_price,
                    'signal': signal,
                    'position_change': position_change,
                    'costs': costs
                })

                position = target_position
                position_price = current_price

            # Calculate P&L if we have a position
            if i > 0 and position != 0:
                price_return = (current_price - prices[i-1]) / prices[i-1]
                pnl = position * capital * price_return
                capital += pnl
                returns.append(pnl / equity_curve[-1])
            else:
                returns.append(0.0)

            equity_curve.append(capital)
            positions.append(position)

        # Calculate metrics
        returns = np.array(returns)
        equity_curve = np.array(equity_curve)

        metrics = self._calculate_metrics(returns, equity_curve, trades)

        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'returns': returns,
            'trades': trades,
            'timestamps': timestamps,
            'metrics': metrics
        }

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: List[Dict]
    ) -> Dict:
        """Calculate performance metrics."""

        # Basic metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Sharpe ratio (assuming daily returns, 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino_ratio = np.sqrt(252) * returns.mean() / (downside_std + 1e-8)
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades),
            'final_capital': equity_curve[-1]
        }

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualize backtest results."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Equity curve
        axes[0].plot(results['timestamps'], results['equity_curve'][1:])
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Capital ($)')
        axes[0].grid(True)

        # Positions
        axes[1].plot(results['timestamps'], results['positions'][1:])
        axes[1].set_title('Position Size')
        axes[1].set_ylabel('Position')
        axes[1].grid(True)

        # Drawdown
        equity = np.array(results['equity_curve'][1:])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        axes[2].fill_between(results['timestamps'], drawdown, 0, alpha=0.5, color='red')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        # Print metrics
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:>12.4f}")
            else:
                print(f"{metric:20s}: {value:>12}")
        print("="*50)


def main():
    """Example backtest."""
    from model import NystromformerTrading
    from data import prepare_long_sequence_data, load_bybit_data

    # Load model
    model = NystromformerTrading(
        input_dim=12,
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_landmarks=64,
        seq_len=4096,
        output_type='regression',
        pred_horizon=24
    )
    model.load_state_dict(torch.load('checkpoints/nystromformer_best.pt'))

    # Prepare test data
    X, y = prepare_long_sequence_data(
        symbols=['BTCUSDT', 'ETHUSDT'],
        lookback=4096,
        horizon=24,
        source='bybit'
    )

    # Get corresponding prices
    price_data = load_bybit_data('BTCUSDT')

    # Use last 20% for testing
    test_start = int(0.8 * len(X))
    X_test = X[test_start:]
    prices = price_data['close'].values[4096+test_start:4096+test_start+len(X_test)]
    timestamps = price_data['timestamp'].values[4096+test_start:4096+test_start+len(X_test)]

    # Run backtest
    backtester = NystromBacktester(model)
    results = backtester.run_backtest(X_test, prices, pd.DatetimeIndex(timestamps))

    # Visualize
    backtester.plot_results(results, save_path='backtest_results.png')


if __name__ == '__main__':
    main()
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation using Bybit data.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client for Bybit
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
│   ├── model/              # Nyströmformer architecture
│   │   ├── mod.rs
│   │   ├── attention.rs    # Nyström attention implementation
│   │   ├── encoder.rs      # Encoder layers
│   │   └── nystromformer.rs # Complete model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download Bybit data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── model.py              # Nyströmformer implementation
├── data.py               # Data loading and preprocessing
├── features.py           # Feature engineering
├── train.py              # Training pipeline
├── backtest.py           # Backtesting utilities
├── requirements.txt      # Dependencies
└── examples/
    ├── 01_data_preparation.ipynb
    ├── 02_model_training.ipynb
    ├── 03_backtesting.ipynb
    └── 04_visualization.ipynb
```

### Quick Start (Python)

```bash
# Install dependencies
cd python
pip install -r requirements.txt

# Fetch data
python data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Train model
python train.py --config configs/default.yaml

# Run backtest
python backtest.py --model checkpoints/nystromformer_best.pt
```

## Best Practices

### When to Use Nyströmformer

**Ideal use cases:**
- Processing sequences >2048 tokens (tick data, LOB snapshots)
- Real-time inference with limited compute
- Multi-timeframe analysis requiring long context
- Resource-constrained deployment

**Consider alternatives when:**
- Sequence length <512 (standard attention is fine)
- Exact attention is critical (some precision loss)
- Very sparse attention patterns (use sparse attention instead)

### Hyperparameter Guidelines

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `num_landmarks` | 32-64 | More landmarks = better approximation, more compute |
| `seq_len` | 2048-8192 | Power of 2, divisible by landmarks |
| `d_model` | 128-512 | Match to task complexity |
| `n_layers` | 3-6 | More layers for complex patterns |
| `pinv_iterations` | 6 | Sufficient for convergence |

### Common Pitfalls

1. **Sequence length not divisible by landmarks**: Ensure seq_len % num_landmarks == 0
2. **Insufficient landmarks**: Use at least 32 for long sequences
3. **Numerical instability**: Use proper normalization in pseudoinverse
4. **Memory issues**: Reduce batch size or landmarks if OOM

## Resources

### Papers

- [Nyströmformer: A Nyström-based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902) — Original paper (AAAI 2021)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) — Comparison of efficient attention methods
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) — Related architecture

### Implementations

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/nystromformer) — Official HF implementation
- [Official Repository](https://github.com/mlpen/Nystromformer) — Original authors' code
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) — Time series library

### Related Chapters

- [Chapter 44: ProbSparse Attention](../44_probsparse_attention) — Another efficient attention mechanism
- [Chapter 51: Linformer Long Sequences](../51_linformer_long_sequences) — Linear attention approach
- [Chapter 52: Performer Efficient Attention](../52_performer_efficient_attention) — Random feature attention
- [Chapter 57: Longformer Financial](../57_longformer_financial) — Sliding window + global attention

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Understanding of transformer architecture and self-attention
- Familiarity with matrix approximation methods
- Basic knowledge of time series forecasting
- PyTorch or Rust ML library experience
