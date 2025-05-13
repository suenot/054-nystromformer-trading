# Nyströmformer Trading - Rust Implementation

High-performance implementation of Nyströmformer for efficient long-sequence financial time series analysis.

## Features

- **O(n) Attention Complexity**: Nyström approximation reduces attention from O(n²) to O(n)
- **Segment-Means Landmarks**: Stable landmark selection through segment averaging
- **Newton-Schulz Pseudoinverse**: Efficient iterative computation of Ã⁺
- **Bybit API Integration**: Real-time cryptocurrency data fetching
- **Comprehensive Backtesting**: Full trading simulation with risk management

## Quick Start

```bash
# Build the library
cargo build --release

# Run examples
cargo run --example fetch_data
cargo run --example train
cargo run --example predict
cargo run --example backtest
```

## Usage

### Basic Model Creation

```rust
use nystromformer_trading::{
    NystromformerConfig, NystromformerModel, OutputType
};

// Configure for long sequences
let config = NystromformerConfig {
    input_dim: 6,
    d_model: 128,
    n_heads: 4,
    n_layers: 3,
    num_landmarks: 64,    // Only 64 landmarks for 4096 sequence
    seq_len: 4096,        // Process 4096 time steps efficiently
    pred_horizon: 24,
    output_type: OutputType::Regression,
    ..Default::default()
};

let model = NystromformerModel::new(config);
```

### Fetching Data from Bybit

```rust
use nystromformer_trading::{BybitClient, SequenceLoader};

#[tokio::main]
async fn main() {
    let client = BybitClient::new();

    // Fetch 1-minute candles
    let klines = client.get_klines("BTCUSDT", "1", 5000).await?;

    // Prepare dataset
    let loader = SequenceLoader::new();
    let dataset = loader.prepare_dataset(&klines, 4096, 24)?;

    println!("Samples: {}, Features: {}", dataset.len(), dataset.num_features());
}
```

### Making Predictions

```rust
use nystromformer_trading::{NystromformerModel, SignalGenerator};
use ndarray::Array3;

// Forward pass
let (predictions, attention_weights) = model.forward(&x_batch);

// Generate trading signals
let generator = SignalGenerator::new(0.001, -0.001);
let signals = generator.generate(&predictions);

// Analyze attention patterns
let top_connections = attention_weights.top_k_landmarks(5);
```

### Backtesting

```rust
use nystromformer_trading::{
    BacktestConfig, NystromBacktester, NystromformerModel
};

let config = BacktestConfig {
    initial_capital: 100_000.0,
    transaction_cost: 0.001,
    slippage: 0.0005,
    max_position_size: 0.5,
    use_stop_loss: true,
    stop_loss_pct: 0.02,
    ..Default::default()
};

let backtester = NystromBacktester::new(model, config);
let result = backtester.run_backtest(&test_x, &test_prices);

result.print_summary();
// Output: Sharpe Ratio, Max Drawdown, Win Rate, etc.
```

## Module Structure

```
src/
├── lib.rs           # Main library exports
├── model/
│   ├── attention.rs # Nyström attention mechanism
│   ├── config.rs    # Model configuration
│   └── encoder.rs   # Encoder layers and full model
├── api/
│   ├── client.rs    # Bybit API client
│   └── types.rs     # Kline, OrderBook, Ticker types
├── data/
│   ├── features.rs  # Technical indicators (RSI, MACD, etc.)
│   └── loader.rs    # Data loading and preprocessing
└── strategy/
    ├── signals.rs   # Trading signal generation
    └── backtest.rs  # Backtesting engine
```

## Key Algorithms

### Nyström Attention Approximation

```
Full Attention (O(n²)):
    Attention = softmax(Q·Kᵀ/√d) · V

Nyström Approximation (O(n)):
    1. Select m landmarks using segment-means
    2. Compute F̃ = softmax(Q·K̃ᵀ/√d)    [n × m]
    3. Compute Ã = softmax(Q̃·K̃ᵀ/√d)    [m × m]
    4. Compute B̃ = softmax(Q̃·Kᵀ/√d)    [m × n]
    5. Approximate: Ŝ·V ≈ F̃ · Ã⁺ · B̃ · V
```

### Newton-Schulz Pseudoinverse

```
Z₀ = α·Aᵀ  where α = 1/‖A‖²
Zₙ₊₁ = Zₙ · (2I - A·Zₙ)

After 6 iterations: Zₙ ≈ A⁺
```

## Performance Benchmarks

| Sequence Length | Full Attention | Nyström (m=64) | Speedup |
|-----------------|----------------|----------------|---------|
| 512             | 262,144        | 65,536         | 4x      |
| 1024            | 1,048,576      | 131,072        | 8x      |
| 4096            | 16,777,216     | 524,288        | 32x     |
| 8192            | 67,108,864     | 1,048,576      | 64x     |

## Technical Indicators

The data loader computes these features automatically:

- Log Returns (normalized)
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Rolling Volatility
- Bollinger Band %B
- Volume Ratio
- MACD Histogram

## Configuration Options

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 3 | Number of encoder layers |
| `num_landmarks` | 64 | Nyström landmarks |
| `seq_len` | 4096 | Input sequence length |
| `pred_horizon` | 24 | Prediction horizon |
| `pinv_iterations` | 6 | Newton-Schulz iterations |

### Backtest Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 100,000 | Starting capital |
| `transaction_cost` | 0.1% | Per-trade cost |
| `slippage` | 0.05% | Execution slippage |
| `max_position_size` | 50% | Maximum position |
| `stop_loss_pct` | 2% | Stop loss trigger |

## License

MIT License
