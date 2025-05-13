//! # Nyströmformer Trading
//!
//! Efficient O(n) attention mechanism for long-sequence financial time series
//! analysis using the Nyström approximation method.
//!
//! ## Features
//!
//! - Nyström attention with O(n) complexity instead of O(n²)
//! - Segment-means landmark selection for stable approximation
//! - Iterative pseudoinverse using Newton-Schulz method
//! - Support for multiple output types: regression, classification, allocation
//! - Integration with Bybit API for cryptocurrency data
//!
//! ## Key Concepts
//!
//! The Nyström method approximates the full attention matrix by:
//! 1. Selecting m << n landmark points (segment means)
//! 2. Computing three smaller matrices:
//!    - F̃ = softmax(Q·K̃ᵀ) [n × m]
//!    - Ã = softmax(Q̃·K̃ᵀ) [m × m]
//!    - B̃ = softmax(Q̃·Kᵀ) [m × n]
//! 3. Approximating attention: Ŝ ≈ F̃ · Ã⁺ · B̃
//!
//! ## Example Usage
//!
//! ```no_run
//! use nystromformer_trading::{
//!     BybitClient, SequenceLoader, NystromformerConfig,
//!     NystromformerModel, OutputType
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     // Fetch data from Bybit
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1m", 10000).await.unwrap();
//!
//!     // Prepare long sequences
//!     let loader = SequenceLoader::new();
//!     let dataset = loader.prepare_dataset(&klines, 4096, 24).unwrap();
//!
//!     // Create Nyströmformer model
//!     let config = NystromformerConfig {
//!         d_model: 128,
//!         n_heads: 4,
//!         n_layers: 3,
//!         num_landmarks: 64,
//!         seq_len: 4096,
//!         output_type: OutputType::Regression,
//!         ..Default::default()
//!     };
//!     let model = NystromformerModel::new(config);
//!
//!     // Make predictions with attention weights using dataset's features
//!     let (predictions, attention) = model.predict_with_attention(&dataset.x);
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, BybitError, Kline, OrderBook, Ticker};
pub use data::{SequenceLoader, TradingDataset, Features};
pub use model::{
    NystromAttention, NystromEncoderLayer, NystromformerConfig,
    NystromformerModel, OutputType, AttentionWeights,
};
pub use strategy::{BacktestResult, TradingSignal, SignalGenerator, NystromBacktester, BacktestConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    /// Hidden layer dimension
    pub const D_MODEL: usize = 128;

    /// Number of attention heads
    pub const N_HEADS: usize = 4;

    /// Number of encoder layers
    pub const N_LAYERS: usize = 3;

    /// Number of Nyström landmarks
    pub const NUM_LANDMARKS: usize = 64;

    /// Dropout rate
    pub const DROPOUT: f64 = 0.1;

    /// Default sequence length (long for Nyström efficiency)
    pub const SEQ_LEN: usize = 4096;

    /// Prediction horizon
    pub const PRED_HORIZON: usize = 24;

    /// Learning rate
    pub const LEARNING_RATE: f64 = 0.0001;

    /// Batch size
    pub const BATCH_SIZE: usize = 16;

    /// Number of training epochs
    pub const EPOCHS: usize = 100;

    /// Newton-Schulz iterations for pseudoinverse
    pub const PINV_ITERATIONS: usize = 6;

    /// Regularization epsilon for numerical stability
    pub const EPSILON: f64 = 1e-6;
}
