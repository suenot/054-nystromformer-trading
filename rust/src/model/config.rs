//! Configuration for Nyströmformer model

use crate::defaults;

/// Output type for the model
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputType {
    /// Regression: predict continuous future returns
    Regression,
    /// Classification: predict direction (up/down/neutral)
    Classification,
    /// Allocation: predict portfolio weights
    Allocation,
}

impl Default for OutputType {
    fn default() -> Self {
        OutputType::Regression
    }
}

/// Configuration for Nyströmformer model
#[derive(Debug, Clone)]
pub struct NystromformerConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Number of Nyström landmarks (m)
    pub num_landmarks: usize,
    /// Input sequence length
    pub seq_len: usize,
    /// Prediction horizon
    pub pred_horizon: usize,
    /// Number of classes (for classification)
    pub num_classes: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Output type
    pub output_type: OutputType,
    /// Newton-Schulz iterations for pseudoinverse
    pub pinv_iterations: usize,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Whether to use residual connections
    pub use_residual: bool,
    /// Whether to use layer normalization
    pub use_layer_norm: bool,
    /// FFN hidden dimension multiplier
    pub ffn_multiplier: usize,
}

impl Default for NystromformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 6,
            d_model: defaults::D_MODEL,
            n_heads: defaults::N_HEADS,
            n_layers: defaults::N_LAYERS,
            num_landmarks: defaults::NUM_LANDMARKS,
            seq_len: defaults::SEQ_LEN,
            pred_horizon: defaults::PRED_HORIZON,
            num_classes: 3,
            dropout: defaults::DROPOUT,
            output_type: OutputType::Regression,
            pinv_iterations: defaults::PINV_ITERATIONS,
            epsilon: defaults::EPSILON,
            use_residual: true,
            use_layer_norm: true,
            ffn_multiplier: 4,
        }
    }
}

impl NystromformerConfig {
    /// Creates a new configuration with custom parameters
    pub fn new(
        input_dim: usize,
        d_model: usize,
        n_heads: usize,
        num_landmarks: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            input_dim,
            d_model,
            n_heads,
            num_landmarks,
            seq_len,
            ..Default::default()
        }
    }

    /// Validates the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }

        // num_landmarks must be positive to avoid divide-by-zero in segment_size
        if self.num_landmarks == 0 {
            return Err("num_landmarks must be greater than 0".to_string());
        }

        if self.num_landmarks > self.seq_len {
            return Err(format!(
                "num_landmarks ({}) cannot exceed seq_len ({})",
                self.num_landmarks, self.seq_len
            ));
        }

        if self.seq_len % self.num_landmarks != 0 {
            return Err(format!(
                "seq_len ({}) should be divisible by num_landmarks ({}) for segment-means",
                self.seq_len, self.num_landmarks
            ));
        }

        Ok(())
    }

    /// Returns the head dimension
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Returns the segment size for landmark computation
    pub fn segment_size(&self) -> usize {
        self.seq_len / self.num_landmarks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NystromformerConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.num_landmarks, 64);
    }

    #[test]
    fn test_config_validation() {
        let mut config = NystromformerConfig::default();

        // Valid config
        assert!(config.validate().is_ok());

        // Invalid: d_model not divisible by n_heads
        config.d_model = 100;
        config.n_heads = 3;
        assert!(config.validate().is_err());

        // Invalid: landmarks > seq_len
        config.d_model = 128;
        config.n_heads = 4;
        config.num_landmarks = 5000;
        config.seq_len = 4096;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim() {
        let config = NystromformerConfig {
            d_model: 128,
            n_heads: 8,
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 16);
    }

    #[test]
    fn test_segment_size() {
        let config = NystromformerConfig {
            seq_len: 4096,
            num_landmarks: 64,
            ..Default::default()
        };
        assert_eq!(config.segment_size(), 64);
    }
}
