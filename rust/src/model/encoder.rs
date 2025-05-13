//! Nyströmformer Encoder and Full Model
//!
//! Implements the complete Nyströmformer architecture for trading applications.

use ndarray::{Array1, Array2, Array3};

use crate::model::attention::{AttentionWeights, NystromAttention};
use crate::model::config::{NystromformerConfig, OutputType};

/// Generates a random number from standard normal distribution
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Nyström Encoder Layer
///
/// Consists of:
/// 1. Nyström multi-head self-attention
/// 2. Layer normalization
/// 3. Feed-forward network
/// 4. Residual connections
#[derive(Debug, Clone)]
pub struct NystromEncoderLayer {
    /// Nyström attention
    attention: NystromAttention,
    /// FFN first layer [d_model, d_ff]
    ffn_w1: Array2<f64>,
    /// FFN second layer [d_ff, d_model]
    ffn_w2: Array2<f64>,
    /// FFN bias 1
    ffn_b1: Array1<f64>,
    /// FFN bias 2
    ffn_b2: Array1<f64>,
    /// Layer norm weights for attention
    ln1_gamma: Array1<f64>,
    ln1_beta: Array1<f64>,
    /// Layer norm weights for FFN
    ln2_gamma: Array1<f64>,
    ln2_beta: Array1<f64>,
    /// Model dimension
    d_model: usize,
    /// FFN hidden dimension
    d_ff: usize,
    /// Whether to use residual connections
    use_residual: bool,
    /// Whether to use layer normalization
    use_layer_norm: bool,
    /// Epsilon for layer norm
    epsilon: f64,
}

impl NystromEncoderLayer {
    /// Creates a new encoder layer
    pub fn new(config: &NystromformerConfig) -> Self {
        let d_model = config.d_model;
        let d_ff = d_model * config.ffn_multiplier;

        // Xavier initialization for FFN
        let scale_w1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale_w2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        let ffn_w1 = Array2::from_shape_fn((d_model, d_ff), |_| rand_normal() * scale_w1);
        let ffn_w2 = Array2::from_shape_fn((d_ff, d_model), |_| rand_normal() * scale_w2);
        let ffn_b1 = Array1::zeros(d_ff);
        let ffn_b2 = Array1::zeros(d_model);

        // Layer norm parameters
        let ln1_gamma = Array1::ones(d_model);
        let ln1_beta = Array1::zeros(d_model);
        let ln2_gamma = Array1::ones(d_model);
        let ln2_beta = Array1::zeros(d_model);

        Self {
            attention: NystromAttention::new(config),
            ffn_w1,
            ffn_w2,
            ffn_b1,
            ffn_b2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            d_model,
            d_ff,
            use_residual: config.use_residual,
            use_layer_norm: config.use_layer_norm,
            epsilon: config.epsilon,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - [batch, seq_len, d_model]
    ///
    /// # Returns
    /// * `output` - [batch, seq_len, d_model]
    /// * `weights` - Attention weights
    pub fn forward(&self, x: &Array3<f64>) -> (Array3<f64>, AttentionWeights) {
        let (_batch_size, _seq_len, _) = x.dim();

        // Self-attention with Nyström approximation
        let (attn_out, weights) = self.attention.forward(x);

        // Add & Norm (post-attention)
        let mut hidden = if self.use_residual {
            self.add_tensors(x, &attn_out)
        } else {
            attn_out
        };

        if self.use_layer_norm {
            hidden = self.layer_norm(&hidden, &self.ln1_gamma, &self.ln1_beta);
        }

        // Feed-forward network
        let ffn_out = self.feed_forward(&hidden);

        // Add & Norm (post-FFN)
        let mut output = if self.use_residual {
            self.add_tensors(&hidden, &ffn_out)
        } else {
            ffn_out
        };

        if self.use_layer_norm {
            output = self.layer_norm(&output, &self.ln2_gamma, &self.ln2_beta);
        }

        (output, weights)
    }

    /// Feed-forward network with GELU activation
    fn feed_forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let mut hidden = Array3::zeros((batch_size, seq_len, self.d_ff));
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        // First linear layer + GELU
        for b in 0..batch_size {
            for t in 0..seq_len {
                for f in 0..self.d_ff {
                    let mut sum = self.ffn_b1[f];
                    for d in 0..self.d_model {
                        sum += x[[b, t, d]] * self.ffn_w1[[d, f]];
                    }
                    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                    hidden[[b, t, f]] = self.gelu(sum);
                }
            }
        }

        // Second linear layer
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..self.d_model {
                    let mut sum = self.ffn_b2[d];
                    for f in 0..self.d_ff {
                        sum += hidden[[b, t, f]] * self.ffn_w2[[f, d]];
                    }
                    output[[b, t, d]] = sum;
                }
            }
        }

        output
    }

    /// GELU activation function
    fn gelu(&self, x: f64) -> f64 {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        x: &Array3<f64>,
        gamma: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Compute mean
                let mean: f64 = (0..d_model).map(|d| x[[b, t, d]]).sum::<f64>() / d_model as f64;

                // Compute variance
                let var: f64 = (0..d_model)
                    .map(|d| (x[[b, t, d]] - mean).powi(2))
                    .sum::<f64>()
                    / d_model as f64;

                // Normalize
                let std = (var + self.epsilon).sqrt();
                for d in 0..d_model {
                    output[[b, t, d]] = gamma[d] * (x[[b, t, d]] - mean) / std + beta[d];
                }
            }
        }

        output
    }

    /// Element-wise addition of two tensors
    fn add_tensors(&self, a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = a.dim();
        let mut result = Array3::zeros((batch_size, seq_len, d_model));

        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..d_model {
                    result[[i, j, k]] = a[[i, j, k]] + b[[i, j, k]];
                }
            }
        }

        result
    }
}

/// Complete Nyströmformer Model for Trading
#[derive(Debug, Clone)]
pub struct NystromformerModel {
    /// Input projection [input_dim, d_model]
    input_proj: Array2<f64>,
    /// Positional encoding [seq_len, d_model]
    pos_encoding: Array2<f64>,
    /// Encoder layers
    layers: Vec<NystromEncoderLayer>,
    /// Output projection for regression [d_model, pred_horizon]
    output_proj_reg: Array2<f64>,
    /// Output projection for classification [d_model, num_classes]
    output_proj_cls: Array2<f64>,
    /// Output projection for allocation [d_model, 1]
    output_proj_alloc: Array2<f64>,
    /// Configuration
    config: NystromformerConfig,
}

impl NystromformerModel {
    /// Creates a new Nyströmformer model
    pub fn new(config: NystromformerConfig) -> Self {
        // Input projection
        let scale_input = (2.0 / (config.input_dim + config.d_model) as f64).sqrt();
        let input_proj = Array2::from_shape_fn((config.input_dim, config.d_model), |_| {
            rand_normal() * scale_input
        });

        // Sinusoidal positional encoding
        let pos_encoding = Self::create_positional_encoding(config.seq_len, config.d_model);

        // Create encoder layers
        let layers: Vec<NystromEncoderLayer> = (0..config.n_layers)
            .map(|_| NystromEncoderLayer::new(&config))
            .collect();

        // Output projections
        let scale_out = (2.0 / (config.d_model + config.pred_horizon) as f64).sqrt();
        let output_proj_reg = Array2::from_shape_fn((config.d_model, config.pred_horizon), |_| {
            rand_normal() * scale_out
        });

        let scale_cls = (2.0 / (config.d_model + config.num_classes) as f64).sqrt();
        let output_proj_cls = Array2::from_shape_fn((config.d_model, config.num_classes), |_| {
            rand_normal() * scale_cls
        });

        let scale_alloc = (2.0 / (config.d_model + 1) as f64).sqrt();
        let output_proj_alloc =
            Array2::from_shape_fn((config.d_model, 1), |_| rand_normal() * scale_alloc);

        Self {
            input_proj,
            pos_encoding,
            layers,
            output_proj_reg,
            output_proj_cls,
            output_proj_alloc,
            config,
        }
    }

    /// Creates sinusoidal positional encoding
    fn create_positional_encoding(seq_len: usize, d_model: usize) -> Array2<f64> {
        let mut pe = Array2::zeros((seq_len, d_model));

        for pos in 0..seq_len {
            for i in 0..(d_model / 2) {
                let div_term = (10000.0_f64).powf(-2.0 * i as f64 / d_model as f64);
                pe[[pos, 2 * i]] = (pos as f64 * div_term).sin();
                if 2 * i + 1 < d_model {
                    pe[[pos, 2 * i + 1]] = (pos as f64 * div_term).cos();
                }
            }
        }

        pe
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - [batch, seq_len, input_dim]
    ///
    /// # Returns
    /// * `output` - Predictions based on output_type
    /// * `weights` - Attention weights from last layer
    pub fn forward(&self, x: &Array3<f64>) -> (Array2<f64>, AttentionWeights) {
        let (batch_size, seq_len, _) = x.dim();

        // Input projection
        let mut hidden = self.project_input(x);

        // Add positional encoding
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..self.config.d_model {
                    hidden[[b, t, d]] += self.pos_encoding[[t, d]];
                }
            }
        }

        // Pass through encoder layers
        let mut last_weights = AttentionWeights::new();
        for layer in &self.layers {
            let (out, weights) = layer.forward(&hidden);
            hidden = out;
            last_weights = weights;
        }

        // Global average pooling over sequence dimension
        let pooled = self.global_avg_pool(&hidden);

        // Output projection based on type
        let output = match self.config.output_type {
            OutputType::Regression => self.linear_2d(&pooled, &self.output_proj_reg),
            OutputType::Classification => {
                let logits = self.linear_2d(&pooled, &self.output_proj_cls);
                self.softmax_2d(&logits)
            }
            OutputType::Allocation => {
                let raw = self.linear_2d(&pooled, &self.output_proj_alloc);
                self.sigmoid_2d(&raw)
            }
        };

        (output, last_weights)
    }

    /// Makes predictions with attention weights
    pub fn predict_with_attention(&self, x: &Array3<f64>) -> (Array2<f64>, AttentionWeights) {
        self.forward(x)
    }

    /// Projects input to d_model dimension
    fn project_input(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.config.d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..self.config.d_model {
                    let mut sum = 0.0;
                    for i in 0..self.config.input_dim {
                        sum += x[[b, t, i]] * self.input_proj[[i, d]];
                    }
                    output[[b, t, d]] = sum;
                }
            }
        }

        output
    }

    /// Global average pooling over sequence
    fn global_avg_pool(&self, x: &Array3<f64>) -> Array2<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array2::zeros((batch_size, d_model));

        for b in 0..batch_size {
            for d in 0..d_model {
                let sum: f64 = (0..seq_len).map(|t| x[[b, t, d]]).sum();
                output[[b, d]] = sum / seq_len as f64;
            }
        }

        output
    }

    /// 2D linear transformation
    fn linear_2d(&self, x: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
        let (batch_size, d_in) = x.dim();
        let d_out = w.dim().1;
        let mut output = Array2::zeros((batch_size, d_out));

        for b in 0..batch_size {
            for o in 0..d_out {
                let mut sum = 0.0;
                for i in 0..d_in {
                    sum += x[[b, i]] * w[[i, o]];
                }
                output[[b, o]] = sum;
            }
        }

        output
    }

    /// Softmax for 2D tensor
    fn softmax_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        let (batch_size, num_classes) = x.dim();
        let mut output = Array2::zeros((batch_size, num_classes));

        for b in 0..batch_size {
            let max_val = (0..num_classes)
                .map(|c| x[[b, c]])
                .fold(f64::NEG_INFINITY, f64::max);

            let exp_sum: f64 = (0..num_classes).map(|c| (x[[b, c]] - max_val).exp()).sum();

            for c in 0..num_classes {
                output[[b, c]] = (x[[b, c]] - max_val).exp() / (exp_sum + self.config.epsilon);
            }
        }

        output
    }

    /// Sigmoid for 2D tensor
    fn sigmoid_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        let (batch_size, d) = x.dim();
        let mut output = Array2::zeros((batch_size, d));

        for b in 0..batch_size {
            for i in 0..d {
                output[[b, i]] = 1.0 / (1.0 + (-x[[b, i]]).exp());
            }
        }

        output
    }

    /// Returns model configuration
    pub fn config(&self) -> &NystromformerConfig {
        &self.config
    }

    /// Returns total number of trainable parameters
    ///
    /// Note: pos_encoding is a fixed sinusoidal encoding and is NOT included
    /// in this count as it is not trainable.
    pub fn num_parameters(&self) -> usize {
        let input_params = self.input_proj.len();
        // Note: pos_encoding is fixed (sinusoidal), not trainable - excluded from count
        let d_model = self.config.d_model;
        let d_ff = d_model * self.config.ffn_multiplier;

        let layer_params = self.layers.len()
            * (d_model * d_model * 4 // attention QKVO weights
            + d_model * d_ff + d_ff * d_model // FFN weights (w1 and w2)
            + d_ff + d_model // FFN biases (ffn_b1 and ffn_b2)
            + d_model * 4); // layer norm (ln1_gamma, ln1_beta, ln2_gamma, ln2_beta)

        let output_params = self.output_proj_reg.len()
            + self.output_proj_cls.len()
            + self.output_proj_alloc.len();

        input_params + layer_params + output_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> NystromformerConfig {
        NystromformerConfig {
            input_dim: 6,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            num_landmarks: 8,
            seq_len: 64,
            pred_horizon: 12,
            num_classes: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_encoder_layer() {
        let config = test_config();
        let layer = NystromEncoderLayer::new(&config);

        let x = Array3::from_shape_fn((2, 64, 32), |_| rand_normal());
        let (output, weights) = layer.forward(&x);

        assert_eq!(output.dim(), (2, 64, 32));
        assert!(weights.landmark_weights.is_some());
    }

    #[test]
    fn test_model_regression() {
        let mut config = test_config();
        config.output_type = OutputType::Regression;

        let model = NystromformerModel::new(config.clone());
        let x = Array3::from_shape_fn((2, 64, 6), |_| rand_normal());

        let (output, _) = model.forward(&x);

        assert_eq!(output.dim(), (2, config.pred_horizon));
    }

    #[test]
    fn test_model_classification() {
        let mut config = test_config();
        config.output_type = OutputType::Classification;

        let model = NystromformerModel::new(config.clone());
        let x = Array3::from_shape_fn((2, 64, 6), |_| rand_normal());

        let (output, _) = model.forward(&x);

        assert_eq!(output.dim(), (2, config.num_classes));

        // Check softmax sums to 1
        for b in 0..2 {
            let sum: f64 = (0..config.num_classes).map(|c| output[[b, c]]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");
        }
    }

    #[test]
    fn test_model_allocation() {
        let mut config = test_config();
        config.output_type = OutputType::Allocation;

        let model = NystromformerModel::new(config);
        let x = Array3::from_shape_fn((2, 64, 6), |_| rand_normal());

        let (output, _) = model.forward(&x);

        assert_eq!(output.dim(), (2, 1));

        // Check sigmoid output is in [0, 1]
        for b in 0..2 {
            assert!(output[[b, 0]] >= 0.0 && output[[b, 0]] <= 1.0);
        }
    }

    #[test]
    fn test_positional_encoding() {
        let pe = NystromformerModel::create_positional_encoding(100, 32);

        assert_eq!(pe.dim(), (100, 32));

        // Check that positions have different encodings
        let diff: f64 = (0..32).map(|d| (pe[[0, d]] - pe[[50, d]]).abs()).sum();
        assert!(diff > 0.1, "Different positions should have different encodings");
    }

    #[test]
    fn test_num_parameters() {
        let config = test_config();
        let model = NystromformerModel::new(config);

        let num_params = model.num_parameters();
        assert!(num_params > 0);
    }
}
