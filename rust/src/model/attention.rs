//! Nyström Attention Mechanism
//!
//! Implements the Nyström approximation for efficient O(n) attention
//! instead of the standard O(n²) complexity.
//!
//! ## Key Formula
//!
//! The Nyström approximation computes:
//! ```text
//! Ŝ ≈ F̃ · Ã⁺ · B̃
//! ```
//! Where:
//! - F̃ = softmax(Q·K̃ᵀ/√d) [n × m] - queries to landmarks
//! - Ã = softmax(Q̃·K̃ᵀ/√d) [m × m] - landmarks to landmarks
//! - B̃ = softmax(Q̃·Kᵀ/√d) [m × n] - landmarks to keys
//! - Ã⁺ is the Moore-Penrose pseudoinverse computed via Newton-Schulz iteration

use ndarray::{Array2, Array3, Array4, Axis, s};
use std::cmp::Ordering;

use crate::model::config::NystromformerConfig;

/// Attention weights for interpretability
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Landmark attention matrix [batch, num_heads, num_landmarks, num_landmarks]
    pub landmark_weights: Option<Array4<f64>>,
    /// Full approximated attention (optional, expensive to store)
    pub full_weights: Option<Array4<f64>>,
    /// Landmark positions used
    pub landmark_positions: Option<Vec<usize>>,
}

impl AttentionWeights {
    pub fn new() -> Self {
        Self {
            landmark_weights: None,
            full_weights: None,
            landmark_positions: None,
        }
    }

    /// Gets the top-k most important landmark connections
    pub fn top_k_landmarks(&self, k: usize) -> Vec<(usize, usize, f64)> {
        let mut results = Vec::new();

        if let Some(ref weights) = self.landmark_weights {
            let (_, _, num_landmarks, _) = weights.dim();

            // Average over batch and heads
            let mean_weights = weights
                .mean_axis(Axis(0))
                .unwrap()
                .mean_axis(Axis(0))
                .unwrap();

            for i in 0..num_landmarks {
                for j in 0..num_landmarks {
                    if i != j {
                        results.push((i, j, mean_weights[[i, j]]));
                    }
                }
            }

            results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            results.truncate(k);
        }

        results
    }
}

impl Default for AttentionWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Nyström Attention
///
/// Approximates self-attention using the Nyström method with O(n) complexity.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NystromAttention {
    /// Query projection [d_model, d_model]
    w_q: Array2<f64>,
    /// Key projection [d_model, d_model]
    w_k: Array2<f64>,
    /// Value projection [d_model, d_model]
    w_v: Array2<f64>,
    /// Output projection [d_model, d_model]
    w_o: Array2<f64>,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Number of landmarks
    num_landmarks: usize,
    /// Sequence length
    seq_len: usize,
    /// Segment size for landmark computation
    segment_size: usize,
    /// Scaling factor 1/sqrt(d_k)
    scale: f64,
    /// Number of Newton-Schulz iterations
    pinv_iterations: usize,
    /// Epsilon for numerical stability
    epsilon: f64,
}

impl NystromAttention {
    /// Creates a new Nyström Attention layer
    ///
    /// # Panics
    /// Panics if the config is invalid (e.g., num_landmarks == 0, seq_len % num_landmarks != 0)
    pub fn new(config: &NystromformerConfig) -> Self {
        // Validate config to ensure segment_size computation won't divide by zero
        config
            .validate()
            .expect("Invalid NystromformerConfig for NystromAttention");

        let d_model = config.d_model;
        let num_heads = config.n_heads;
        let head_dim = config.head_dim();

        // Xavier initialization
        let scale_init = (2.0 / (d_model * 2) as f64).sqrt();

        let w_q = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_k = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_v = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);
        let w_o = Array2::from_shape_fn((d_model, d_model), |_| rand_normal() * scale_init);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            head_dim,
            num_landmarks: config.num_landmarks,
            seq_len: config.seq_len,
            segment_size: config.segment_size(),
            scale: (head_dim as f64).sqrt(),
            pinv_iterations: config.pinv_iterations,
            epsilon: config.epsilon,
        }
    }

    /// Forward pass of Nyström Attention
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    ///
    /// # Returns
    /// * `output` - [batch, seq_len, d_model]
    /// * `weights` - Attention weights for interpretability
    pub fn forward(&self, x: &Array3<f64>) -> (Array3<f64>, AttentionWeights) {
        let (batch_size, seq_len, d_model) = x.dim();

        // Linear projections
        let q = self.linear_transform(x, &self.w_q);
        let k = self.linear_transform(x, &self.w_k);
        let v = self.linear_transform(x, &self.w_v);

        // Compute landmarks using segment-means
        let q_landmarks = self.compute_landmarks(&q);
        let k_landmarks = self.compute_landmarks(&k);

        let mut output = Array3::zeros((batch_size, seq_len, d_model));
        let mut landmark_weights = Array4::zeros((
            batch_size,
            self.num_heads,
            self.num_landmarks,
            self.num_landmarks,
        ));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                let h_start = h * self.head_dim;
                let h_end = (h + 1) * self.head_dim;

                // Extract head slices
                let q_h = q.slice(s![b, .., h_start..h_end]).to_owned();
                let k_h = k.slice(s![b, .., h_start..h_end]).to_owned();
                let v_h = v.slice(s![b, .., h_start..h_end]).to_owned();

                let q_l_h = q_landmarks.slice(s![b, .., h_start..h_end]).to_owned();
                let k_l_h = k_landmarks.slice(s![b, .., h_start..h_end]).to_owned();

                // Compute Nyström attention components
                // F̃ = softmax(Q @ K̃ᵀ / sqrt(d)) [n × m]
                let f_tilde = self.compute_attention_scores(&q_h, &k_l_h);

                // Ã = softmax(Q̃ @ K̃ᵀ / sqrt(d)) [m × m]
                let a_tilde = self.compute_attention_scores(&q_l_h, &k_l_h);

                // B̃ = softmax(Q̃ @ Kᵀ / sqrt(d)) [m × n]
                let b_tilde = self.compute_attention_scores(&q_l_h, &k_h);

                // Store landmark-to-landmark weights for interpretability
                for i in 0..self.num_landmarks {
                    for j in 0..self.num_landmarks {
                        landmark_weights[[b, h, i, j]] = a_tilde[[i, j]];
                    }
                }

                // Compute pseudoinverse of Ã using Newton-Schulz iteration
                let a_pinv = self.iterative_pinv(&a_tilde);

                // Compute attention output: Ŝ @ V = F̃ @ Ã⁺ @ B̃ @ V
                // First: B̃ @ V [m × d]
                let bv = self.matmul_2d(&b_tilde, &v_h);

                // Second: Ã⁺ @ (B̃ @ V) [m × d]
                let abv = self.matmul_2d(&a_pinv, &bv);

                // Third: F̃ @ (Ã⁺ @ B̃ @ V) [n × d]
                let out_h = self.matmul_2d(&f_tilde, &abv);

                // Write output
                for t in 0..seq_len {
                    for d in 0..self.head_dim {
                        output[[b, t, h_start + d]] = out_h[[t, d]];
                    }
                }
            }
        }

        // Output projection
        let projected = self.linear_transform(&output, &self.w_o);

        let weights = AttentionWeights {
            landmark_weights: Some(landmark_weights),
            full_weights: None,
            landmark_positions: None,
        };

        (projected, weights)
    }

    /// Computes landmarks using segment-means
    ///
    /// Divides the sequence into segments and takes the mean of each segment
    fn compute_landmarks(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut landmarks = Array3::zeros((batch_size, self.num_landmarks, d_model));

        for b in 0..batch_size {
            for l in 0..self.num_landmarks {
                let start = l * self.segment_size;
                let end = (start + self.segment_size).min(seq_len);

                // Compute mean over the segment
                for d in 0..d_model {
                    let mut sum = 0.0;
                    for t in start..end {
                        sum += x[[b, t, d]];
                    }
                    landmarks[[b, l, d]] = sum / (end - start) as f64;
                }
            }
        }

        landmarks
    }

    /// Computes attention scores with softmax
    fn compute_attention_scores(&self, q: &Array2<f64>, k: &Array2<f64>) -> Array2<f64> {
        let n = q.dim().0;
        let m = k.dim().0;

        let mut scores = Array2::zeros((n, m));

        // Q @ K^T / sqrt(d)
        for i in 0..n {
            for j in 0..m {
                let mut score = 0.0;
                for d in 0..q.dim().1 {
                    score += q[[i, d]] * k[[j, d]];
                }
                scores[[i, j]] = score / self.scale;
            }
        }

        // Softmax over last dimension
        for i in 0..n {
            let max_val = (0..m)
                .map(|j| scores[[i, j]])
                .fold(f64::NEG_INFINITY, f64::max);

            let exp_sum: f64 = (0..m).map(|j| (scores[[i, j]] - max_val).exp()).sum();

            for j in 0..m {
                scores[[i, j]] = (scores[[i, j]] - max_val).exp() / (exp_sum + self.epsilon);
            }
        }

        scores
    }

    /// Iterative pseudoinverse using Newton-Schulz iteration
    ///
    /// Computes A⁺ ≈ Z_n where:
    /// Z_0 = αAᵀ (α = 1/||A||²)
    /// Z_{n+1} = Z_n @ (2I - A @ Z_n)
    fn iterative_pinv(&self, a: &Array2<f64>) -> Array2<f64> {
        let m = a.dim().0;

        // Compute ||A||_F^2 for initial scaling
        let norm_sq: f64 = a.iter().map(|&x| x * x).sum();
        let alpha = 1.0 / (norm_sq + self.epsilon);

        // Z_0 = α * A^T
        let mut z = a.t().to_owned() * alpha;

        // Identity matrix
        let eye = Array2::from_shape_fn((m, m), |(i, j)| if i == j { 1.0 } else { 0.0 });

        // Newton-Schulz iterations
        for _ in 0..self.pinv_iterations {
            // AZ
            let az = self.matmul_2d(a, &z);

            // 2I - AZ
            let two_i_minus_az =
                Array2::from_shape_fn((m, m), |(i, j)| 2.0 * eye[[i, j]] - az[[i, j]]);

            // Z = Z @ (2I - AZ)
            z = self.matmul_2d(&z, &two_i_minus_az);
        }

        z
    }

    /// 2D matrix multiplication
    fn matmul_2d(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let n = a.dim().0;
        let k = a.dim().1;
        let m = b.dim().1;

        let mut c = Array2::zeros((n, m));

        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                c[[i, j]] = sum;
            }
        }

        c
    }

    /// Linear transformation for 3D tensor
    fn linear_transform(&self, x: &Array3<f64>, w: &Array2<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_in) = x.dim();
        let d_out = w.dim().1;
        let mut output = Array3::zeros((batch_size, seq_len, d_out));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for d_o in 0..d_out {
                    let mut sum = 0.0;
                    for d_i in 0..d_in {
                        sum += x[[b, t, d_i]] * w[[d_i, d_o]];
                    }
                    output[[b, t, d_o]] = sum;
                }
            }
        }

        output
    }
}

/// Generates a random number from standard normal distribution
fn rand_normal() -> f64 {
    use std::f64::consts::PI;
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> NystromformerConfig {
        NystromformerConfig {
            d_model: 32,
            n_heads: 4,
            num_landmarks: 8,
            seq_len: 64,
            pinv_iterations: 6,
            epsilon: 1e-6,
            ..Default::default()
        }
    }

    #[test]
    fn test_nystrom_attention_output_shape() {
        let config = test_config();
        let attn = NystromAttention::new(&config);

        // [batch=2, seq_len=64, d_model=32]
        let x = Array3::from_shape_fn((2, 64, 32), |_| rand_normal());

        let (output, weights) = attn.forward(&x);

        assert_eq!(output.dim(), (2, 64, 32));
        assert!(weights.landmark_weights.is_some());

        let lw = weights.landmark_weights.unwrap();
        assert_eq!(lw.dim(), (2, 4, 8, 8)); // [batch, heads, landmarks, landmarks]
    }

    #[test]
    fn test_landmark_computation() {
        let config = test_config();
        let attn = NystromAttention::new(&config);

        let x = Array3::from_shape_fn((1, 64, 32), |(_, t, _)| t as f64);

        let landmarks = attn.compute_landmarks(&x);

        assert_eq!(landmarks.dim(), (1, 8, 32)); // 64/8 = 8 landmarks
    }

    #[test]
    fn test_iterative_pinv() {
        let config = test_config();
        let attn = NystromAttention::new(&config);

        // Create a simple matrix
        let a = Array2::from_shape_fn((4, 4), |(i, j)| {
            if i == j {
                1.0 + rand_normal() * 0.1
            } else {
                rand_normal() * 0.1
            }
        });

        let a_pinv = attn.iterative_pinv(&a);

        // Check A @ A^+ @ A ≈ A (pseudoinverse property)
        let aa_pinv = attn.matmul_2d(&a, &a_pinv);
        let aa_pinv_a = attn.matmul_2d(&aa_pinv, &a);

        for i in 0..4 {
            for j in 0..4 {
                let diff = (aa_pinv_a[[i, j]] - a[[i, j]]).abs();
                assert!(diff < 0.5, "Pseudoinverse property violated: diff = {}", diff);
            }
        }
    }

    #[test]
    fn test_softmax_normalization() {
        let config = test_config();
        let attn = NystromAttention::new(&config);

        let q = Array2::from_shape_fn((16, 8), |_| rand_normal());
        let k = Array2::from_shape_fn((8, 8), |_| rand_normal());

        let scores = attn.compute_attention_scores(&q, &k);

        // Check rows sum to 1
        for i in 0..16 {
            let sum: f64 = (0..8).map(|j| scores[[i, j]]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Softmax sum should be 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_top_k_landmarks() {
        let mut weights = AttentionWeights::new();

        // Create test weights [batch=1, heads=1, landmarks=4, landmarks=4]
        let mut lw = Array4::zeros((1, 1, 4, 4));
        lw[[0, 0, 0, 1]] = 0.9;
        lw[[0, 0, 1, 2]] = 0.7;
        lw[[0, 0, 2, 3]] = 0.5;

        weights.landmark_weights = Some(lw);

        let top = weights.top_k_landmarks(2);

        assert_eq!(top.len(), 2);
        assert!(top[0].2 > top[1].2);
        assert_eq!(top[0], (0, 1, 0.9));
    }
}
