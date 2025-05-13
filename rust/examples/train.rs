//! Example: Train a Nyströmformer model with synthetic data

use nystromformer_trading::{
    NystromformerConfig, NystromformerModel, OutputType, SequenceLoader,
};

fn main() {
    env_logger::init();

    println!("=== Nyströmformer Trading: Training Example ===\n");

    // Configuration
    let seq_len = 256;
    let num_landmarks = 32;
    let num_features = 6;
    let horizon = 24;
    let batch_size = 16;

    // Create model
    println!("Creating Nyströmformer model...");
    let config = NystromformerConfig {
        input_dim: num_features,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        num_landmarks,
        seq_len,
        pred_horizon: horizon,
        output_type: OutputType::Regression,
        ..Default::default()
    };

    let model = NystromformerModel::new(config.clone());
    println!("Model parameters: ~{}", model.num_parameters());

    // Generate synthetic data
    println!("\nGenerating synthetic training data...");
    let loader = SequenceLoader::new();
    let dataset = loader.generate_synthetic(500, seq_len, num_features, horizon);

    // Split data
    let (train, val, test) = dataset.split(0.7, 0.15);
    println!("Train samples: {}", train.len());
    println!("Validation samples: {}", val.len());
    println!("Test samples: {}", test.len());

    // Simulate training loop
    println!("\nSimulating training (forward passes only)...");
    let num_epochs = 3;
    // Use ceiling division to handle partial batches
    let num_batches = (train.len() + batch_size - 1) / batch_size;

    if num_batches == 0 {
        println!("Not enough training samples for one batch.");
        return;
    }

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut valid_batches = 0;

        for batch_idx in 0..num_batches {
            // Get batch indices, handling partial last batch
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train.len());
            let indices: Vec<usize> = (start..end).collect();

            if indices.is_empty() {
                continue;
            }

            // Get batch
            let (x_batch, y_batch) = train.get_batch(&indices);

            // Forward pass
            let (predictions, _attention_weights) = model.forward(&x_batch);

            // Calculate MSE loss
            let mut batch_loss = 0.0;
            let actual_batch_size = indices.len();
            for b in 0..actual_batch_size {
                for h in 0..horizon {
                    let diff = predictions[[b, h]] - y_batch[[b, h]];
                    batch_loss += diff * diff;
                }
            }
            batch_loss /= (actual_batch_size * horizon) as f64;
            epoch_loss += batch_loss;
            valid_batches += 1;
        }

        if valid_batches > 0 {
            epoch_loss /= valid_batches as f64;
            println!("Epoch {}/{}: Loss = {:.6}", epoch + 1, num_epochs, epoch_loss);
        }
    }

    // Validation
    println!("\nValidating model...");
    if val.is_empty() {
        println!("Skipping validation: no validation samples.");
    } else {
        let val_indices: Vec<usize> = (0..val.len().min(batch_size)).collect();
        let (x_val, y_val) = val.get_batch(&val_indices);
        let (predictions, attention) = model.forward(&x_val);

        let mut val_loss = 0.0;
        for b in 0..val_indices.len() {
            for h in 0..horizon {
                let diff = predictions[[b, h]] - y_val[[b, h]];
                val_loss += diff * diff;
            }
        }
        // Guard against division by zero
        let divisor = val_indices.len() * horizon;
        if divisor > 0 {
            val_loss /= divisor as f64;
            println!("Validation Loss: {:.6}", val_loss);
        }

        // Show attention statistics
        if let Some(ref lw) = attention.landmark_weights {
            let mean_attention: f64 = if lw.is_empty() {
                0.0
            } else {
                lw.iter().sum::<f64>() / lw.len() as f64
            };
            println!("\nAttention Statistics:");
            println!("  Mean landmark attention: {:.6}", mean_attention);

            let top_k = attention.top_k_landmarks(3);
            println!("  Top 3 landmark connections:");
            for (i, j, weight) in top_k {
                println!("    Landmark {} -> {}: {:.4}", i, j, weight);
            }
        }

        // Test prediction
        println!("\nSample predictions vs targets (first 5 samples, first horizon step):");
        for b in 0..5.min(predictions.dim().0) {
            println!(
                "  Sample {}: Predicted={:+.4}, Actual={:+.4}",
                b,
                predictions[[b, 0]],
                y_val[[b, 0]]
            );
        }
    }

    println!("\n=== Training Example Complete ===");
}
