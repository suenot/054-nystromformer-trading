//! Example: Make predictions with a trained Nyströmformer model

use nystromformer_trading::{
    NystromformerConfig, NystromformerModel, OutputType, SequenceLoader, SignalGenerator,
};

fn main() {
    env_logger::init();

    println!("=== Nyströmformer Trading: Prediction Example ===\n");

    // Configuration
    let seq_len = 256;
    let num_landmarks = 32;
    let num_features = 6;
    let horizon = 24;

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

    let model = NystromformerModel::new(config);

    // Generate synthetic data
    println!("Generating synthetic data for prediction...");
    let loader = SequenceLoader::new();
    let dataset = loader.generate_synthetic(10, seq_len, num_features, horizon);

    // Make predictions
    println!("\nMaking predictions...");
    let (predictions, attention_weights) = model.forward(&dataset.x);

    println!("Predictions shape: {:?}", predictions.dim());

    // Generate trading signals
    let signal_generator = SignalGenerator::new(0.005, -0.005);
    let signals = signal_generator.generate(&predictions);

    println!("\n=== Trading Signals ===");
    for (i, signal) in signals.iter().enumerate() {
        let predicted_return: f64 = (0..horizon).map(|h| predictions[[i, h]]).sum();
        let price = dataset.prices[i];

        match signal {
            nystromformer_trading::TradingSignal::Buy(size) => {
                println!(
                    "Sample {}: BUY  (size={:.2}) | Predicted return: {:+.4}% | Price: {:.2}",
                    i, size, predicted_return * 100.0, price
                );
            }
            nystromformer_trading::TradingSignal::Sell(size) => {
                println!(
                    "Sample {}: SELL (size={:.2}) | Predicted return: {:+.4}% | Price: {:.2}",
                    i, size, predicted_return * 100.0, price
                );
            }
            nystromformer_trading::TradingSignal::Hold => {
                println!(
                    "Sample {}: HOLD            | Predicted return: {:+.4}% | Price: {:.2}",
                    i, predicted_return * 100.0, price
                );
            }
        }
    }

    // Analyze attention patterns
    println!("\n=== Attention Analysis ===");
    if let Some(ref lw) = attention_weights.landmark_weights {
        let (batch, heads, landmarks, _) = lw.dim();
        println!(
            "Landmark attention shape: [{}, {}, {}, {}]",
            batch, heads, landmarks, landmarks
        );

        // Find most important landmark connections
        println!("\nTop 5 most important landmark connections:");
        let top_connections = attention_weights.top_k_landmarks(5);
        for (rank, (from, to, weight)) in top_connections.iter().enumerate() {
            println!(
                "  {}. Landmark {} -> Landmark {}: attention = {:.4}",
                rank + 1,
                from,
                to,
                weight
            );
        }

        // Analyze per-head attention
        println!("\nPer-head attention statistics (first sample):");
        for h in 0..heads {
            let mut head_sum = 0.0;
            let mut head_max = 0.0_f64;
            for i in 0..landmarks {
                for j in 0..landmarks {
                    let val = lw[[0, h, i, j]];
                    head_sum += val;
                    head_max = head_max.max(val);
                }
            }
            println!(
                "  Head {}: mean={:.4}, max={:.4}",
                h,
                head_sum / (landmarks * landmarks) as f64,
                head_max
            );
        }
    }

    // Show prediction horizon profile
    println!("\n=== Prediction Horizon Profile (Sample 0) ===");
    println!("Hour\tPredicted Return");
    for h in 0..horizon {
        let bar_len = ((predictions[[0, h]] * 1000.0).abs() as usize).min(20);
        let bar = if predictions[[0, h]] >= 0.0 {
            "+".repeat(bar_len)
        } else {
            "-".repeat(bar_len)
        };
        println!("h+{:02}\t{:+.4}%\t{}", h + 1, predictions[[0, h]] * 100.0, bar);
    }

    println!("\n=== Prediction Example Complete ===");
}
