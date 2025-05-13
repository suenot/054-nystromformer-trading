//! Example: Backtest a Nyströmformer trading strategy

use nystromformer_trading::{
    BacktestConfig, NystromBacktester, NystromformerConfig, NystromformerModel,
    OutputType, SequenceLoader,
};

fn main() {
    env_logger::init();

    println!("=== Nyströmformer Trading: Backtest Example ===\n");

    // Configuration
    let seq_len = 128;
    let num_landmarks = 16;
    let num_features = 6;
    let horizon = 12;

    // Create model
    println!("Creating Nyströmformer model...");
    let model_config = NystromformerConfig {
        input_dim: num_features,
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        num_landmarks,
        seq_len,
        pred_horizon: horizon,
        output_type: OutputType::Regression,
        ..Default::default()
    };

    let model = NystromformerModel::new(model_config);

    // Generate synthetic data
    println!("Generating synthetic data...");
    let loader = SequenceLoader::new();
    let dataset = loader.generate_synthetic(200, seq_len, num_features, horizon);

    // Split data (use test set for backtesting)
    let (_train, _val, test) = dataset.split(0.7, 0.15);
    println!("Test samples for backtest: {}", test.len());

    // Backtest configuration
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost: 0.001, // 0.1% per trade
        slippage: 0.0005,        // 0.05% slippage
        max_position_size: 0.5,  // Max 50% of capital per position
        use_stop_loss: true,
        stop_loss_pct: 0.02,     // 2% stop loss
        use_take_profit: true,
        take_profit_pct: 0.05,   // 5% take profit
        buy_threshold: 0.002,    // 0.2% predicted return to buy
        sell_threshold: -0.002,  // -0.2% predicted return to sell
    };

    println!("\nBacktest Configuration:");
    println!("  Initial Capital: ${:.2}", backtest_config.initial_capital);
    println!("  Transaction Cost: {:.2}%", backtest_config.transaction_cost * 100.0);
    println!("  Slippage: {:.2}%", backtest_config.slippage * 100.0);
    println!("  Max Position: {:.0}%", backtest_config.max_position_size * 100.0);
    println!("  Stop Loss: {:.1}%", backtest_config.stop_loss_pct * 100.0);
    println!("  Take Profit: {:.1}%", backtest_config.take_profit_pct * 100.0);

    // Run backtest
    println!("\nRunning backtest...");
    let backtester = NystromBacktester::new(model, backtest_config);
    let result = backtester.run_backtest(&test.x, &test.prices);

    // Print results
    result.print_summary();

    // Additional analysis
    println!("=== Detailed Analysis ===\n");

    // Equity curve statistics
    if !result.equity_curve.is_empty() {
        let final_equity = result.equity_curve.last().unwrap();
        let min_equity = result
            .equity_curve
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_equity = result
            .equity_curve
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        println!("Equity Curve:");
        println!("  Starting: $100,000.00");
        println!("  Ending:   ${:.2}", final_equity);
        println!("  Minimum:  ${:.2}", min_equity);
        println!("  Maximum:  ${:.2}", max_equity);
    }

    // Returns distribution
    if !result.returns.is_empty() {
        let positive_returns = result.returns.iter().filter(|&&r| r > 0.0).count();
        let negative_returns = result.returns.iter().filter(|&&r| r < 0.0).count();
        let zero_returns = result.returns.iter().filter(|&&r| r == 0.0).count();

        println!("\nReturns Distribution:");
        println!(
            "  Positive periods: {} ({:.1}%)",
            positive_returns,
            positive_returns as f64 / result.returns.len() as f64 * 100.0
        );
        println!(
            "  Negative periods: {} ({:.1}%)",
            negative_returns,
            negative_returns as f64 / result.returns.len() as f64 * 100.0
        );
        println!(
            "  Zero periods: {} ({:.1}%)",
            zero_returns,
            zero_returns as f64 / result.returns.len() as f64 * 100.0
        );

        // Best and worst periods
        let best = result
            .returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let worst = result.returns.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("\n  Best period:  {:+.4}%", best * 100.0);
        println!("  Worst period: {:+.4}%", worst * 100.0);
    }

    // Position analysis
    if !result.positions.is_empty() {
        let long_periods = result.positions.iter().filter(|&&p| p > 0.0).count();
        let short_periods = result.positions.iter().filter(|&&p| p < 0.0).count();
        let flat_periods = result.positions.iter().filter(|&&p| p == 0.0).count();

        println!("\nPosition Analysis:");
        println!(
            "  Long:  {} periods ({:.1}%)",
            long_periods,
            long_periods as f64 / result.positions.len() as f64 * 100.0
        );
        println!(
            "  Short: {} periods ({:.1}%)",
            short_periods,
            short_periods as f64 / result.positions.len() as f64 * 100.0
        );
        println!(
            "  Flat:  {} periods ({:.1}%)",
            flat_periods,
            flat_periods as f64 / result.positions.len() as f64 * 100.0
        );
    }

    // Draw simple equity curve
    println!("\n=== Equity Curve (ASCII) ===");
    if result.equity_curve.len() > 1 {
        let min_eq = result
            .equity_curve
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_eq = result
            .equity_curve
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_eq - min_eq;

        let width = 60;
        let height = 10;
        let step = result.equity_curve.len().max(1) / width;

        for row in 0..height {
            let threshold = max_eq - (row as f64 / height as f64) * range;
            let mut line = String::new();

            for col in 0..width {
                let idx = col * step;
                if idx < result.equity_curve.len() {
                    let eq = result.equity_curve[idx];
                    if eq >= threshold {
                        line.push('*');
                    } else {
                        line.push(' ');
                    }
                }
            }

            if row == 0 {
                println!("{:>10.0} |{}", max_eq, line);
            } else if row == height - 1 {
                println!("{:>10.0} |{}", min_eq, line);
            } else {
                println!("{:>10} |{}", "", line);
            }
        }
        println!("{:>10} +{}", "", "-".repeat(width));
        println!("{:>10} {:>width$}", "", "Time →");
    }

    println!("\n=== Backtest Example Complete ===");
}
