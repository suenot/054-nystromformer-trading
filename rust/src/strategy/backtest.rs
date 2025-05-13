//! Backtesting engine for Nyströmformer trading strategies

use ndarray::Array3;

use crate::model::NystromformerModel;
use crate::strategy::signals::{SignalGenerator, TradingSignal};

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (percentage)
    pub transaction_cost: f64,
    /// Slippage (percentage)
    pub slippage: f64,
    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,
    /// Whether to use stop loss
    pub use_stop_loss: bool,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Whether to use take profit
    pub use_take_profit: bool,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Signal buy threshold
    pub buy_threshold: f64,
    /// Signal sell threshold
    pub sell_threshold: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            max_position_size: 0.5,
            use_stop_loss: true,
            stop_loss_pct: 0.02,
            use_take_profit: false,
            take_profit_pct: 0.05,
            buy_threshold: 0.001,
            sell_threshold: -0.001,
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Returns
    pub returns: Vec<f64>,
    /// Positions over time
    pub positions: Vec<f64>,
    /// Number of trades
    pub num_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Profit factor
    pub profit_factor: f64,
}

impl BacktestResult {
    /// Prints a summary of the results
    pub fn print_summary(&self) {
        println!("\n=== Backtest Results ===");
        println!("Total Return:      {:>10.2}%", self.total_return * 100.0);
        println!("Annualized Return: {:>10.2}%", self.annualized_return * 100.0);
        println!("Sharpe Ratio:      {:>10.3}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:>10.3}", self.sortino_ratio);
        println!("Max Drawdown:      {:>10.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio:      {:>10.3}", self.calmar_ratio);
        println!("Number of Trades:  {:>10}", self.num_trades);
        println!("Win Rate:          {:>10.2}%", self.win_rate * 100.0);
        println!("Avg Trade Return:  {:>10.4}%", self.avg_trade_return * 100.0);
        println!("Profit Factor:     {:>10.3}", self.profit_factor);
        println!("========================\n");
    }
}

/// Nyströmformer backtester
pub struct NystromBacktester {
    /// Model for predictions
    model: NystromformerModel,
    /// Configuration
    config: BacktestConfig,
    /// Signal generator
    signal_generator: SignalGenerator,
}

impl NystromBacktester {
    /// Creates a new backtester
    pub fn new(model: NystromformerModel, config: BacktestConfig) -> Self {
        let signal_generator = SignalGenerator::new(config.buy_threshold, config.sell_threshold);

        Self {
            model,
            config,
            signal_generator,
        }
    }

    /// Runs backtest on test data
    ///
    /// # Arguments
    /// * `x` - Feature sequences [num_samples, seq_len, num_features]
    /// * `prices` - Price series corresponding to each sample
    pub fn run_backtest(&self, x: &Array3<f64>, prices: &[f64]) -> BacktestResult {
        let n_samples = x.dim().0;

        if n_samples == 0 || prices.len() < n_samples {
            return self.empty_result();
        }

        let mut equity = self.config.initial_capital;
        let mut equity_curve = vec![equity];
        let mut returns = Vec::new();
        let mut positions = vec![0.0];
        let mut current_position = 0.0;
        let mut entry_price = 0.0;
        let mut num_trades = 0;
        let mut wins = 0;
        let mut trade_returns = Vec::new();

        // Process each sample
        for i in 0..n_samples.saturating_sub(1) {
            let price = prices[i];
            let next_price = prices.get(i + 1).copied().unwrap_or(price);

            // Get model prediction
            let x_sample = x.slice(ndarray::s![i..i + 1, .., ..]).to_owned();
            let (predictions, _) = self.model.forward(&x_sample);

            // Generate signal
            let pred_slice: Vec<f64> = (0..predictions.dim().1)
                .map(|j| predictions[[0, j]])
                .collect();
            let signal = self.signal_generator.generate_single(&pred_slice);

            // Check stop loss / take profit
            if current_position != 0.0 {
                let pnl_pct = if current_position > 0.0 {
                    (price - entry_price) / entry_price
                } else {
                    (entry_price - price) / entry_price
                };

                // Stop loss
                if self.config.use_stop_loss && pnl_pct < -self.config.stop_loss_pct {
                    let trade_return = self.close_position(
                        &mut equity,
                        current_position,
                        price,
                        entry_price,
                    );
                    trade_returns.push(trade_return);
                    if trade_return > 0.0 {
                        wins += 1;
                    }
                    current_position = 0.0;
                    num_trades += 1;
                }

                // Take profit
                if self.config.use_take_profit && pnl_pct > self.config.take_profit_pct {
                    let trade_return = self.close_position(
                        &mut equity,
                        current_position,
                        price,
                        entry_price,
                    );
                    trade_returns.push(trade_return);
                    if trade_return > 0.0 {
                        wins += 1;
                    }
                    current_position = 0.0;
                    num_trades += 1;
                }
            }

            // Execute signal
            match signal {
                TradingSignal::Buy(size) if current_position <= 0.0 => {
                    // Close short if any
                    if current_position < 0.0 {
                        let trade_return = self.close_position(
                            &mut equity,
                            current_position,
                            price,
                            entry_price,
                        );
                        trade_returns.push(trade_return);
                        if trade_return > 0.0 {
                            wins += 1;
                        }
                        num_trades += 1;
                    }

                    // Open long
                    let position_size = (size * self.config.max_position_size).min(1.0);
                    current_position = position_size;
                    entry_price = price * (1.0 + self.config.slippage + self.config.transaction_cost);
                }
                TradingSignal::Sell(size) if current_position >= 0.0 => {
                    // Close long if any
                    if current_position > 0.0 {
                        let trade_return = self.close_position(
                            &mut equity,
                            current_position,
                            price,
                            entry_price,
                        );
                        trade_returns.push(trade_return);
                        if trade_return > 0.0 {
                            wins += 1;
                        }
                        num_trades += 1;
                    }

                    // Open short
                    let position_size = (size * self.config.max_position_size).min(1.0);
                    current_position = -position_size;
                    entry_price = price * (1.0 - self.config.slippage - self.config.transaction_cost);
                }
                TradingSignal::Hold => {
                    // Hold current position
                }
                _ => {
                    // Signal doesn't match position state
                }
            }

            // Update equity with mark-to-market
            // Note: current_position already includes max_position_size scaling (from position_size assignment)
            let prev_equity = equity;
            if current_position != 0.0 {
                let price_change = (next_price - price) / price;
                // Don't multiply by max_position_size again - it's already in current_position
                let position_pnl = current_position * price_change * equity;
                equity += position_pnl;
            }

            let period_return = (equity - prev_equity) / prev_equity;
            returns.push(period_return);
            equity_curve.push(equity);
            positions.push(current_position);
        }

        // Close final position
        if current_position != 0.0 && !prices.is_empty() {
            let final_price = *prices.last().unwrap();
            let trade_return = self.close_position(
                &mut equity,
                current_position,
                final_price,
                entry_price,
            );
            trade_returns.push(trade_return);
            if trade_return > 0.0 {
                wins += 1;
            }
            num_trades += 1;
        }

        // Calculate metrics
        self.calculate_metrics(
            equity_curve,
            returns,
            positions,
            num_trades,
            wins,
            trade_returns,
        )
    }

    /// Closes a position and returns the trade return
    ///
    /// Note: position already includes max_position_size scaling, so we don't multiply again
    fn close_position(
        &self,
        equity: &mut f64,
        position: f64,
        exit_price: f64,
        entry_price: f64,
    ) -> f64 {
        let gross_return = if position > 0.0 {
            (exit_price - entry_price) / entry_price
        } else {
            (entry_price - exit_price) / entry_price
        };

        let net_return = gross_return - self.config.transaction_cost - self.config.slippage;
        // position.abs() already includes max_position_size, don't multiply again
        let pnl = position.abs() * net_return * *equity;
        *equity += pnl;

        net_return
    }

    /// Calculates performance metrics
    fn calculate_metrics(
        &self,
        equity_curve: Vec<f64>,
        returns: Vec<f64>,
        positions: Vec<f64>,
        num_trades: usize,
        wins: usize,
        trade_returns: Vec<f64>,
    ) -> BacktestResult {
        let n = returns.len();

        // Guard against empty returns to avoid NaN/division by zero
        if n == 0 {
            return BacktestResult {
                equity_curve,
                returns,
                positions,
                num_trades,
                win_rate: 0.0,
                total_return: 0.0,
                annualized_return: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                calmar_ratio: 0.0,
                avg_trade_return: 0.0,
                profit_factor: 0.0,
            };
        }

        let periods_per_year = 252.0 * 24.0; // Assuming hourly data

        // Total and annualized return
        let total_return = (equity_curve.last().unwrap_or(&self.config.initial_capital)
            / self.config.initial_capital)
            - 1.0;
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n as f64) - 1.0;

        // Sharpe ratio
        let mean_return: f64 = returns.iter().sum::<f64>() / n as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_return = variance.sqrt();
        let sharpe_ratio = if std_return > 0.0 {
            mean_return / std_return * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance: f64 = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * periods_per_year.sqrt()
        } else {
            sharpe_ratio
        };

        // Maximum drawdown
        let mut peak = equity_curve[0];
        let mut max_drawdown = 0.0;
        for &equity in &equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Win rate
        let win_rate = if num_trades > 0 {
            wins as f64 / num_trades as f64
        } else {
            0.0
        };

        // Average trade return
        let avg_trade_return = if !trade_returns.is_empty() {
            trade_returns.iter().sum::<f64>() / trade_returns.len() as f64
        } else {
            0.0
        };

        // Profit factor
        let gross_profits: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_losses: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_losses > 0.0 {
            gross_profits / gross_losses
        } else if gross_profits > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResult {
            equity_curve,
            returns,
            positions,
            num_trades,
            win_rate,
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            avg_trade_return,
            profit_factor,
        }
    }

    /// Returns empty result
    fn empty_result(&self) -> BacktestResult {
        BacktestResult {
            equity_curve: vec![self.config.initial_capital],
            returns: vec![],
            positions: vec![0.0],
            num_trades: 0,
            win_rate: 0.0,
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            avg_trade_return: 0.0,
            profit_factor: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NystromformerConfig, OutputType};

    fn test_model() -> NystromformerModel {
        let config = NystromformerConfig {
            input_dim: 6,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            num_landmarks: 8,
            seq_len: 64,
            pred_horizon: 12,
            output_type: OutputType::Regression,
            ..Default::default()
        };
        NystromformerModel::new(config)
    }

    #[test]
    fn test_backtester_creation() {
        let model = test_model();
        let config = BacktestConfig::default();
        let _backtester = NystromBacktester::new(model, config);
    }

    #[test]
    fn test_backtest_empty() {
        let model = test_model();
        let config = BacktestConfig::default();
        let backtester = NystromBacktester::new(model, config.clone());

        let x = Array3::zeros((0, 64, 6));
        let prices: Vec<f64> = vec![];

        let result = backtester.run_backtest(&x, &prices);

        assert_eq!(result.num_trades, 0);
        assert_eq!(result.total_return, 0.0);
    }

    #[test]
    fn test_backtest_synthetic() {
        let model = test_model();
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            buy_threshold: 0.0,
            sell_threshold: 0.0,
            ..Default::default()
        };
        let backtester = NystromBacktester::new(model, config);

        // Generate synthetic data
        use crate::data::SequenceLoader;
        let loader = SequenceLoader::new();
        let dataset = loader.generate_synthetic(50, 64, 6, 12);

        let result = backtester.run_backtest(&dataset.x, &dataset.prices);

        // Should have some activity
        assert!(result.equity_curve.len() > 1);
    }

    #[test]
    fn test_metrics_calculation() {
        let model = test_model();
        let config = BacktestConfig::default();
        let backtester = NystromBacktester::new(model, config.clone());

        // Test metrics with known values
        let equity_curve = vec![100_000.0, 101_000.0, 102_000.0, 101_500.0, 103_000.0];
        let returns = vec![0.01, 0.0099, -0.0049, 0.0148];
        let positions = vec![0.0, 1.0, 1.0, 0.0, 1.0];
        let trade_returns = vec![0.02, -0.005, 0.015];

        let result = backtester.calculate_metrics(
            equity_curve,
            returns,
            positions,
            3,
            2,
            trade_returns,
        );

        assert!((result.total_return - 0.03).abs() < 0.001);
        assert!((result.win_rate - 0.6667).abs() < 0.01);
        assert!(result.sharpe_ratio > 0.0);
        assert!(result.max_drawdown > 0.0);
    }
}
