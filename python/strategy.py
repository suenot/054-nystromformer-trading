"""
Trading Strategy and Backtesting for Nyströmformer

This module provides backtesting utilities for evaluating Nyströmformer
trading strategies with realistic transaction costs and slippage.

Main components:
- BacktestConfig: Configuration for backtesting parameters
- NystromBacktester: Backtesting engine
- Signal generation utilities
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    risk_per_trade: float = 0.02  # 2% risk per trade
    min_trade_interval: int = 1  # Minimum bars between trades
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02  # 2% stop loss
    use_take_profit: bool = True
    take_profit_pct: float = 0.04  # 4% take profit


def generate_signals(
    predictions: np.ndarray,
    threshold: float = 0.001,
    signal_type: str = 'binary'
) -> np.ndarray:
    """
    Generate trading signals from model predictions.

    Args:
        predictions: Model predictions (returns or allocations)
        threshold: Minimum predicted return to trigger signal
        signal_type: Type of signals to generate
            - 'binary': {-1, 0, 1} signals
            - 'continuous': Raw prediction values
            - 'quantile': Signals based on quantile thresholds

    Returns:
        signals: Array of trading signals
    """
    if signal_type == 'continuous':
        return predictions

    elif signal_type == 'binary':
        # Use first prediction step
        pred = predictions[:, 0] if len(predictions.shape) > 1 else predictions

        signals = np.zeros_like(pred)
        signals[pred > threshold] = 1   # Long signal
        signals[pred < -threshold] = -1  # Short signal

        return signals

    elif signal_type == 'quantile':
        pred = predictions[:, 0] if len(predictions.shape) > 1 else predictions

        # Use rolling quantiles
        upper_q = np.percentile(pred, 75)
        lower_q = np.percentile(pred, 25)

        signals = np.zeros_like(pred)
        signals[pred > upper_q] = 1
        signals[pred < lower_q] = -1

        return signals

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


class NystromBacktester:
    """
    Backtesting engine for Nyströmformer trading strategy.

    Simulates trading with realistic transaction costs, slippage,
    and position management.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: BacktestConfig = None
    ):
        """
        Initialize backtester.

        Args:
            model: Trained Nyströmformer model
            config: Backtesting configuration
        """
        self.model = model
        self.config = config or BacktestConfig()
        self.model.eval()

    @torch.no_grad()
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Generate predictions from model.

        Args:
            data: Input features [n_samples, seq_len, n_features]

        Returns:
            predictions: Model predictions
        """
        x = torch.tensor(data, dtype=torch.float32)

        # Handle device
        device = next(self.model.parameters()).device
        x = x.to(device)

        predictions, _ = self.model(x)

        return predictions.cpu().numpy()

    def run_backtest(
        self,
        data: np.ndarray,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        signal_threshold: float = 0.001
    ) -> Dict:
        """
        Run full backtest simulation.

        Args:
            data: Feature data [n_samples, seq_len, n_features]
            prices: Price series aligned with predictions
            timestamps: Optional timestamps for the backtest period
            signal_threshold: Threshold for generating signals

        Returns:
            Dictionary with backtest results
        """
        if timestamps is None:
            timestamps = np.arange(len(prices))

        # Generate predictions and signals
        predictions = self.predict(data)
        signals = generate_signals(predictions, threshold=signal_threshold)

        # Initialize tracking
        capital = self.config.initial_capital
        position = 0.0
        position_entry_price = 0.0
        last_trade_idx = -self.config.min_trade_interval

        # Results tracking
        equity_curve = [capital]
        positions = [0.0]
        returns = []
        trades = []

        for i in range(len(signals)):
            current_price = prices[i]
            signal = signals[i]

            # Check stop loss / take profit if we have a position
            if position != 0 and i > 0:
                price_change = (current_price - position_entry_price) / position_entry_price

                # Stop loss check
                if self.config.use_stop_loss:
                    if position > 0 and price_change < -self.config.stop_loss_pct:
                        signal = 0  # Close long
                    elif position < 0 and price_change > self.config.stop_loss_pct:
                        signal = 0  # Close short

                # Take profit check
                if self.config.use_take_profit:
                    if position > 0 and price_change > self.config.take_profit_pct:
                        signal = 0  # Close long
                    elif position < 0 and price_change < -self.config.take_profit_pct:
                        signal = 0  # Close short

            # Calculate target position
            target_position = signal * self.config.max_position_size
            position_change = target_position - position

            # Check minimum trade interval
            can_trade = (i - last_trade_idx) >= self.config.min_trade_interval

            # Execute trade if position changes and we can trade
            if abs(position_change) > 0.01 and can_trade:
                # Calculate trade costs
                trade_value = abs(position_change) * capital
                costs = trade_value * (
                    self.config.transaction_cost + self.config.slippage
                )

                # Execute trade
                capital -= costs
                last_trade_idx = i

                # Update position tracking
                if target_position != 0 and position == 0:
                    position_entry_price = current_price

                trades.append({
                    'timestamp': timestamps[i],
                    'price': current_price,
                    'signal': signal,
                    'position_from': position,
                    'position_to': target_position,
                    'costs': costs,
                    'capital': capital
                })

                position = target_position

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
        returns_array = np.array(returns)
        equity_array = np.array(equity_curve)
        metrics = self._calculate_metrics(returns_array, equity_array, trades)

        return {
            'equity_curve': equity_array,
            'positions': np.array(positions),
            'returns': returns_array,
            'trades': trades,
            'timestamps': timestamps,
            'metrics': metrics,
            'predictions': predictions,
            'signals': signals
        }

    def _calculate_metrics(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
        trades: List[Dict]
    ) -> Dict:
        """Calculate performance metrics."""

        # Filter out zero returns for calculations
        non_zero_returns = returns[returns != 0]

        # Basic metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Annualized return (assuming daily returns, 252 trading days)
        n_periods = len(returns)
        if n_periods > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        else:
            annualized_return = 0.0

        # Sharpe ratio
        if len(non_zero_returns) > 0 and non_zero_returns.std() > 0:
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
        drawdown = (equity_curve - peak) / (peak + 1e-8)
        max_drawdown = drawdown.min()

        # Win rate
        if trades:
            winning_trades = sum(
                1 for i in range(1, len(trades))
                if trades[i]['capital'] > trades[i-1]['capital']
            )
            win_rate = winning_trades / len(trades) if len(trades) > 1 else 0.0
        else:
            win_rate = 0.0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-8)

        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = float('inf') if annualized_return > 0 else 0.0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades),
            'final_capital': equity_curve[-1],
            'initial_capital': equity_curve[0]
        }

    def plot_results(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize backtest results.

        Args:
            results: Backtest results dictionary
            save_path: Optional path to save the figure
            show: Whether to display the figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        timestamps = results['timestamps']
        n_points = len(timestamps)

        # 1. Equity curve
        axes[0].plot(range(len(results['equity_curve'])), results['equity_curve'])
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Capital ($)')
        axes[0].grid(True, alpha=0.3)

        # 2. Positions
        axes[1].plot(range(len(results['positions'])), results['positions'])
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('Position Size')
        axes[1].set_ylabel('Position')
        axes[1].grid(True, alpha=0.3)

        # 3. Drawdown
        equity = results['equity_curve']
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-8)
        axes[2].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)

        # 4. Cumulative returns
        cum_returns = np.cumprod(1 + results['returns']) - 1
        axes[3].plot(range(len(cum_returns)), cum_returns * 100)
        axes[3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[3].set_title('Cumulative Returns')
        axes[3].set_ylabel('Return (%)')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        if show:
            plt.show()

        plt.close()

    def print_metrics(self, results: Dict):
        """Print formatted metrics."""
        metrics = results['metrics']

        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        print(f"\n{'Performance Metrics':^60}")
        print("-"*60)

        print(f"{'Initial Capital:':<30} ${metrics['initial_capital']:>15,.2f}")
        print(f"{'Final Capital:':<30} ${metrics['final_capital']:>15,.2f}")
        print(f"{'Total Return:':<30} {metrics['total_return']*100:>15.2f}%")
        print(f"{'Annualized Return:':<30} {metrics['annualized_return']*100:>15.2f}%")

        print(f"\n{'Risk Metrics':^60}")
        print("-"*60)

        print(f"{'Sharpe Ratio:':<30} {metrics['sharpe_ratio']:>15.2f}")
        print(f"{'Sortino Ratio:':<30} {metrics['sortino_ratio']:>15.2f}")
        print(f"{'Calmar Ratio:':<30} {metrics['calmar_ratio']:>15.2f}")
        print(f"{'Max Drawdown:':<30} {metrics['max_drawdown']*100:>15.2f}%")

        print(f"\n{'Trading Statistics':^60}")
        print("-"*60)

        print(f"{'Number of Trades:':<30} {metrics['num_trades']:>15d}")
        print(f"{'Win Rate:':<30} {metrics['win_rate']*100:>15.2f}%")
        print(f"{'Profit Factor:':<30} {metrics['profit_factor']:>15.2f}")

        print("="*60 + "\n")


class WalkForwardValidator:
    """
    Walk-forward validation for Nyströmformer trading strategy.

    Performs rolling out-of-sample backtests to evaluate model performance.
    """

    def __init__(
        self,
        model_factory,
        train_window: int = 5000,
        test_window: int = 500,
        step_size: int = 500
    ):
        """
        Initialize walk-forward validator.

        Args:
            model_factory: Function that creates and trains a model
            train_window: Number of samples for training
            test_window: Number of samples for testing
            step_size: Step size for rolling window
        """
        self.model_factory = model_factory
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray
    ) -> Dict:
        """
        Run walk-forward validation.

        Args:
            X: Feature data
            y: Target data
            prices: Price series

        Returns:
            Dictionary with validation results
        """
        results = []
        all_equity = []

        n_samples = len(X)
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= n_samples:
            # Define train/test split
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            # Train model
            X_train = X[start_idx:train_end]
            y_train = y[start_idx:train_end]

            model = self.model_factory(X_train, y_train)

            # Backtest on test period
            X_test = X[train_end:test_end]
            test_prices = prices[train_end:test_end]

            backtester = NystromBacktester(model)
            result = backtester.run_backtest(X_test, test_prices)

            results.append(result['metrics'])
            all_equity.extend(result['equity_curve'].tolist())

            logger.info(
                f"Window {len(results)}: "
                f"Return={result['metrics']['total_return']*100:.2f}%, "
                f"Sharpe={result['metrics']['sharpe_ratio']:.2f}"
            )

            start_idx += self.step_size

        # Aggregate results
        metrics_df = pd.DataFrame(results)

        aggregate_metrics = {
            'mean_return': metrics_df['total_return'].mean(),
            'std_return': metrics_df['total_return'].std(),
            'mean_sharpe': metrics_df['sharpe_ratio'].mean(),
            'std_sharpe': metrics_df['sharpe_ratio'].std(),
            'mean_max_dd': metrics_df['max_drawdown'].mean(),
            'num_windows': len(results)
        }

        return {
            'window_results': results,
            'aggregate_metrics': aggregate_metrics,
            'combined_equity': np.array(all_equity)
        }


if __name__ == "__main__":
    # Example usage with dummy data
    print("Testing strategy module...")

    # Create dummy model
    from model import NystromformerTrading

    model = NystromformerTrading(
        input_dim=6,
        d_model=64,
        n_heads=2,
        n_layers=1,
        num_landmarks=16,
        seq_len=256,
        output_type='regression',
        pred_horizon=24
    )

    # Create dummy data
    n_samples = 100
    seq_len = 256
    n_features = 6

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)

    # Run backtest
    config = BacktestConfig(
        initial_capital=10000,
        transaction_cost=0.001
    )

    backtester = NystromBacktester(model, config)
    results = backtester.run_backtest(X, prices)

    # Print results
    backtester.print_metrics(results)

    print("Test passed!")
