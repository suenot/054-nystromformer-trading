"""
Example Usage of Nyströmformer for Trading

This script demonstrates how to:
1. Load market data from Bybit
2. Prepare long sequence data for training
3. Train a Nyströmformer model
4. Backtest the trading strategy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from model import NystromformerTrading
from data import load_bybit_data, calculate_features, create_dataloaders
from strategy import NystromBacktester, BacktestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 1000,
    seq_len: int = 512,
    n_features: int = 6,
    horizon: int = 24
):
    """
    Generate synthetic data for testing.

    This simulates realistic-looking financial data for demonstration
    when API data is not available.
    """
    logger.info("Generating synthetic data for demonstration...")

    # Generate price series with realistic properties
    returns = np.random.randn(n_samples + seq_len + horizon) * 0.02
    # Add some autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]

    prices = 100 * np.exp(np.cumsum(returns))

    # Generate features
    X = np.zeros((n_samples, seq_len, n_features))

    for i in range(n_samples):
        start_idx = i
        end_idx = i + seq_len

        # Feature 1: Log returns
        price_window = prices[start_idx:end_idx+1]
        X[i, :, 0] = np.diff(np.log(price_window))

        # Feature 2: Volatility (rolling std of returns)
        vol = np.zeros(seq_len)
        for j in range(20, seq_len):
            vol[j] = X[i, j-20:j, 0].std()
        X[i, :, 1] = vol

        # Feature 3: Volume ratio (simulated)
        X[i, :, 2] = np.random.lognormal(0, 0.5, seq_len)

        # Feature 4: Price MA ratio
        ma = np.convolve(price_window[:-1], np.ones(20)/20, mode='same')
        X[i, :, 3] = price_window[:-1] / ma - 1

        # Feature 5: RSI-like feature
        X[i, :, 4] = np.tanh(X[i, :, 0] * 50)

        # Feature 6: Momentum
        X[i, :, 5] = np.roll(X[i, :, 0], -10).cumsum() - X[i, :, 0].cumsum()

    # Generate targets (future returns)
    y = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        start_idx = i + seq_len
        y[i] = returns[start_idx:start_idx + horizon]

    # Replace NaN with 0
    X = np.nan_to_num(X, 0)
    y = np.nan_to_num(y, 0)

    # Get corresponding prices for backtesting
    backtest_prices = prices[seq_len:seq_len + n_samples]

    return X.astype(np.float32), y.astype(np.float32), backtest_prices


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cpu'
):
    """
    Train the Nyströmformer model.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions, _ = model(batch_x)
            loss = model.compute_loss(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions, _ = model(batch_x)
                loss = model.compute_loss(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

    return model


def main():
    """Main example function."""
    print("="*60)
    print("Nyströmformer Trading Example")
    print("="*60)

    # Configuration
    seq_len = 512  # Use shorter sequence for example
    num_landmarks = 32
    n_features = 6
    horizon = 24
    batch_size = 16
    num_epochs = 30

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Step 1: Generate or load data
    print("\n1. Preparing data...")
    X, y, prices = generate_synthetic_data(
        n_samples=1000,
        seq_len=seq_len,
        n_features=n_features,
        horizon=horizon
    )

    logger.info(f"Data shapes: X={X.shape}, y={y.shape}, prices={prices.shape}")

    # Step 2: Create data loaders
    print("\n2. Creating data loaders...")

    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    test_prices = prices[train_size+val_size:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False
    )

    # Step 3: Create model
    print("\n3. Creating Nyströmformer model...")

    model = NystromformerTrading(
        input_dim=n_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_landmarks=num_landmarks,
        seq_len=seq_len,
        output_type='regression',
        pred_horizon=horizon,
        dropout=0.1
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Step 4: Train model
    print("\n4. Training model...")

    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device
    )

    # Step 5: Backtest
    print("\n5. Running backtest...")

    config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005,
        max_position_size=0.5,
        use_stop_loss=True,
        stop_loss_pct=0.02
    )

    backtester = NystromBacktester(model.to('cpu'), config)
    results = backtester.run_backtest(
        X_test,
        test_prices,
        signal_threshold=0.001
    )

    # Print results
    backtester.print_metrics(results)

    # Optional: Plot results
    try:
        backtester.plot_results(results, save_path='backtest_results.png', show=False)
        logger.info("Saved backtest plot to backtest_results.png")
    except Exception as e:
        logger.warning(f"Could not plot results: {e}")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
