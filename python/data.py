"""
Data Loading and Preprocessing for Nyströmformer Trading

This module provides utilities for loading market data from various sources
(Bybit, Binance, Yahoo Finance) and preparing it for Nyströmformer training.

Main functions:
- load_bybit_data: Load OHLCV data from Bybit API
- prepare_long_sequence_data: Create sequences for training
- calculate_features: Engineer features for the model
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bybit_data(
    symbol: str,
    interval: str = '1m',
    limit: int = 10000,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Load historical OHLCV data from Bybit API.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d')
        limit: Maximum number of records to fetch
        start_time: Start time for data
        end_time: End time for data

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover
    """
    import requests

    # Map interval to Bybit format
    interval_map = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
        '1d': 'D', '1w': 'W', '1M': 'M'
    }

    bybit_interval = interval_map.get(interval, interval)

    url = "https://api.bybit.com/v5/market/kline"
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': bybit_interval,
        'limit': min(limit, 1000)  # API limit per request
    }

    if start_time:
        params['start'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['end'] = int(end_time.timestamp() * 1000)

    all_data = []

    try:
        # Fetch data in batches if needed
        while len(all_data) < limit:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            result = response.json()

            if result['retCode'] != 0:
                logger.error(f"Bybit API error: {result['retMsg']}")
                break

            data = result['result']['list']

            if not data:
                break

            all_data.extend(data)

            # Update end time for next batch
            if len(data) < params['limit']:
                break

            # Oldest timestamp from current batch
            oldest_ts = int(data[-1][0])
            params['end'] = oldest_ts - 1

            if len(all_data) >= limit:
                break

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        # Return empty DataFrame on error
        return pd.DataFrame()

    if not all_data:
        logger.warning(f"No data returned for {symbol}")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

    # Sort by time and reset index
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Limit to requested size
    df = df.tail(limit).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} records for {symbol}")

    return df


def calculate_features(df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
    """
    Calculate technical features from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        include_advanced: Whether to include advanced features

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Basic features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']

    # Volume features
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)

    # Volatility
    df['volatility_20'] = df['log_return'].rolling(20).std()
    df['volatility_50'] = df['log_return'].rolling(50).std()

    # Price relative to moving averages
    df['price_ma_20'] = df['close'].rolling(20).mean()
    df['price_ma_50'] = df['close'].rolling(50).mean()
    df['price_ma_200'] = df['close'].rolling(200).mean()

    df['price_ma_20_ratio'] = df['close'] / (df['price_ma_20'] + 1e-8)
    df['price_ma_50_ratio'] = df['close'] / (df['price_ma_50'] + 1e-8)
    df['price_ma_200_ratio'] = df['close'] / (df['price_ma_200'] + 1e-8)

    if include_advanced:
        # RSI
        df['rsi_14'] = calculate_rsi(df['close'], period=14)

        # ATR
        df['atr_14'] = calculate_atr(df, period=14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
            df['close'], period=20, std_dev=2
        )
        df['bb_position'] = (df['close'] - df['bb_lower']) / (
            df['bb_upper'] - df['bb_lower'] + 1e-8
        )

        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()

    return atr


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal

    return macd, signal, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return upper, middle, lower


def prepare_long_sequence_data(
    symbols: List[str],
    lookback: int = 4096,
    horizon: int = 24,
    source: str = 'bybit',
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare long sequence data for Nyströmformer training.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        lookback: Historical sequence length
        horizon: Prediction horizon
        source: Data source ('bybit', 'binance', 'yahoo')
        feature_cols: List of feature columns to use

    Returns:
        X: Features array [n_samples, lookback, n_features]
        y: Targets array [n_samples, horizon]
    """
    if feature_cols is None:
        feature_cols = [
            'log_return', 'volatility_20', 'volume_ratio',
            'price_ma_20_ratio', 'rsi_14', 'atr_14'
        ]

    all_features = []

    for symbol in symbols:
        logger.info(f"Loading data for {symbol}...")

        # Load data from source
        if source == 'bybit':
            df = load_bybit_data(symbol, interval='1m', limit=lookback + horizon + 500)
        else:
            raise NotImplementedError(f"Source {source} not implemented")

        if df.empty:
            logger.warning(f"No data for {symbol}, skipping...")
            continue

        # Calculate features
        df = calculate_features(df)

        all_features.append(df)

    if not all_features:
        raise ValueError("No data loaded for any symbol")

    # Combine features from all symbols
    # For single symbol, just use that data
    if len(all_features) == 1:
        features = all_features[0]
    else:
        # Align all dataframes on timestamp
        features = all_features[0][['timestamp'] + feature_cols].copy()
        for i, df in enumerate(all_features[1:], 1):
            suffix = f'_{symbols[i]}'
            df_subset = df[['timestamp'] + feature_cols].copy()
            df_subset.columns = ['timestamp'] + [f"{c}{suffix}" for c in feature_cols]
            features = features.merge(df_subset, on='timestamp', how='inner')

    # Drop NaN values
    features = features.dropna()

    if len(features) < lookback + horizon:
        raise ValueError(
            f"Not enough data: {len(features)} rows, need at least {lookback + horizon}"
        )

    # Get feature columns (exclude timestamp)
    all_feature_cols = [c for c in features.columns if c != 'timestamp']

    # Create sequences
    X, y = [], []
    feature_values = features[all_feature_cols].values

    # Target is the log_return of the first symbol
    target_col_idx = all_feature_cols.index('log_return')

    for i in range(lookback, len(features) - horizon):
        # Input: [lookback, n_features]
        x_seq = feature_values[i-lookback:i]
        X.append(x_seq)

        # Target: future returns
        y_seq = feature_values[i:i+horizon, target_col_idx]
        y.append(y_seq)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Created {len(X)} sequences with shape X={X.shape}, y={y.shape}")

    return X, y


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Dict:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        X: Features array
        y: Targets array
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        batch_size: Batch size for DataLoaders
        shuffle_train: Whether to shuffle training data

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)

    X_train = torch.tensor(X[:train_size])
    y_train = torch.tensor(y[:train_size])

    X_val = torch.tensor(X[train_size:train_size+val_size])
    y_val = torch.tensor(y[train_size:train_size+val_size])

    X_test = torch.tensor(X[train_size+val_size:])
    y_test = torch.tensor(y[train_size+val_size:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=shuffle_train
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Example usage
    print("Testing data loading...")

    # Test Bybit data loading
    df = load_bybit_data('BTCUSDT', interval='1h', limit=100)

    if not df.empty:
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Test feature calculation
        df = calculate_features(df)
        print(f"Features: {df.columns.tolist()}")
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("No data loaded (API might be unavailable)")
