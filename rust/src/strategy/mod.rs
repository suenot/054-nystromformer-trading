//! Trading strategy and backtesting module

mod backtest;
mod signals;

pub use backtest::{NystromBacktester, BacktestConfig, BacktestResult};
pub use signals::{TradingSignal, SignalGenerator};
