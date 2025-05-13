//! Trading signal generation

use ndarray::Array2;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Buy signal with size (0, 1]
    Buy(f64),
    /// Sell signal with size (0, 1]
    Sell(f64),
    /// Hold / neutral
    Hold,
}

impl TradingSignal {
    /// Returns the position size (-1 to 1)
    pub fn position_size(&self) -> f64 {
        match self {
            TradingSignal::Buy(size) => *size,
            TradingSignal::Sell(size) => -*size,
            TradingSignal::Hold => 0.0,
        }
    }

    /// Returns true if this is a buy signal
    pub fn is_buy(&self) -> bool {
        matches!(self, TradingSignal::Buy(_))
    }

    /// Returns true if this is a sell signal
    pub fn is_sell(&self) -> bool {
        matches!(self, TradingSignal::Sell(_))
    }

    /// Returns true if this is a hold signal
    pub fn is_hold(&self) -> bool {
        matches!(self, TradingSignal::Hold)
    }
}

/// Signal generator from model predictions
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Threshold for generating buy signal
    pub buy_threshold: f64,
    /// Threshold for generating sell signal
    pub sell_threshold: f64,
    /// Minimum confidence for trade
    pub min_confidence: f64,
    /// Whether to use cumulative return over horizon
    pub use_cumulative: bool,
    /// Maximum position size
    pub max_position: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.001, -0.001)
    }
}

impl SignalGenerator {
    /// Creates a new signal generator
    pub fn new(buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
            min_confidence: 0.0,
            use_cumulative: true,
            max_position: 1.0,
        }
    }

    /// Creates generator with all parameters
    pub fn with_params(
        buy_threshold: f64,
        sell_threshold: f64,
        min_confidence: f64,
        max_position: f64,
    ) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
            min_confidence,
            use_cumulative: true,
            max_position,
        }
    }

    /// Generates signals from model predictions
    ///
    /// # Arguments
    /// * `predictions` - Model output [batch, horizon]
    ///
    /// # Returns
    /// Vector of trading signals
    pub fn generate(&self, predictions: &Array2<f64>) -> Vec<TradingSignal> {
        let batch_size = predictions.dim().0;
        let horizon = predictions.dim().1;
        let mut signals = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Calculate expected return
            let expected_return = if self.use_cumulative {
                // Sum of returns over horizon
                (0..horizon).map(|h| predictions[[b, h]]).sum::<f64>()
            } else {
                // Just first period return
                predictions[[b, 0]]
            };

            // Calculate confidence (inverse of prediction variance)
            let mean = expected_return / horizon as f64;
            let variance: f64 = (0..horizon)
                .map(|h| (predictions[[b, h]] - mean).powi(2))
                .sum::<f64>()
                / horizon as f64;
            let confidence = 1.0 / (1.0 + variance.sqrt() * 100.0);

            // Generate signal
            let signal = if confidence < self.min_confidence {
                TradingSignal::Hold
            } else if expected_return > self.buy_threshold {
                let size = self.calculate_position_size(expected_return, confidence);
                TradingSignal::Buy(size)
            } else if expected_return < self.sell_threshold {
                let size = self.calculate_position_size(-expected_return, confidence);
                TradingSignal::Sell(size)
            } else {
                TradingSignal::Hold
            };

            signals.push(signal);
        }

        signals
    }

    /// Generates a single signal from prediction
    pub fn generate_single(&self, prediction: &[f64]) -> TradingSignal {
        let expected_return: f64 = if self.use_cumulative {
            prediction.iter().sum()
        } else {
            *prediction.first().unwrap_or(&0.0)
        };

        if expected_return > self.buy_threshold {
            let size = (expected_return / self.buy_threshold).min(self.max_position);
            TradingSignal::Buy(size)
        } else if expected_return < self.sell_threshold {
            let size = (expected_return / self.sell_threshold).min(self.max_position);
            TradingSignal::Sell(size)
        } else {
            TradingSignal::Hold
        }
    }

    /// Calculates position size based on signal strength and confidence
    fn calculate_position_size(&self, signal_strength: f64, confidence: f64) -> f64 {
        let raw_size = signal_strength.abs() * confidence * 100.0;
        raw_size.min(self.max_position).max(0.1)
    }
}

/// Signal statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SignalStats {
    /// Total number of signals
    pub total: usize,
    /// Number of buy signals
    pub buys: usize,
    /// Number of sell signals
    pub sells: usize,
    /// Number of hold signals
    pub holds: usize,
    /// Average buy size
    pub avg_buy_size: f64,
    /// Average sell size
    pub avg_sell_size: f64,
}

#[allow(dead_code)]
impl SignalStats {
    /// Calculates statistics from a slice of signals
    pub fn from_signals(signals: &[TradingSignal]) -> Self {
        let total = signals.len();
        let mut buys = 0;
        let mut sells = 0;
        let mut holds = 0;
        let mut buy_size_sum = 0.0;
        let mut sell_size_sum = 0.0;

        for signal in signals {
            match signal {
                TradingSignal::Buy(size) => {
                    buys += 1;
                    buy_size_sum += size;
                }
                TradingSignal::Sell(size) => {
                    sells += 1;
                    sell_size_sum += size;
                }
                TradingSignal::Hold => {
                    holds += 1;
                }
            }
        }

        Self {
            total,
            buys,
            sells,
            holds,
            avg_buy_size: if buys > 0 {
                buy_size_sum / buys as f64
            } else {
                0.0
            },
            avg_sell_size: if sells > 0 {
                sell_size_sum / sells as f64
            } else {
                0.0
            },
        }
    }

    /// Returns buy ratio
    pub fn buy_ratio(&self) -> f64 {
        if self.total > 0 {
            self.buys as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Returns sell ratio
    pub fn sell_ratio(&self) -> f64 {
        if self.total > 0 {
            self.sells as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Returns activity ratio (non-hold)
    pub fn activity_ratio(&self) -> f64 {
        if self.total > 0 {
            (self.buys + self.sells) as f64 / self.total as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_position_size() {
        assert_eq!(TradingSignal::Buy(0.5).position_size(), 0.5);
        assert_eq!(TradingSignal::Sell(0.5).position_size(), -0.5);
        assert_eq!(TradingSignal::Hold.position_size(), 0.0);
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::new(0.01, -0.01);

        // Positive prediction -> Buy
        let signal = generator.generate_single(&[0.02, 0.01, 0.01]);
        assert!(signal.is_buy());

        // Negative prediction -> Sell
        let signal = generator.generate_single(&[-0.02, -0.01, -0.01]);
        assert!(signal.is_sell());

        // Near-zero prediction -> Hold
        let signal = generator.generate_single(&[0.001, -0.001, 0.0]);
        assert!(signal.is_hold());
    }

    #[test]
    fn test_generate_batch() {
        let generator = SignalGenerator::new(0.01, -0.01);

        let predictions = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.02, 0.01, 0.01, 0.01, // Buy
                -0.02, -0.01, -0.01, -0.01, // Sell
                0.001, -0.001, 0.0, 0.0, // Hold
            ],
        )
        .unwrap();

        let signals = generator.generate(&predictions);

        assert_eq!(signals.len(), 3);
        assert!(signals[0].is_buy());
        assert!(signals[1].is_sell());
        assert!(signals[2].is_hold());
    }

    #[test]
    fn test_signal_stats() {
        let signals = vec![
            TradingSignal::Buy(0.5),
            TradingSignal::Buy(0.7),
            TradingSignal::Sell(0.3),
            TradingSignal::Hold,
            TradingSignal::Hold,
        ];

        let stats = SignalStats::from_signals(&signals);

        assert_eq!(stats.total, 5);
        assert_eq!(stats.buys, 2);
        assert_eq!(stats.sells, 1);
        assert_eq!(stats.holds, 2);
        assert!((stats.avg_buy_size - 0.6).abs() < 0.01);
        assert!((stats.avg_sell_size - 0.3).abs() < 0.01);
        assert!((stats.activity_ratio() - 0.6).abs() < 0.01);
    }
}
