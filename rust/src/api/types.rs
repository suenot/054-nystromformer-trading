//! API data types for Bybit integration

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Open time (Unix timestamp in milliseconds)
    pub open_time: i64,
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Quote volume (in quote currency)
    pub quote_volume: f64,
}

impl Kline {
    /// Creates a new Kline
    pub fn new(
        open_time: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            open_time,
            open,
            high,
            low,
            close,
            volume,
            quote_volume: close * volume,
        }
    }

    /// Returns the datetime
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.open_time)
            .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap())
    }

    /// Calculates log return from previous candle
    pub fn log_return(&self, prev: &Kline) -> f64 {
        (self.close / prev.close).ln()
    }

    /// Calculates typical price (HLC/3)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculates true range
    pub fn true_range(&self, prev: &Kline) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev.close).abs();
        let lc = (self.low - prev.close).abs();
        hl.max(hc).max(lc)
    }

    /// Calculates body size (|close - open|)
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    /// Returns true if bullish (close > open)
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Order book level (price and quantity)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this price
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: i64,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Returns the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Returns the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Returns the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Returns the spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Returns the spread as percentage of mid price
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculates order imbalance at top N levels
    pub fn imbalance(&self, levels: usize) -> f64 {
        let bid_qty: f64 = self.bids.iter().take(levels).map(|l| l.quantity).sum();
        let ask_qty: f64 = self.asks.iter().take(levels).map(|l| l.quantity).sum();
        let total = bid_qty + ask_qty;

        if total > 0.0 {
            (bid_qty - ask_qty) / total
        } else {
            0.0
        }
    }

    /// Calculates volume-weighted average price for given quantity
    ///
    /// Returns None if:
    /// - target_qty is zero or negative
    /// - there is insufficient depth to fill the order
    pub fn vwap_bid(&self, target_qty: f64) -> Option<f64> {
        // Guard against invalid target quantity
        if target_qty <= 0.0 {
            return None;
        }

        let mut remaining = target_qty;
        let mut total_value = 0.0;
        let mut total_qty = 0.0;

        for level in &self.bids {
            let fill_qty = remaining.min(level.quantity);
            total_value += fill_qty * level.price;
            total_qty += fill_qty;
            remaining -= fill_qty;

            if remaining <= 0.0 {
                break;
            }
        }

        // Return None if insufficient depth (couldn't fill the order)
        if remaining > 0.0 {
            return None;
        }

        if total_qty > 0.0 {
            Some(total_value / total_qty)
        } else {
            None
        }
    }

    /// Calculates volume-weighted average price for given quantity (asks)
    ///
    /// Returns None if:
    /// - target_qty is zero or negative
    /// - there is insufficient depth to fill the order
    pub fn vwap_ask(&self, target_qty: f64) -> Option<f64> {
        // Guard against invalid target quantity
        if target_qty <= 0.0 {
            return None;
        }

        let mut remaining = target_qty;
        let mut total_value = 0.0;
        let mut total_qty = 0.0;

        for level in &self.asks {
            let fill_qty = remaining.min(level.quantity);
            total_value += fill_qty * level.price;
            total_qty += fill_qty;
            remaining -= fill_qty;

            if remaining <= 0.0 {
                break;
            }
        }

        // Return None if insufficient depth (couldn't fill the order)
        if remaining > 0.0 {
            return None;
        }

        if total_qty > 0.0 {
            Some(total_value / total_qty)
        } else {
            None
        }
    }
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h price change
    pub price_change_24h: f64,
    /// 24h price change percentage
    pub price_change_pct_24h: f64,
    /// 24h high
    pub high_24h: f64,
    /// 24h low
    pub low_24h: f64,
    /// 24h volume
    pub volume_24h: f64,
    /// 24h quote volume
    pub quote_volume_24h: f64,
    /// Best bid price
    pub bid_price: f64,
    /// Best ask price
    pub ask_price: f64,
    /// Timestamp
    pub timestamp: i64,
}

impl Ticker {
    /// Returns the spread
    pub fn spread(&self) -> f64 {
        self.ask_price - self.bid_price
    }

    /// Returns the mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    /// Returns volatility estimate (high-low range)
    pub fn range_volatility(&self) -> f64 {
        if self.last_price > 0.0 {
            (self.high_24h - self.low_24h) / self.last_price
        } else {
            0.0
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Klines result from Bybit
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct KlinesResult {
    pub category: Option<String>,
    pub symbol: String,
    pub list: Vec<Vec<String>>,
}

/// Ticker result from Bybit
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct TickerResult {
    pub category: Option<String>,
    pub list: Vec<TickerData>,
}

#[derive(Debug, Deserialize)]
pub struct TickerData {
    pub symbol: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
    #[serde(rename = "price24hPcnt")]
    pub price_24h_pcnt: String,
    #[serde(rename = "highPrice24h")]
    pub high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    pub low_price_24h: String,
    #[serde(rename = "volume24h")]
    pub volume_24h: String,
    #[serde(rename = "turnover24h")]
    pub turnover_24h: String,
    #[serde(rename = "bid1Price")]
    pub bid1_price: String,
    #[serde(rename = "ask1Price")]
    pub ask1_price: String,
}

/// Order book result from Bybit
#[derive(Debug, Deserialize)]
pub struct OrderBookResult {
    pub s: String, // symbol
    pub b: Vec<Vec<String>>, // bids
    pub a: Vec<Vec<String>>, // asks
    pub ts: i64, // timestamp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_log_return() {
        let prev = Kline::new(0, 100.0, 105.0, 95.0, 100.0, 1000.0);
        let curr = Kline::new(60000, 100.0, 110.0, 98.0, 105.0, 1200.0);

        let lr = curr.log_return(&prev);
        assert!((lr - 0.04879).abs() < 0.001);
    }

    #[test]
    fn test_order_book_imbalance() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 100.0, quantity: 10.0 },
                OrderBookLevel { price: 99.0, quantity: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 5.0 },
                OrderBookLevel { price: 102.0, quantity: 15.0 },
            ],
        };

        let imb = ob.imbalance(2);
        // bid_qty = 30, ask_qty = 20, imb = (30-20)/50 = 0.2
        assert!((imb - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_order_book_spread() {
        let ob = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![OrderBookLevel { price: 99.5, quantity: 10.0 }],
            asks: vec![OrderBookLevel { price: 100.5, quantity: 10.0 }],
        };

        assert_eq!(ob.spread(), Some(1.0));
        assert_eq!(ob.mid_price(), Some(100.0));
        assert!((ob.spread_bps().unwrap() - 100.0).abs() < 0.01);
    }
}
