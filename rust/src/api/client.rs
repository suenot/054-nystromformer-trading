//! Bybit API client for fetching market data

use reqwest::Client;
use std::time::Duration;
use thiserror::Error;

use crate::api::types::*;

/// Bybit API errors
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid interval: {0}")]
    InvalidInterval(String),
}

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Default request timeout in seconds
    const DEFAULT_TIMEOUT_SECS: u64 = 10;

    /// Creates a new Bybit client with default timeout
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(Self::DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("Failed to build reqwest client");
        Self {
            client,
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Creates a client with custom base URL (for testnet)
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(Self::DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("Failed to build reqwest client");
        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Fetches klines (candlestick) data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
    /// * `limit` - Number of klines to fetch (max 1000)
    ///
    /// # Returns
    /// Vector of Kline data sorted by time (oldest first)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Kline>, BybitError> {
        self.validate_interval(interval)?;

        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(1000)
        );

        let response: BybitResponse<KlinesResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        // Parse klines, propagating errors instead of silently dropping them
        let mut klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .map(|row| self.parse_kline(&row))
            .collect::<Result<_, _>>()?;

        // Sort by time ascending
        klines.sort_by_key(|k| k.open_time);

        Ok(klines)
    }

    /// Fetches multiple pages of klines for long sequences
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval
    /// * `total` - Total number of klines to fetch
    pub async fn get_klines_extended(
        &self,
        symbol: &str,
        interval: &str,
        total: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        let mut all_klines = Vec::with_capacity(total);
        let mut end_time: Option<i64> = None;

        while all_klines.len() < total {
            let url = match end_time {
                Some(t) => format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit=1000&end={}",
                    self.base_url, symbol, interval, t
                ),
                None => format!(
                    "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit=1000",
                    self.base_url, symbol, interval
                ),
            };

            let response: BybitResponse<KlinesResult> = self.client
                .get(&url)
                .send()
                .await?
                .json()
                .await?;

            if response.ret_code != 0 {
                return Err(BybitError::ApiError {
                    code: response.ret_code,
                    message: response.ret_msg,
                });
            }

            // Parse klines, propagating errors instead of silently dropping them
            let klines: Vec<Kline> = response
                .result
                .list
                .into_iter()
                .map(|row| self.parse_kline(&row))
                .collect::<Result<_, _>>()?;

            if klines.is_empty() {
                break;
            }

            // Update end_time for next page (get data before the oldest)
            end_time = klines.iter().map(|k| k.open_time).min();

            all_klines.extend(klines);
        }

        // Sort and truncate
        all_klines.sort_by_key(|k| k.open_time);
        all_klines.truncate(total);

        Ok(all_klines)
    }

    /// Fetches ticker data
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError> {
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            self.base_url, symbol
        );

        let response: BybitResponse<TickerResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let data = response.result.list.into_iter().next()
            .ok_or_else(|| BybitError::ParseError("No ticker data".to_string()))?;

        // Parse last price and percentage change
        let last_price: f64 = data.last_price.parse().unwrap_or(0.0);
        let pct: f64 = data.price_24h_pcnt.parse().unwrap_or(0.0); // decimal (e.g., 0.05 == 5%)

        // Calculate absolute price change from last_price and percentage
        // If current = previous * (1 + pct), then change = current - previous = current * pct / (1 + pct)
        let price_change_24h = if (1.0_f64 + pct).abs() > f64::EPSILON {
            last_price * pct / (1.0 + pct)
        } else {
            0.0
        };

        Ok(Ticker {
            symbol: data.symbol,
            last_price,
            price_change_24h,
            price_change_pct_24h: pct * 100.0,
            high_24h: data.high_price_24h.parse().unwrap_or(0.0),
            low_24h: data.low_price_24h.parse().unwrap_or(0.0),
            volume_24h: data.volume_24h.parse().unwrap_or(0.0),
            quote_volume_24h: data.turnover_24h.parse().unwrap_or(0.0),
            bid_price: data.bid1_price.parse().unwrap_or(0.0),
            ask_price: data.ask1_price.parse().unwrap_or(0.0),
            timestamp: chrono::Utc::now().timestamp_millis(),
        })
    }

    /// Fetches order book data
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: u32,
    ) -> Result<OrderBook, BybitError> {
        let url = format!(
            "{}/v5/market/orderbook?category=linear&symbol={}&limit={}",
            self.base_url, symbol, limit.min(500)
        );

        let response: BybitResponse<OrderBookResult> = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        if response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: response.ret_code,
                message: response.ret_msg,
            });
        }

        let bids: Vec<OrderBookLevel> = response.result.b
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().ok()?,
                        quantity: row[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = response.result.a
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().ok()?,
                        quantity: row[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: response.result.s,
            timestamp: response.result.ts,
            bids,
            asks,
        })
    }

    /// Parses a kline from API response
    fn parse_kline(&self, row: &[String]) -> Result<Kline, BybitError> {
        if row.len() < 7 {
            return Err(BybitError::ParseError("Invalid kline data".to_string()));
        }

        Ok(Kline {
            open_time: row[0].parse().map_err(|_| {
                BybitError::ParseError("Invalid timestamp".to_string())
            })?,
            open: row[1].parse().map_err(|_| {
                BybitError::ParseError("Invalid open".to_string())
            })?,
            high: row[2].parse().map_err(|_| {
                BybitError::ParseError("Invalid high".to_string())
            })?,
            low: row[3].parse().map_err(|_| {
                BybitError::ParseError("Invalid low".to_string())
            })?,
            close: row[4].parse().map_err(|_| {
                BybitError::ParseError("Invalid close".to_string())
            })?,
            volume: row[5].parse().map_err(|_| {
                BybitError::ParseError("Invalid volume".to_string())
            })?,
            quote_volume: row[6].parse().map_err(|_| {
                BybitError::ParseError("Invalid quote volume".to_string())
            })?,
        })
    }

    /// Validates interval format
    fn validate_interval(&self, interval: &str) -> Result<(), BybitError> {
        let valid = [
            "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M",
        ];
        if valid.contains(&interval) {
            Ok(())
        } else {
            Err(BybitError::InvalidInterval(interval.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert!(client.base_url.contains("bybit"));
    }

    #[test]
    fn test_interval_validation() {
        let client = BybitClient::new();

        assert!(client.validate_interval("1").is_ok());
        assert!(client.validate_interval("60").is_ok());
        assert!(client.validate_interval("D").is_ok());
        assert!(client.validate_interval("invalid").is_err());
    }

    #[test]
    fn test_parse_kline() {
        let client = BybitClient::new();

        let row = vec![
            "1704067200000".to_string(),
            "42000.50".to_string(),
            "42500.00".to_string(),
            "41800.00".to_string(),
            "42300.00".to_string(),
            "1500.5".to_string(),
            "63000000.0".to_string(),
        ];

        let kline = client.parse_kline(&row).unwrap();

        assert_eq!(kline.open_time, 1704067200000);
        assert!((kline.open - 42000.50).abs() < 0.01);
        assert!((kline.close - 42300.00).abs() < 0.01);
    }
}
