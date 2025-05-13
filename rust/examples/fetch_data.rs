//! Example: Fetch data from Bybit API

use nystromformer_trading::{BybitClient, SequenceLoader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Nyströmformer Trading: Data Fetch Example ===\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch klines
    println!("Fetching BTCUSDT 1-minute klines...");
    let klines = client.get_klines("BTCUSDT", "1", 1000).await?;
    println!("Fetched {} klines", klines.len());

    if let Some(first) = klines.first() {
        println!(
            "First candle: {} O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            first.datetime().format("%Y-%m-%d %H:%M"),
            first.open,
            first.high,
            first.low,
            first.close,
            first.volume
        );
    }

    if let Some(last) = klines.last() {
        println!(
            "Last candle:  {} O={:.2} H={:.2} L={:.2} C={:.2} V={:.0}",
            last.datetime().format("%Y-%m-%d %H:%M"),
            last.open,
            last.high,
            last.low,
            last.close,
            last.volume
        );
    }

    // Prepare dataset
    println!("\nPreparing dataset for Nyströmformer...");
    let loader = SequenceLoader::new();

    if klines.len() >= 500 {
        match loader.prepare_dataset(&klines, 256, 24) {
            Ok(dataset) => {
                println!("Dataset prepared successfully:");
                println!("  Samples: {}", dataset.len());
                println!("  Sequence length: {}", dataset.seq_len());
                println!("  Features: {}", dataset.num_features());
                println!("  Feature names: {:?}", dataset.feature_names);
            }
            Err(e) => {
                println!("Could not prepare dataset: {}", e);
            }
        }
    } else {
        println!("Not enough data for dataset preparation");
    }

    // Fetch ticker
    println!("\nFetching current ticker...");
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!(
        "BTCUSDT: Last={:.2} Bid={:.2} Ask={:.2} 24h Change={:.2}%",
        ticker.last_price,
        ticker.bid_price,
        ticker.ask_price,
        ticker.price_change_pct_24h
    );

    // Fetch order book
    println!("\nFetching order book...");
    let orderbook = client.get_orderbook("BTCUSDT", 10).await?;
    if let Some(mid) = orderbook.mid_price() {
        println!("Mid price: {:.2}", mid);
    }
    if let Some(spread) = orderbook.spread_bps() {
        println!("Spread: {:.2} bps", spread);
    }
    println!("Order imbalance (top 5): {:.4}", orderbook.imbalance(5));

    println!("\n=== Done ===");
    Ok(())
}
