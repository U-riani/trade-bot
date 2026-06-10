-- V23 live order-book feature collection.
--
-- Binance serves only the CURRENT order book, so there is no historical depth
-- data. This table accumulates forward-looking snapshots collected in real time;
-- conclusions about predictive value are impossible until enough rows pile up.
--
-- No unique constraint: each poll is a distinct observation in time, and we must
-- never overwrite an earlier snapshot. Append-only by design.

CREATE TABLE IF NOT EXISTS order_book_snapshots (
    id BIGSERIAL PRIMARY KEY,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    collected_at TIMESTAMPTZ NOT NULL,
    best_bid_price DOUBLE PRECISION NOT NULL,
    best_ask_price DOUBLE PRECISION NOT NULL,
    spread DOUBLE PRECISION NOT NULL,
    spread_pct DOUBLE PRECISION NOT NULL,
    bid_volume_top_5 DOUBLE PRECISION NOT NULL,
    ask_volume_top_5 DOUBLE PRECISION NOT NULL,
    bid_volume_top_10 DOUBLE PRECISION NOT NULL,
    ask_volume_top_10 DOUBLE PRECISION NOT NULL,
    bid_volume_top_20 DOUBLE PRECISION NOT NULL,
    ask_volume_top_20 DOUBLE PRECISION NOT NULL,
    imbalance_top_5 DOUBLE PRECISION NOT NULL,
    imbalance_top_10 DOUBLE PRECISION NOT NULL,
    imbalance_top_20 DOUBLE PRECISION NOT NULL,
    raw_depth_limit INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_order_book_snapshots_lookup
    ON order_book_snapshots(exchange, symbol, collected_at DESC);

-- Aggregated order-book features land in market_features alongside the V22
-- columns. These are added here so the V22 table can hold per-depth imbalance.
ALTER TABLE market_features
    ADD COLUMN IF NOT EXISTS imbalance_top_5 DOUBLE PRECISION;
ALTER TABLE market_features
    ADD COLUMN IF NOT EXISTS imbalance_top_10 DOUBLE PRECISION;
ALTER TABLE market_features
    ADD COLUMN IF NOT EXISTS imbalance_top_20 DOUBLE PRECISION;
ALTER TABLE market_features
    ADD COLUMN IF NOT EXISTS order_book_snapshot_count INTEGER;
