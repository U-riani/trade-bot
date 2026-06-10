-- V22 non-price market feature research layer.
--
-- Stores per-candle non-price features so strategy research can test whether
-- order-flow / volume information improves filtering over OHLC alone.
--
-- HONESTY NOTE on availability:
--   * volume / quote_volume / taker_buy_base_volume / taker_buy_quote_volume /
--     taker_buy_ratio are AVAILABLE HISTORICALLY from Binance klines.
--   * order_book_bid_volume / order_book_ask_volume / order_book_imbalance /
--     spread_pct are ONLY available going forward from a live depth snapshot.
--     Binance REST /depth returns the CURRENT book, not historical books, so
--     these columns are NULL for any row built from historical candles. We do
--     not fabricate them from price data.

CREATE TABLE IF NOT EXISTS market_features (
    id BIGSERIAL PRIMARY KEY,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ NOT NULL,
    close_price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    quote_volume DOUBLE PRECISION,
    taker_buy_base_volume DOUBLE PRECISION,
    taker_buy_quote_volume DOUBLE PRECISION,
    taker_buy_ratio DOUBLE PRECISION,
    order_book_bid_volume DOUBLE PRECISION,
    order_book_ask_volume DOUBLE PRECISION,
    order_book_imbalance DOUBLE PRECISION,
    spread_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(exchange, symbol, timeframe, close_time)
);

CREATE INDEX IF NOT EXISTS idx_market_features_lookup
    ON market_features(exchange, symbol, timeframe, close_time DESC);
