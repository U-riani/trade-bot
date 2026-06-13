-- V26 historical aggregate-trade pressure features.
--
-- These columns are derived from Binance historical aggTrades. They are separate
-- from kline taker fields and from live order-book fields so the research layer
-- does not quietly mix incompatible data sources.

ALTER TABLE market_features
    ADD COLUMN IF NOT EXISTS trade_count INTEGER,
    ADD COLUMN IF NOT EXISTS taker_buy_trade_count INTEGER,
    ADD COLUMN IF NOT EXISTS taker_sell_trade_count INTEGER,
    ADD COLUMN IF NOT EXISTS taker_buy_base_volume_trades DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_sell_base_volume_trades DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_buy_quote_volume_trades DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_sell_quote_volume_trades DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_net_base_volume DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_net_quote_volume DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_buy_trade_ratio DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_buy_base_ratio_trades DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS taker_buy_quote_ratio_trades DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS avg_trade_quote_size DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS trade_count_intensity DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quote_volume_intensity DOUBLE PRECISION;
