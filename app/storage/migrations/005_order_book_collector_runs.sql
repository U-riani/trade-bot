-- V25 collector run tracking.
--
-- One row per collector invocation, so a days/weeks-long collection effort can
-- be audited: when each run started/stopped, how it ended, and how many
-- snapshots/failures it saw. This is operational metadata only -- it stores no
-- market data and never affects trading (there is no trading).
--
-- run_id is UNIQUE so a restarted run with the same id upserts rather than
-- duplicating, which keeps run history resume-safe.

CREATE TABLE IF NOT EXISTS order_book_collector_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    stopped_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    interval_seconds DOUBLE PRECISION NOT NULL,
    depth_limit INTEGER NOT NULL,
    collected_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_snapshot_at TIMESTAMPTZ,
    stop_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_order_book_collector_runs_lookup
    ON order_book_collector_runs(exchange, symbol, started_at DESC);
