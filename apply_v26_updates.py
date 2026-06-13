from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _path(rel: str) -> Path:
    return ROOT / rel


def _read(rel: str) -> str:
    return _path(rel).read_text(encoding="utf-8")


def _write(rel: str, content: str) -> None:
    path = _path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {rel}")


def _replace_once(rel: str, old: str, new: str) -> None:
    path = _path(rel)
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched {rel}")
        return
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {rel}:\n{old[:400]}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {rel}")


TRADE_PRESSURE = r'''"""V26 historical aggregate-trade pressure features.

Binance Spot does not provide historical order-book depth through normal REST, but
it does provide historical aggregate trades. This module converts aggTrades into
bucketed taker pressure features that can be researched while live order-book data
continues to accumulate.

No trading logic lives here. Everything is pure and unit-testable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.market.features import bucket_start_timestamp
from app.utils.time import ms_to_datetime
from app.utils.timeframe import timeframe_to_seconds


@dataclass(slots=True, frozen=True)
class AggTradeRow:
    trade_id: int
    price: float
    quantity: float
    quote_quantity: float
    trade_time: datetime
    buyer_is_maker: bool

    @property
    def taker_is_buy(self) -> bool:
        """False buyer_is_maker means buyer took liquidity, so aggressive buy."""
        return not self.buyer_is_maker


@dataclass(slots=True, frozen=True)
class TradePressureAggregate:
    bucket_start_ts: int
    trade_count: int
    taker_buy_trade_count: int
    taker_sell_trade_count: int
    taker_buy_base_volume: float
    taker_sell_base_volume: float
    taker_buy_quote_volume: float
    taker_sell_quote_volume: float
    taker_net_base_volume: float
    taker_net_quote_volume: float
    taker_buy_trade_ratio: float | None
    taker_buy_base_ratio: float | None
    taker_buy_quote_ratio: float | None
    avg_trade_quote_size: float | None
    trade_count_intensity: float
    quote_volume_intensity: float


def parse_agg_trade_row(item: dict[str, Any]) -> AggTradeRow:
    """Parse one Binance aggTrade row.

    Binance fields:
    - a: aggregate trade id
    - p: price
    - q: quantity
    - T: trade time in ms
    - m: buyer is maker

    Direction mapping:
    - m == True  -> buyer was maker, seller was taker, aggressive sell
    - m == False -> buyer was taker, aggressive buy
    """
    price = float(item["p"])
    quantity = float(item["q"])
    return AggTradeRow(
        trade_id=int(item["a"]),
        price=price,
        quantity=quantity,
        quote_quantity=price * quantity,
        trade_time=ms_to_datetime(int(item["T"])),
        buyer_is_maker=bool(item["m"]),
    )


def parse_agg_trades(items: list[dict[str, Any]]) -> list[AggTradeRow]:
    rows: list[AggTradeRow] = []
    for item in items:
        try:
            rows.append(parse_agg_trade_row(item))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def aggregate_trade_pressure_by_bucket(
    rows: list[AggTradeRow],
    *,
    target_timeframe: str,
) -> dict[int, TradePressureAggregate]:
    """Aggregate parsed trades into UTC-aligned timeframe buckets."""
    target_seconds = timeframe_to_seconds(target_timeframe)
    bucket_minutes = target_seconds / 60.0

    sums: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "trade_count": 0.0,
            "buy_count": 0.0,
            "sell_count": 0.0,
            "buy_base": 0.0,
            "sell_base": 0.0,
            "buy_quote": 0.0,
            "sell_quote": 0.0,
        }
    )

    for row in rows:
        key = bucket_start_timestamp(row.trade_time, target_seconds)
        bucket = sums[key]
        bucket["trade_count"] += 1.0
        if row.taker_is_buy:
            bucket["buy_count"] += 1.0
            bucket["buy_base"] += row.quantity
            bucket["buy_quote"] += row.quote_quantity
        else:
            bucket["sell_count"] += 1.0
            bucket["sell_base"] += row.quantity
            bucket["sell_quote"] += row.quote_quantity

    result: dict[int, TradePressureAggregate] = {}
    for key, values in sums.items():
        trade_count = int(values["trade_count"])
        buy_count = int(values["buy_count"])
        sell_count = int(values["sell_count"])
        buy_base = values["buy_base"]
        sell_base = values["sell_base"]
        buy_quote = values["buy_quote"]
        sell_quote = values["sell_quote"]
        total_base = buy_base + sell_base
        total_quote = buy_quote + sell_quote
        result[key] = TradePressureAggregate(
            bucket_start_ts=key,
            trade_count=trade_count,
            taker_buy_trade_count=buy_count,
            taker_sell_trade_count=sell_count,
            taker_buy_base_volume=buy_base,
            taker_sell_base_volume=sell_base,
            taker_buy_quote_volume=buy_quote,
            taker_sell_quote_volume=sell_quote,
            taker_net_base_volume=buy_base - sell_base,
            taker_net_quote_volume=buy_quote - sell_quote,
            taker_buy_trade_ratio=_ratio(float(buy_count), float(trade_count)),
            taker_buy_base_ratio=_ratio(buy_base, total_base),
            taker_buy_quote_ratio=_ratio(buy_quote, total_quote),
            avg_trade_quote_size=_ratio(total_quote, float(trade_count)),
            trade_count_intensity=float(trade_count) / bucket_minutes if bucket_minutes > 0 else 0.0,
            quote_volume_intensity=total_quote / bucket_minutes if bucket_minutes > 0 else 0.0,
        )
    return result
'''

MIGRATION_006 = r'''-- V26 historical aggregate-trade pressure features.
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
'''

BACKFILL_TRADE_PRESSURE = r'''"""V26: backfill historical Binance aggregate-trade pressure features.

This script uses Binance aggTrades, not order-book depth. It enriches existing
market_features rows with historical taker-pressure features while live
order-book data continues to accumulate forward.

No trading, no strategy, no profitability claim.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.market.features import MarketFeatures, bucket_start_timestamp
from app.market.trade_pressure import aggregate_trade_pressure_by_bucket, parse_agg_trades
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.time import utc_now
from app.utils.timeframe import timeframe_to_seconds
from scripts.backtest_strategy import _resolve_market_data_source

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical aggregate-trade pressure features.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--start", default=None, help="UTC ISO start, e.g. 2026-06-13T00:00:00+00:00")
    parser.add_argument("--end", default=None, help="UTC ISO end. Default: now.")
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--candle-limit", type=int, default=50000)
    parser.add_argument("--max-requests", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true", help="Fetch/compute only; do not save.")
    parser.add_argument("--no-save", action="store_true", help="Alias for --dry-run.")
    return parser


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_timeframes(value: str) -> list[str]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in result:
            result.append(item)
    if not result:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return result


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.candle_limit <= 0:
        raise SystemExit("--candle-limit must be positive")
    if args.lookback_hours <= 0 and args.start is None:
        raise SystemExit("--lookback-hours must be positive when --start is omitted")

    market_data_source, use_testnet_data, exchange_id = _resolve_market_data_source(args.market_data_source)
    symbol = (args.symbol or settings.normalized_symbol).upper().strip()
    end_at = _parse_dt(args.end) or utc_now()
    start_at = _parse_dt(args.start) or (end_at - timedelta(hours=args.lookback_hours))
    if start_at >= end_at:
        raise SystemExit("start must be before end")

    logger.info(
        "trade_pressure_backfill_started",
        symbol=symbol,
        exchange=exchange_id,
        market_data_source=market_data_source,
        start=start_at.isoformat(),
        end=end_at.isoformat(),
        timeframes=args.timeframes,
        dry_run=args.dry_run or args.no_save,
    )

    client = BinanceRestClient(testnet=use_testnet_data)
    try:
        raw_trades = await client.get_historical_agg_trades(
            symbol=symbol,
            start_time_ms=_ms(start_at),
            end_time_ms=_ms(end_at),
            max_requests=args.max_requests,
        )
    finally:
        await client.close()

    trades = parse_agg_trades(raw_trades)
    logger.info("trade_pressure_trades_fetched", raw=len(raw_trades), parsed=len(trades))

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)
    try:
        source_candles = await repository.load_recent_candles(
            exchange=exchange_id,
            symbol=symbol,
            timeframe=args.source_timeframe,
            limit=args.candle_limit,
        )
        source_candles = [c for c in source_candles if start_at <= c.open_time <= end_at]
        if not source_candles:
            raise SystemExit("No source candles in requested window. Run backfill_candles first.")

        for timeframe in _parse_timeframes(args.timeframes):
            if timeframe == args.source_timeframe:
                candles = source_candles
            else:
                candles = resample_candles(
                    source_candles,
                    target_timeframe=timeframe,
                    source_timeframe=args.source_timeframe,
                )
            if not candles:
                logger.warning("trade_pressure_timeframe_skipped", timeframe=timeframe, reason="no_candles")
                continue

            target_seconds = timeframe_to_seconds(timeframe)
            by_bucket = aggregate_trade_pressure_by_bucket(trades, target_timeframe=timeframe)
            rows: list[MarketFeatures] = []
            matched = 0
            for candle in candles:
                bucket_key = bucket_start_timestamp(candle.open_time, target_seconds)
                pressure = by_bucket.get(bucket_key)
                if pressure is None:
                    continue
                matched += 1
                rows.append(
                    MarketFeatures(
                        exchange=exchange_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        open_time=candle.open_time,
                        close_time=candle.close_time,
                        close_price=candle.close,
                        volume=candle.volume,
                        trade_count=pressure.trade_count,
                        taker_buy_trade_count=pressure.taker_buy_trade_count,
                        taker_sell_trade_count=pressure.taker_sell_trade_count,
                        taker_buy_base_volume_trades=pressure.taker_buy_base_volume,
                        taker_sell_base_volume_trades=pressure.taker_sell_base_volume,
                        taker_buy_quote_volume_trades=pressure.taker_buy_quote_volume,
                        taker_sell_quote_volume_trades=pressure.taker_sell_quote_volume,
                        taker_net_base_volume=pressure.taker_net_base_volume,
                        taker_net_quote_volume=pressure.taker_net_quote_volume,
                        taker_buy_trade_ratio=pressure.taker_buy_trade_ratio,
                        taker_buy_base_ratio_trades=pressure.taker_buy_base_ratio,
                        taker_buy_quote_ratio_trades=pressure.taker_buy_quote_ratio,
                        avg_trade_quote_size=pressure.avg_trade_quote_size,
                        trade_count_intensity=pressure.trade_count_intensity,
                        quote_volume_intensity=pressure.quote_volume_intensity,
                    )
                )

            logger.info(
                "trade_pressure_timeframe_ready",
                timeframe=timeframe,
                candles=len(candles),
                buckets=len(by_bucket),
                matched_candles=matched,
                save=not (args.dry_run or args.no_save),
            )
            if rows and not (args.dry_run or args.no_save):
                saved = await repository.upsert_market_features_trade_pressure(rows)
                logger.info("trade_pressure_saved", timeframe=timeframe, rows=saved)
    finally:
        await db.close()

    logger.info("trade_pressure_backfill_finished")


if __name__ == "__main__":
    asyncio.run(main())
'''

COMPARE_FEATURE_GROUPS = r'''"""V26 feature-group comparison lab.

Compares candle-only, kline taker, aggregate-trade pressure, live order-book, and
combined available feature groups. This is research tooling only: no model, no
strategy, no trading.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.backtesting.feature_analysis import analyze_feature
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.analyze_market_features import HORIZONS, _build_feature_series
from scripts.backtest_strategy import _load_candles, _resolve_market_data_source

logger = get_logger(__name__)

FEATURE_GROUPS = {
    "candle_only": ["volume_spike_ratio", "body_pct", "upper_wick_pct", "lower_wick_pct"],
    "kline_taker": ["taker_buy_ratio"],
    "agg_trade_pressure": [
        "trade_count_intensity",
        "quote_volume_intensity",
        "taker_buy_trade_ratio",
        "taker_buy_base_ratio_trades",
        "taker_buy_quote_ratio_trades",
        "taker_net_base_volume",
        "taker_net_quote_volume",
        "avg_trade_quote_size",
    ],
    "live_order_book": ["order_book_imbalance", "spread_pct", "imbalance_top_5", "imbalance_top_10", "imbalance_top_20"],
}
FEATURE_GROUPS["combined_available"] = sorted({f for values in FEATURE_GROUPS.values() for f in values})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare predictive value of feature groups (research only).")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--volume-spike-lookback", type=int, default=20)
    parser.add_argument("--num-buckets", type=int, default=5)
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _parse_timeframes(value: str) -> list[str]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in result:
            result.append(item)
    if not result:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return result


def _quantile_spread(analysis) -> float | None:
    if not analysis.buckets:
        return None
    values = [bucket.avg_forward_return_pct for bucket in analysis.buckets]
    return max(values) - min(values)


def _export_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("feature_group_comparison_json_exported", path=str(path), rows=len(rows))


def _export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timeframe",
        "horizon",
        "group",
        "feature_count",
        "best_abs_correlation_feature",
        "best_abs_correlation",
        "best_quantile_spread_feature",
        "best_quantile_spread",
        "max_sample_size",
        "warning",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("feature_group_comparison_csv_exported", path=str(path), rows=len(rows))


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    source_candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not source_candles:
        raise SystemExit("No candles available for feature group comparison")

    market_data_source, _use_testnet, exchange_id = _resolve_market_data_source(args.market_data_source)
    logger.info(
        "feature_group_comparison_started",
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        source_candles=len(source_candles),
        timeframes=args.timeframes,
    )

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)
    payload: list[dict[str, Any]] = []
    try:
        for timeframe in _parse_timeframes(args.timeframes):
            candles = resample_candles(source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe)
            if not candles:
                continue
            rows = await repository.load_market_features(
                exchange=exchange_id,
                symbol=settings.normalized_symbol,
                timeframe=timeframe,
                limit=len(candles) + 10,
            )
            features_by_close_time = {row.close_time: row for row in rows}
            series = _build_feature_series(
                candles, features_by_close_time, volume_spike_lookback=args.volume_spike_lookback
            )
            closes = [candle.close for candle in candles]

            for horizon in HORIZONS:
                analyses = {
                    feature: analyze_feature(feature, values, closes, horizon, num_buckets=args.num_buckets)
                    for feature, values in series.items()
                }
                for group, feature_names in FEATURE_GROUPS.items():
                    group_analyses = [analyses[name] for name in feature_names if name in analyses]
                    valid_corr = [a for a in group_analyses if a.correlation is not None]
                    best_corr = max(valid_corr, key=lambda a: abs(a.correlation or 0.0), default=None)
                    with_spread = [(a, _quantile_spread(a)) for a in group_analyses]
                    with_spread = [(a, s) for a, s in with_spread if s is not None]
                    best_spread = max(with_spread, key=lambda item: item[1], default=None)
                    max_sample = max((a.sample_size for a in group_analyses), default=0)
                    warning = "sample_too_small" if max_sample < args.min_feature_samples else ""
                    payload.append(
                        {
                            "timeframe": timeframe,
                            "horizon": horizon,
                            "group": group,
                            "feature_count": len(group_analyses),
                            "best_abs_correlation_feature": None if best_corr is None else best_corr.feature,
                            "best_abs_correlation": None if best_corr is None else best_corr.correlation,
                            "best_quantile_spread_feature": None if best_spread is None else best_spread[0].feature,
                            "best_quantile_spread": None if best_spread is None else best_spread[1],
                            "max_sample_size": max_sample,
                            "warning": warning,
                        }
                    )
    finally:
        await db.close()

    if args.export_json:
        _export_json(args.export_json, payload)
    if args.export_csv:
        _export_csv(args.export_csv, payload)
    logger.info("feature_group_comparison_finished", rows=len(payload), note="research only; no strategy/trading")


if __name__ == "__main__":
    asyncio.run(main())
'''

TEST_TRADE_PRESSURE = r'''from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.market.trade_pressure import aggregate_trade_pressure_by_bucket, parse_agg_trade_row, parse_agg_trades


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_parse_agg_trade_direction_mapping() -> None:
    buy = parse_agg_trade_row({"a": 1, "p": "100", "q": "2", "T": _ms(datetime(2026, 1, 1, tzinfo=UTC)), "m": False})
    sell = parse_agg_trade_row({"a": 2, "p": "100", "q": "3", "T": _ms(datetime(2026, 1, 1, tzinfo=UTC)), "m": True})
    assert buy.taker_is_buy is True
    assert sell.taker_is_buy is False
    assert buy.quote_quantity == 200


def test_aggregate_trade_pressure_ratios_and_intensity() -> None:
    raw = [
        {"a": 1, "p": "100", "q": "2", "T": _ms(datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)), "m": False},
        {"a": 2, "p": "100", "q": "1", "T": _ms(datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)), "m": True},
    ]
    rows = parse_agg_trades(raw)
    buckets = aggregate_trade_pressure_by_bucket(rows, target_timeframe="1m")
    assert len(buckets) == 1
    agg = next(iter(buckets.values()))
    assert agg.trade_count == 2
    assert agg.taker_buy_trade_count == 1
    assert agg.taker_sell_trade_count == 1
    assert agg.taker_buy_base_volume == 2
    assert agg.taker_sell_base_volume == 1
    assert agg.taker_net_base_volume == 1
    assert agg.taker_buy_trade_ratio == pytest.approx(0.5)
    assert agg.taker_buy_base_ratio == pytest.approx(2 / 3)
    assert agg.avg_trade_quote_size == pytest.approx(150)
    assert agg.trade_count_intensity == pytest.approx(2)
'''

README_V26 = r'''

## V26 historical trade-pressure features and comparison lab

V26 adds historical Binance aggregate-trade pressure features so research can move
forward while live order-book data continues to accumulate.

Important limitation: Binance Spot REST does **not** provide historical order-book
depth. V26 does not fabricate historical order-book imbalance and does not attach
current order-book snapshots to past candles.

Recommended flow:

```powershell
python -m scripts.apply_migrations
python -m scripts.backfill_candles --market-data-source production --symbol BTCUSDT --timeframe 1m --limit 50000
python -m scripts.build_market_features --market-data-source production --source db --limit 50000 --timeframes 1m,5m,15m
python -m scripts.backfill_trade_pressure_features --market-data-source production --symbol BTCUSDT --lookback-hours 24 --timeframes 1m,5m,15m
python -m scripts.analyze_market_features --market-data-source production --source db --limit 50000 --timeframes 1m,5m,15m --min-feature-samples 100 --export-json reports/feature_analysis_v26.json --export-csv reports/feature_analysis_v26.csv
python -m scripts.compare_feature_groups --market-data-source production --source db --limit 50000 --timeframes 1m,5m,15m --export-json reports/feature_group_comparison_v26.json --export-csv reports/feature_group_comparison_v26.csv
```

V26 does not build a strategy. It adds historical trade-pressure features and
comparison tooling while live order-book data continues to accumulate.
'''


def patch_binance_rest() -> None:
    marker = "    async def get_order_book(self, *, symbol: str, limit: int = 100) -> dict[str, Any]:\n"
    insert = r'''    async def get_agg_trades(
        self,
        *,
        symbol: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        from_id: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Fetch Binance aggregate trades.

        This endpoint is public and historical/database-backed. It is not order
        book data. It lets V26 build historical trade-pressure features without
        fabricating historical depth.
        """
        if limit <= 0:
            return []
        params: dict[str, Any] = {"symbol": symbol.upper(), "limit": min(limit, 1000)}
        if from_id is not None:
            params["fromId"] = from_id
        else:
            if start_time_ms is not None:
                params["startTime"] = start_time_ms
            if end_time_ms is not None:
                params["endTime"] = end_time_ms

        response = await self.client.get(f"{self.base_url}/v3/aggTrades", params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected Binance aggTrades response: {data!r}")
        return data

    async def get_historical_agg_trades(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit_per_request: int = 1000,
        max_requests: int | None = None,
    ) -> list[dict[str, Any]]:
        """Page aggregate trades oldest -> newest over a time window.

        First request uses start/end. Later requests use fromId so we do not skip
        trades when many trades share a timestamp. Rows outside the requested
        window are filtered out.
        """
        if start_time_ms >= end_time_ms:
            return []
        safe_limit = min(max(limit_per_request, 1), 1000)
        rows_by_id: dict[int, dict[str, Any]] = {}
        from_id: int | None = None
        previous_max_id: int | None = None
        requests = 0

        while True:
            if max_requests is not None and requests >= max_requests:
                logger.warning("binance_agg_trades_max_requests_reached", max_requests=max_requests)
                break
            raw = await self.get_agg_trades(
                symbol=symbol,
                start_time_ms=start_time_ms if from_id is None else None,
                end_time_ms=end_time_ms if from_id is None else None,
                from_id=from_id,
                limit=safe_limit,
            )
            requests += 1
            if not raw:
                break

            max_id: int | None = None
            saw_in_window = False
            all_after_end = True
            for item in raw:
                trade_id = int(item["a"])
                trade_time = int(item["T"])
                max_id = trade_id if max_id is None else max(max_id, trade_id)
                if trade_time <= end_time_ms:
                    all_after_end = False
                if start_time_ms <= trade_time <= end_time_ms:
                    rows_by_id[trade_id] = item
                    saw_in_window = True

            if max_id is None:
                break
            if previous_max_id is not None and max_id <= previous_max_id:
                logger.warning("binance_agg_trades_stopped_no_progress", max_id=max_id)
                break
            previous_max_id = max_id
            from_id = max_id + 1

            if all_after_end:
                break
            if len(raw) < safe_limit and not saw_in_window:
                break
            if len(raw) < safe_limit:
                break

        return [rows_by_id[key] for key in sorted(rows_by_id)]

'''
    if "async def get_agg_trades" in _read("app/exchange/binance_rest.py"):
        print("already patched app/exchange/binance_rest.py")
        return
    _replace_once("app/exchange/binance_rest.py", marker, insert + marker)


def patch_features() -> None:
    rel = "app/market/features.py"
    text = _read(rel)
    if "trade_count: int | None" in text:
        print(f"already patched {rel}")
        return
    old = "    taker_buy_quote_volume: float | None = None\n    taker_buy_ratio: float | None = None\n    order_book_bid_volume: float | None = None\n"
    new = "    taker_buy_quote_volume: float | None = None\n    taker_buy_ratio: float | None = None\n    # V26 historical aggregate-trade pressure fields. These are separate from\n    # kline taker fields and from live order-book fields.\n    trade_count: int | None = None\n    taker_buy_trade_count: int | None = None\n    taker_sell_trade_count: int | None = None\n    taker_buy_base_volume_trades: float | None = None\n    taker_sell_base_volume_trades: float | None = None\n    taker_buy_quote_volume_trades: float | None = None\n    taker_sell_quote_volume_trades: float | None = None\n    taker_net_base_volume: float | None = None\n    taker_net_quote_volume: float | None = None\n    taker_buy_trade_ratio: float | None = None\n    taker_buy_base_ratio_trades: float | None = None\n    taker_buy_quote_ratio_trades: float | None = None\n    avg_trade_quote_size: float | None = None\n    trade_count_intensity: float | None = None\n    quote_volume_intensity: float | None = None\n    order_book_bid_volume: float | None = None\n"
    _replace_once(rel, old, new)


def patch_repositories() -> None:
    rel = "app/storage/repositories.py"
    text = _read(rel)
    if "trade_count, taker_buy_trade_count" not in text:
        old = """                   quote_volume, taker_buy_base_volume, taker_buy_quote_volume, taker_buy_ratio,
                   order_book_bid_volume, order_book_ask_volume, order_book_imbalance, spread_pct,
                   imbalance_top_5, imbalance_top_10, imbalance_top_20, order_book_snapshot_count
"""
        new = """                   quote_volume, taker_buy_base_volume, taker_buy_quote_volume, taker_buy_ratio,
                   trade_count, taker_buy_trade_count, taker_sell_trade_count,
                   taker_buy_base_volume_trades, taker_sell_base_volume_trades,
                   taker_buy_quote_volume_trades, taker_sell_quote_volume_trades,
                   taker_net_base_volume, taker_net_quote_volume,
                   taker_buy_trade_ratio, taker_buy_base_ratio_trades, taker_buy_quote_ratio_trades,
                   avg_trade_quote_size, trade_count_intensity, quote_volume_intensity,
                   order_book_bid_volume, order_book_ask_volume, order_book_imbalance, spread_pct,
                   imbalance_top_5, imbalance_top_10, imbalance_top_20, order_book_snapshot_count
"""
        _replace_once(rel, old, new)
    text = _read(rel)
    if "trade_count=_opt_int" not in text:
        old = """                taker_buy_quote_volume=_opt_float(row["taker_buy_quote_volume"]),
                taker_buy_ratio=_opt_float(row["taker_buy_ratio"]),
                order_book_bid_volume=_opt_float(row["order_book_bid_volume"]),
"""
        new = """                taker_buy_quote_volume=_opt_float(row["taker_buy_quote_volume"]),
                taker_buy_ratio=_opt_float(row["taker_buy_ratio"]),
                trade_count=_opt_int(row["trade_count"]),
                taker_buy_trade_count=_opt_int(row["taker_buy_trade_count"]),
                taker_sell_trade_count=_opt_int(row["taker_sell_trade_count"]),
                taker_buy_base_volume_trades=_opt_float(row["taker_buy_base_volume_trades"]),
                taker_sell_base_volume_trades=_opt_float(row["taker_sell_base_volume_trades"]),
                taker_buy_quote_volume_trades=_opt_float(row["taker_buy_quote_volume_trades"]),
                taker_sell_quote_volume_trades=_opt_float(row["taker_sell_quote_volume_trades"]),
                taker_net_base_volume=_opt_float(row["taker_net_base_volume"]),
                taker_net_quote_volume=_opt_float(row["taker_net_quote_volume"]),
                taker_buy_trade_ratio=_opt_float(row["taker_buy_trade_ratio"]),
                taker_buy_base_ratio_trades=_opt_float(row["taker_buy_base_ratio_trades"]),
                taker_buy_quote_ratio_trades=_opt_float(row["taker_buy_quote_ratio_trades"]),
                avg_trade_quote_size=_opt_float(row["avg_trade_quote_size"]),
                trade_count_intensity=_opt_float(row["trade_count_intensity"]),
                quote_volume_intensity=_opt_float(row["quote_volume_intensity"]),
                order_book_bid_volume=_opt_float(row["order_book_bid_volume"]),
"""
        _replace_once(rel, old, new)
    text = _read(rel)
    if "upsert_market_features_trade_pressure" not in text:
        old = """    async def count_market_features(self, *, exchange: str, symbol: str, timeframe: str) -> int:
"""
        method = r'''    async def upsert_market_features_trade_pressure(self, rows: list[MarketFeatures]) -> int:
        """Insert/update only V26 aggregate-trade pressure columns.

        Existing candle/kline/order-book columns are intentionally left untouched
        on conflict so V26 cannot corrupt V23/V24/V25 order-book data.
        """
        if not rows:
            return 0
        pool = self.db.require_pool()
        await pool.executemany(
            """
            INSERT INTO market_features(
                exchange, symbol, timeframe, open_time, close_time, close_price, volume,
                trade_count, taker_buy_trade_count, taker_sell_trade_count,
                taker_buy_base_volume_trades, taker_sell_base_volume_trades,
                taker_buy_quote_volume_trades, taker_sell_quote_volume_trades,
                taker_net_base_volume, taker_net_quote_volume,
                taker_buy_trade_ratio, taker_buy_base_ratio_trades, taker_buy_quote_ratio_trades,
                avg_trade_quote_size, trade_count_intensity, quote_volume_intensity
            )
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22)
            ON CONFLICT (exchange, symbol, timeframe, close_time) DO UPDATE SET
                trade_count = EXCLUDED.trade_count,
                taker_buy_trade_count = EXCLUDED.taker_buy_trade_count,
                taker_sell_trade_count = EXCLUDED.taker_sell_trade_count,
                taker_buy_base_volume_trades = EXCLUDED.taker_buy_base_volume_trades,
                taker_sell_base_volume_trades = EXCLUDED.taker_sell_base_volume_trades,
                taker_buy_quote_volume_trades = EXCLUDED.taker_buy_quote_volume_trades,
                taker_sell_quote_volume_trades = EXCLUDED.taker_sell_quote_volume_trades,
                taker_net_base_volume = EXCLUDED.taker_net_base_volume,
                taker_net_quote_volume = EXCLUDED.taker_net_quote_volume,
                taker_buy_trade_ratio = EXCLUDED.taker_buy_trade_ratio,
                taker_buy_base_ratio_trades = EXCLUDED.taker_buy_base_ratio_trades,
                taker_buy_quote_ratio_trades = EXCLUDED.taker_buy_quote_ratio_trades,
                avg_trade_quote_size = EXCLUDED.avg_trade_quote_size,
                trade_count_intensity = EXCLUDED.trade_count_intensity,
                quote_volume_intensity = EXCLUDED.quote_volume_intensity
            """,
            [
                (
                    row.exchange,
                    row.symbol,
                    row.timeframe,
                    row.open_time,
                    row.close_time,
                    row.close_price,
                    row.volume,
                    row.trade_count,
                    row.taker_buy_trade_count,
                    row.taker_sell_trade_count,
                    row.taker_buy_base_volume_trades,
                    row.taker_sell_base_volume_trades,
                    row.taker_buy_quote_volume_trades,
                    row.taker_sell_quote_volume_trades,
                    row.taker_net_base_volume,
                    row.taker_net_quote_volume,
                    row.taker_buy_trade_ratio,
                    row.taker_buy_base_ratio_trades,
                    row.taker_buy_quote_ratio_trades,
                    row.avg_trade_quote_size,
                    row.trade_count_intensity,
                    row.quote_volume_intensity,
                )
                for row in rows
            ],
        )
        return len(rows)

'''
        _replace_once(rel, old, method + old)


def patch_analyzer() -> None:
    rel = "scripts/analyze_market_features.py"
    text = _read(rel)
    if "trade_count_intensity" in text:
        print(f"already patched {rel}")
        return
    old = """        "taker_buy_ratio": _feature_attr("taker_buy_ratio"),
        # V23 live order-book features. NULL until enough snapshots are
"""
    new = """        "taker_buy_ratio": _feature_attr("taker_buy_ratio"),
        # V26 historical aggregate-trade pressure features.
        "trade_count_intensity": _feature_attr("trade_count_intensity"),
        "quote_volume_intensity": _feature_attr("quote_volume_intensity"),
        "taker_buy_trade_ratio": _feature_attr("taker_buy_trade_ratio"),
        "taker_buy_base_ratio_trades": _feature_attr("taker_buy_base_ratio_trades"),
        "taker_buy_quote_ratio_trades": _feature_attr("taker_buy_quote_ratio_trades"),
        "taker_net_base_volume": _feature_attr("taker_net_base_volume"),
        "taker_net_quote_volume": _feature_attr("taker_net_quote_volume"),
        "avg_trade_quote_size": _feature_attr("avg_trade_quote_size"),
        # V23 live order-book features. NULL until enough snapshots are
"""
    _replace_once(rel, old, new)


def patch_migrations() -> None:
    rel = "scripts/apply_migrations.py"
    if "006_trade_pressure_features.sql" not in _read(rel):
        old = """    collector_runs_migration_path = migrations_dir / "005_order_book_collector_runs.sql"
    timescale_migration_path = migrations_dir / "002_timescale_optional.sql"
"""
        new = """    collector_runs_migration_path = migrations_dir / "005_order_book_collector_runs.sql"
    trade_pressure_migration_path = migrations_dir / "006_trade_pressure_features.sql"
    timescale_migration_path = migrations_dir / "002_timescale_optional.sql"
"""
        _replace_once(rel, old, new)
        old2 = """        await db.apply_migration_file(collector_runs_migration_path)
        logger.info("db_migration_applied", migration=str(collector_runs_migration_path))

        if settings.database_use_timescaledb:
"""
        new2 = """        await db.apply_migration_file(collector_runs_migration_path)
        logger.info("db_migration_applied", migration=str(collector_runs_migration_path))

        await db.apply_migration_file(trade_pressure_migration_path)
        logger.info("db_migration_applied", migration=str(trade_pressure_migration_path))

        if settings.database_use_timescaledb:
"""
        _replace_once(rel, old2, new2)
    else:
        print(f"already patched {rel}")

    rel = "app/main.py"
    if "006_trade_pressure_features.sql" not in _read(rel):
        old = """        collector_runs_migration_path = migrations_dir / "005_order_book_collector_runs.sql"
        await db.apply_migration_file(collector_runs_migration_path)
        logger.info("db_migration_applied", migration=str(collector_runs_migration_path))

        if settings.database_use_timescaledb:
"""
        new = """        collector_runs_migration_path = migrations_dir / "005_order_book_collector_runs.sql"
        await db.apply_migration_file(collector_runs_migration_path)
        logger.info("db_migration_applied", migration=str(collector_runs_migration_path))

        trade_pressure_migration_path = migrations_dir / "006_trade_pressure_features.sql"
        await db.apply_migration_file(trade_pressure_migration_path)
        logger.info("db_migration_applied", migration=str(trade_pressure_migration_path))

        if settings.database_use_timescaledb:
"""
        _replace_once(rel, old, new)
    else:
        print(f"already patched {rel}")


def patch_readme() -> None:
    rel = "README.md"
    text = _read(rel)
    if "## V26 historical trade-pressure features" in text:
        print(f"already patched {rel}")
        return
    _path(rel).write_text(text.rstrip() + README_V26 + "\n", encoding="utf-8")
    print(f"patched {rel}")


def main() -> None:
    if not _path("pyproject.toml").exists() or not _path("app").exists():
        raise SystemExit("Run this from the trade-bot project root.")

    _write("app/market/trade_pressure.py", TRADE_PRESSURE)
    _write("app/storage/migrations/006_trade_pressure_features.sql", MIGRATION_006)
    _write("scripts/backfill_trade_pressure_features.py", BACKFILL_TRADE_PRESSURE)
    _write("scripts/compare_feature_groups.py", COMPARE_FEATURE_GROUPS)
    _write("tests/test_trade_pressure.py", TEST_TRADE_PRESSURE)

    patch_binance_rest()
    patch_features()
    patch_repositories()
    patch_analyzer()
    patch_migrations()
    patch_readme()

    print("\nV26 files applied. Now run:")
    print("  python -m scripts.apply_migrations")
    print("  python -m pytest -q")


if __name__ == "__main__":
    main()
