"""V26 feature-group comparison lab.

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
