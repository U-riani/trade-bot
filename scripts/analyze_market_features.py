"""V22 Phase 3: test whether market features predict forward returns.

For each timeframe (5m, 15m) and each feature, this computes:
  * Pearson correlation with forward return over 1, 3, and 6 candles
  * quantile-bucket analysis: average forward return + win rate + sample size

Price-shape features (volume spike, body %, wick %) are derived from candles.
Non-price features (taker_buy_ratio, order_book_imbalance, spread_pct) come from
the market_features table. Order-book features are NULL historically, so they
report sample_size 0 here - which is the honest answer, not a fabricated one.

Example:
    python -m scripts.analyze_market_features --market-data-source production \
        --source db --limit 50000 --timeframes 5m,15m \
        --export-json reports/feature_analysis_v22.json \
        --export-csv reports/feature_analysis_v22.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.backtesting.feature_analysis import (
    FeatureHorizonAnalysis,
    analyze_feature,
    classify_feature_data_reason,
)
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import (
    candle_body_pct,
    candle_lower_wick_pct,
    candle_upper_wick_pct,
    volume_spike_ratios,
)
from app.market.models import Candle
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.backtest_strategy import _load_candles, _resolve_market_data_source

logger = get_logger(__name__)

HORIZONS = (1, 3, 6)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze predictive value of market features vs forward returns.")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--timeframes", default="5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--min-candles-per-timeframe", type=int, default=100)
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--volume-spike-lookback", type=int, default=20)
    parser.add_argument("--num-buckets", type=int, default=5)
    parser.add_argument(
        "--min-feature-samples",
        type=int,
        default=100,
        help="Warn that results are statistically useless below this sample size.",
    )
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _build_feature_series(
    candles: list[Candle],
    features_by_close_time: dict,
    *,
    volume_spike_lookback: int,
) -> dict[str, list[float | None]]:
    """Assemble all feature series aligned to the candle list (index-aligned)."""

    def _feature_attr(attr: str) -> list[float | None]:
        values: list[float | None] = []
        for candle in candles:
            row = features_by_close_time.get(candle.close_time)
            values.append(getattr(row, attr) if row is not None else None)
        return values

    return {
        "volume_spike_ratio": volume_spike_ratios(candles, volume_spike_lookback),
        "body_pct": [candle_body_pct(c) for c in candles],
        "upper_wick_pct": [candle_upper_wick_pct(c) for c in candles],
        "lower_wick_pct": [candle_lower_wick_pct(c) for c in candles],
        "taker_buy_ratio": _feature_attr("taker_buy_ratio"),
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
        # collected and aggregated, so these report sample_size 0 honestly until
        # the forward-looking dataset has accumulated.
        "order_book_imbalance": _feature_attr("order_book_imbalance"),
        "spread_pct": _feature_attr("spread_pct"),
        "imbalance_top_5": _feature_attr("imbalance_top_5"),
        "imbalance_top_10": _feature_attr("imbalance_top_10"),
        "imbalance_top_20": _feature_attr("imbalance_top_20"),
    }


def _analysis_payload(timeframe: str, analysis: FeatureHorizonAnalysis) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "feature": analysis.feature,
        "horizon": analysis.horizon,
        "sample_size": analysis.sample_size,
        "correlation": analysis.correlation,
        "buckets": [
            {
                "bucket": bucket.bucket,
                "lower": bucket.lower,
                "upper": bucket.upper,
                "count": bucket.count,
                "avg_forward_return_pct": bucket.avg_forward_return_pct,
                "win_rate": bucket.win_rate,
            }
            for bucket in analysis.buckets
        ],
    }


def _export_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("feature_analysis_json_exported", path=str(path), rows=len(payload))


def _export_csv(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timeframe",
        "feature",
        "horizon",
        "sample_size",
        "correlation",
        "bucket",
        "lower",
        "upper",
        "count",
        "avg_forward_return_pct",
        "win_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload:
            base = {
                "timeframe": item["timeframe"],
                "feature": item["feature"],
                "horizon": item["horizon"],
                "sample_size": item["sample_size"],
                "correlation": item["correlation"],
            }
            if not item["buckets"]:
                writer.writerow({**base, "bucket": "", "lower": "", "upper": "", "count": 0,
                                 "avg_forward_return_pct": "", "win_rate": ""})
                continue
            for bucket in item["buckets"]:
                writer.writerow({**base, **bucket})
    logger.info("feature_analysis_csv_exported", path=str(path), rows=len(payload))


def _parse_timeframes(value: str) -> list[str]:
    timeframes: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in timeframes:
            timeframes.append(item)
    if not timeframes:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return timeframes


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    source_candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not source_candles:
        raise SystemExit("No candles available for feature analysis")

    market_data_source, _use_testnet, exchange_id = _resolve_market_data_source(args.market_data_source)
    logger.info(
        "feature_analysis_started",
        source=args.source,
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        source_candles=len(source_candles),
        timeframes=args.timeframes,
        horizons=",".join(str(h) for h in HORIZONS),
    )

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)

    payload: list[dict[str, Any]] = []
    try:
        for timeframe in _parse_timeframes(args.timeframes):
            candles = resample_candles(
                source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe
            )
            if len(candles) < args.min_candles_per_timeframe:
                logger.warning("feature_analysis_timeframe_skipped", timeframe=timeframe, candles=len(candles))
                continue

            feature_rows = await repository.load_market_features(
                exchange=exchange_id,
                symbol=settings.normalized_symbol,
                timeframe=timeframe,
                limit=len(candles) + 10,
            )
            features_by_close_time = {row.close_time: row for row in feature_rows}
            logger.info(
                "feature_analysis_timeframe_ready",
                timeframe=timeframe,
                candles=len(candles),
                feature_rows=len(feature_rows),
            )

            series = _build_feature_series(
                candles, features_by_close_time, volume_spike_lookback=args.volume_spike_lookback
            )
            closes = [candle.close for candle in candles]

            for feature_name, values in series.items():
                present_count = sum(1 for value in values if value is not None)
                for horizon in HORIZONS:
                    analysis = analyze_feature(
                        feature_name, values, closes, horizon, num_buckets=args.num_buckets
                    )
                    payload.append(_analysis_payload(timeframe, analysis))

                    reason = classify_feature_data_reason(
                        feature_name=feature_name,
                        present_count=present_count,
                        sample_size=analysis.sample_size,
                        min_samples=args.min_feature_samples,
                    )
                    if reason is not None:
                        logger.info(
                            "feature_analysis_insufficient",
                            timeframe=timeframe,
                            feature=feature_name,
                            horizon=horizon,
                            present_count=present_count,
                            sample_size=analysis.sample_size,
                            reason=reason,
                        )
                        continue

                    best = max(analysis.buckets, key=lambda b: b.avg_forward_return_pct, default=None)
                    worst = min(analysis.buckets, key=lambda b: b.avg_forward_return_pct, default=None)
                    logger.info(
                        "feature_analysis_result",
                        timeframe=timeframe,
                        feature=feature_name,
                        horizon=horizon,
                        sample_size=analysis.sample_size,
                        correlation=None if analysis.correlation is None else round(analysis.correlation, 5),
                        top_bucket_avg_ret=None if best is None else round(best.avg_forward_return_pct, 5),
                        top_bucket_win_rate=None if best is None else round(best.win_rate, 4),
                        bottom_bucket_avg_ret=None if worst is None else round(worst.avg_forward_return_pct, 5),
                    )
    finally:
        await db.close()

    if args.export_json:
        _export_json(args.export_json, payload)
    if args.export_csv:
        _export_csv(args.export_csv, payload)

    logger.info("feature_analysis_finished", rows=len(payload))


if __name__ == "__main__":
    asyncio.run(main())
