"""Audit order-book freshness at V29.1 price-only pullback setups.

The script deliberately does *not* test PnL. It measures whether order-book
confirmation can be evaluated at the moments where a valid 15m-trend/5m-
pullback setup occurs. It uses a backward-looking as-of join and reports
staleness instead of silently filling sparse depth values.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.backtesting.multitimeframe_pullback_strategy import MultiTimeframePullbackConfig, build_pullback_setup_cache
from app.backtesting.order_book_alignment import alignment_records, alignment_summary
from app.backtesting.order_book_strategy import rows_with_feature
from app.config.logging import configure_logging, get_logger
from scripts.backtest_multitimeframe_pullback_strategy import _coverage_split, _load_price_timelines

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AuditSegment:
    name: str
    entry_rows: list


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit order-book freshness at multi-timeframe pullback setups.")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--order-book-features", default="imbalance_top_20,imbalance_top_5")
    parser.add_argument("--train-ratio", type=Decimal, default=Decimal("0.7"))
    parser.add_argument("--max-age-seconds", default="30,60,120,180,300")
    parser.add_argument("--max-pair-gap-seconds", default="60,120,180,300")
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _csv_values(value: str) -> list[str]:
    values: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if item and item not in values:
            values.append(item)
    if not values:
        raise SystemExit("argument must contain at least one item")
    return values


def _positive_ints(value: str, *, argument: str) -> list[int]:
    values = [int(item) for item in _csv_values(value)]
    if any(item <= 0 for item in values):
        raise SystemExit(f"{argument} values must be positive integers")
    return values


def _payload(
    *,
    feature: str,
    segment: str,
    coverage: Any,
    summary: dict[str, object],
    records: list,
) -> dict[str, object]:
    return {
        "feature": feature,
        "segment": segment,
        "coverage": {
            "start": coverage.coverage_start.isoformat(),
            "split_time": coverage.split_time.isoformat(),
            "end": coverage.coverage_end.isoformat(),
            "train_feature_samples": coverage.train_feature_samples,
            "validation_feature_samples": coverage.validation_feature_samples,
        },
        "summary": summary,
        "records": [record.to_payload() for record in records],
    }


def _export_csv(path: Path, payloads: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "feature", "segment", "price_setup_candidates", "exact_observation_at_setup", "has_any_asof_observation",
        "has_observed_pair", "fresh_60", "fresh_120", "fresh_180", "fresh_300",
        "eligible_age120_gap120", "eligible_age180_gap180", "eligible_age300_gap300",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            summary = payload["summary"]
            assert isinstance(summary, dict)
            ages = summary["age_seconds"]
            eligible = summary["eligible_pair_by_age_and_gap"]
            assert isinstance(ages, dict) and isinstance(eligible, dict)
            writer.writerow(
                {
                    "feature": payload["feature"],
                    "segment": payload["segment"],
                    "price_setup_candidates": summary["price_setup_candidates"],
                    "exact_observation_at_setup": summary["exact_observation_at_setup"],
                    "has_any_asof_observation": summary["has_any_asof_observation"],
                    "has_observed_pair": summary["has_observed_pair"],
                    "fresh_60": ages.get("60", ""),
                    "fresh_120": ages.get("120", ""),
                    "fresh_180": ages.get("180", ""),
                    "fresh_300": ages.get("300", ""),
                    "eligible_age120_gap120": eligible.get("age<=120_pair_gap<=120", ""),
                    "eligible_age180_gap180": eligible.get("age<=180_pair_gap<=180", ""),
                    "eligible_age300_gap300": eligible.get("age<=300_pair_gap<=300", ""),
                }
            )
    logger.info("order_book_alignment_csv_exported", path=str(path), rows=len(payloads))


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    if args.limit <= 0 or args.min_feature_samples <= 0:
        raise SystemExit("limit and min-feature-samples must be positive")
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise SystemExit("train-ratio must be between 0 and 1")

    features = _csv_values(args.order_book_features)
    max_age_seconds = _positive_ints(args.max_age_seconds, argument="max-age-seconds")
    max_pair_gap_seconds = _positive_ints(args.max_pair_gap_seconds, argument="max-pair-gap-seconds")
    entry_rows, pullback_rows, trend_rows, _symbol = await _load_price_timelines(args)

    outputs: list[dict[str, object]] = []
    for feature in features:
        coverage = _coverage_split(entry_rows, feature=feature, train_ratio=args.train_ratio)
        if coverage.train_feature_samples < max(1, args.min_feature_samples // 2):
            logger.info("order_book_alignment_feature_skipped", feature=feature, reason="not_enough_train_feature_samples")
            continue

        setup_config = MultiTimeframePullbackConfig(
            feature_name=feature,
            reversal_threshold=0.0,
            horizon_bars=5,
            strategy_name="order_book_alignment_audit",
            require_order_book_reversal=False,
        )
        segments = {
            "full": coverage.full_rows,
            "train": coverage.train_rows,
            "validation": coverage.validation_rows,
        }
        for segment, rows in segments.items():
            setups = build_pullback_setup_cache(
                entry_rows=rows,
                pullback_rows=pullback_rows,
                trend_rows=trend_rows,
                config=setup_config,
            )
            records = alignment_records(
                segment=segment,
                entry_rows=rows,
                setups=setups,
                feature_name=feature,
                observed_feature_rows=coverage.full_rows,
            )
            summary = alignment_summary(
                records,
                max_age_seconds=max_age_seconds,
                max_pair_gap_seconds=max_pair_gap_seconds,
            )
            logger.info(
                "order_book_alignment_segment",
                feature=feature,
                segment=segment,
                price_setup_candidates=summary["price_setup_candidates"],
                exact=summary["exact_observation_at_setup"],
                asof=summary["has_any_asof_observation"],
                observed_pairs=summary["has_observed_pair"],
                fresh_120=summary["age_seconds"].get("120"),
                eligible_age120_gap120=summary["eligible_pair_by_age_and_gap"].get("age<=120_pair_gap<=120"),
            )
            outputs.append(_payload(feature=feature, segment=segment, coverage=coverage, summary=summary, records=records))

    if not outputs:
        raise SystemExit("No alignment audit rows produced. Check order-book feature coverage.")
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        logger.info("order_book_alignment_json_exported", path=str(args.export_json), rows=len(outputs))
    if args.export_csv:
        _export_csv(args.export_csv, outputs)


if __name__ == "__main__":
    asyncio.run(main())
