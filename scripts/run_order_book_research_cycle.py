"""V24 Phase 4: run the safe order-book research cycle in the right order.

The order-book pipeline only works when the steps run in sequence:
  1. backfill recent candles      (so snapshot buckets have closed candles)
  2. aggregate order-book snapshots (join snapshots to candle buckets)
  3. pipeline status               (report what is / isn't ready)
  4. analyze (optional)            (only meaningful once samples accumulate)

This helper orchestrates that sequence and, in --dry-run mode, prints the exact
commands without executing them. It adds NO trading logic; it only shells out to
the existing read/aggregate/analyze scripts.

Example:
    python -m scripts.run_order_book_research_cycle --market-data-source production \
        --symbol BTCUSDT --timeframes 1m,5m,15m --analyze
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence

from app.config.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_cycle_commands(
    *,
    market_data_source: str,
    symbol: str,
    timeframes: str,
    backfill_limit: int,
    candle_limit: int,
    min_feature_samples: int,
    analyze: bool,
    analyze_limit: int,
    python_exe: str = sys.executable,
) -> list[tuple[str, list[str]]]:
    """Build the ordered (label, argv) command list for the research cycle.

    Pure: no execution, no I/O. Returned argv lists are exactly what would run,
    so this is what the tests assert against and what --dry-run prints.
    """
    commands: list[tuple[str, list[str]]] = [
        (
            "backfill_candles",
            [python_exe, "-m", "scripts.backfill_candles",
             "--market-data-source", market_data_source, "--limit", str(backfill_limit)],
        ),
        (
            "aggregate_order_book_features",
            [python_exe, "-m", "scripts.aggregate_order_book_features",
             "--market-data-source", market_data_source, "--source", "db",
             "--symbol", symbol, "--candle-limit", str(candle_limit), "--timeframes", timeframes],
        ),
        (
            "order_book_pipeline_status",
            [python_exe, "-m", "scripts.order_book_pipeline_status",
             "--market-data-source", market_data_source, "--symbol", symbol,
             "--timeframes", timeframes, "--candle-limit", str(candle_limit),
             "--min-feature-samples", str(min_feature_samples)],
        ),
    ]
    if analyze:
        commands.append(
            (
                "analyze_market_features",
                [python_exe, "-m", "scripts.analyze_market_features",
                 "--market-data-source", market_data_source, "--source", "db",
                 "--limit", str(analyze_limit), "--timeframes", timeframes,
                 "--min-feature-samples", str(min_feature_samples)],
            )
        )
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the order-book research cycle: backfill -> aggregate -> status -> analyze.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default="production")
    parser.add_argument("--backfill-limit", type=int, default=200)
    parser.add_argument("--candle-limit", type=int, default=5000)
    parser.add_argument("--analyze-limit", type=int, default=50000)
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--analyze", action="store_true", help="Also run analyze_market_features at the end.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands without executing them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _parser().parse_args(argv)

    commands = build_cycle_commands(
        market_data_source=args.market_data_source,
        symbol=args.symbol.upper().strip(),
        timeframes=args.timeframes,
        backfill_limit=args.backfill_limit,
        candle_limit=args.candle_limit,
        min_feature_samples=args.min_feature_samples,
        analyze=args.analyze,
        analyze_limit=args.analyze_limit,
    )

    logger.info("research_cycle_started", steps=len(commands), dry_run=args.dry_run, analyze=args.analyze)
    for label, argv_cmd in commands:
        printable = shlex.join(argv_cmd)
        if args.dry_run:
            logger.info("research_cycle_command", step=label, command=printable, executed=False)
            continue

        logger.info("research_cycle_running", step=label, command=printable)
        result = subprocess.run(argv_cmd, check=False)  # noqa: S603 - args are built internally, not user shell
        logger.info("research_cycle_step_done", step=label, return_code=result.returncode)
        if result.returncode != 0:
            logger.warning("research_cycle_step_failed", step=label, return_code=result.returncode)
            return result.returncode

    logger.info("research_cycle_finished", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
