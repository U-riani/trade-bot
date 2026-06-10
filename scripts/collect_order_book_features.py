"""V23/V25 live Binance order-book feature collector.

Polls Binance /api/v3/depth on an interval, derives spread + per-depth imbalance
(top 5/10/20), and appends each observation to order_book_snapshots. This is
DATA COLLECTION ONLY: no trading, no signals. Binance has no historical depth,
so this dataset must accumulate forward in time before any analysis is
meaningful.

V25 adds long-run reliability: exponential backoff through transient errors,
periodic heartbeat/stats, an optional consecutive-failure stop guard, and run
tracking in order_book_collector_runs (when a DB is connected). None of this is
trading logic; it only governs how reliably data is gathered over days/weeks.

Examples:
    python -m scripts.collect_order_book_features --symbol BTCUSDT --interval-seconds 5 --limit 100
    python -m scripts.collect_order_book_features --dry-run --max-snapshots 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from collections.abc import Sequence

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.market.collector_runtime import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    CollectorRun,
    apply_jitter,
    average_interval_seconds,
    compute_backoff_seconds,
    default_run_id,
    should_emit,
    should_stop_on_failures,
)
from app.market.order_book import snapshot_from_depth_response
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.time import utc_now
from scripts.backtest_strategy import _resolve_market_data_source

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect live Binance order-book snapshots (spread + imbalance) into PostgreSQL."
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol. Default BTCUSDT.")
    parser.add_argument("--interval-seconds", type=float, default=5.0, help="Seconds between polls. Default 5.")
    parser.add_argument("--limit", type=int, default=100, help="Depth levels to request. Default 100.")
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Stop after collecting this many snapshots. Default: run until Ctrl+C.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute and log only; no DB writes, no run tracking.")
    parser.add_argument("--exchange", default=None, help="Exchange id override. Default from --market-data-source.")
    parser.add_argument(
        "--market-data-source",
        choices=("production", "testnet"),
        default=None,
        help="Resolves the exchange id and which Binance host to poll. Default HISTORICAL_MARKET_DATA_SOURCE.",
    )
    # V25 reliability options.
    parser.add_argument("--run-id", default=None, help="Run id for tracking. Default: auto-generated.")
    parser.add_argument(
        "--heartbeat-every-seconds", type=float, default=60.0, help="Heartbeat log/run-update cadence. 0 disables."
    )
    parser.add_argument(
        "--max-failures", type=int, default=0, help="Stop after this many CONSECUTIVE failures. 0 disables (default)."
    )
    parser.add_argument("--backoff-initial-seconds", type=float, default=1.0, help="Initial backoff after a failure.")
    parser.add_argument("--backoff-max-seconds", type=float, default=60.0, help="Max backoff cap.")
    parser.add_argument("--jitter", action="store_true", help="Add randomized jitter to backoff sleeps.")
    parser.add_argument("--quiet-http-logs", action="store_true", help="Silence per-request httpx/httpcore logs.")
    parser.add_argument(
        "--stats-every-snapshots", type=int, default=0, help="Log a stats line every N snapshots. 0 disables."
    )
    return parser


async def _collect_once(
    *,
    client: BinanceRestClient,
    repository: TradingRepository | None,
    symbol: str,
    exchange: str,
    limit: int,
) -> bool:
    """Fetch one snapshot, store it (unless dry-run). Returns True on success."""
    depth = await client.get_order_book(symbol=symbol, limit=limit)
    snapshot = snapshot_from_depth_response(
        depth,
        exchange=exchange,
        symbol=symbol,
        collected_at=utc_now(),
        raw_depth_limit=limit,
    )
    if snapshot is None:
        logger.warning("order_book_snapshot_skipped", reason="empty_book", symbol=symbol)
        return False

    if repository is not None:
        await repository.insert_order_book_snapshots([snapshot])

    logger.info(
        "order_book_snapshot",
        symbol=symbol,
        spread_pct=round(snapshot.spread_pct, 6),
        imbalance_top_5=round(snapshot.imbalance_top_5, 5),
        imbalance_top_10=round(snapshot.imbalance_top_10, 5),
        imbalance_top_20=round(snapshot.imbalance_top_20, 5),
        stored=repository is not None,
    )
    return True


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _validate(args: argparse.Namespace) -> None:
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        raise SystemExit("--max-snapshots must be positive when set")
    if args.heartbeat_every_seconds < 0:
        raise SystemExit("--heartbeat-every-seconds cannot be negative")
    if args.max_failures < 0:
        raise SystemExit("--max-failures cannot be negative")
    if args.backoff_initial_seconds <= 0:
        raise SystemExit("--backoff-initial-seconds must be positive")
    if args.backoff_max_seconds < args.backoff_initial_seconds:
        raise SystemExit("--backoff-max-seconds must be >= --backoff-initial-seconds")
    if args.stats_every_snapshots < 0:
        raise SystemExit("--stats-every-snapshots cannot be negative")


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()
    _validate(args)

    if args.quiet_http_logs:
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)

    _market_data_source, use_testnet_data, default_exchange = _resolve_market_data_source(args.market_data_source)
    exchange = args.exchange or default_exchange
    symbol = args.symbol.upper().strip()
    run_id = default_run_id(args.run_id)

    db: Database | None = None
    repository: TradingRepository | None = None
    if not args.dry_run:
        db = Database(settings.database_url)
        await db.connect()
        repository = TradingRepository(db)

    client = BinanceRestClient(testnet=use_testnet_data)
    started_at = utc_now()

    logger.info(
        "order_book_collector_started",
        run_id=run_id,
        symbol=symbol,
        exchange=exchange,
        interval_seconds=args.interval_seconds,
        depth_limit=args.limit,
        max_snapshots=args.max_snapshots,
        max_failures=args.max_failures,
        heartbeat_every_seconds=args.heartbeat_every_seconds,
        dry_run=args.dry_run,
    )

    if repository is not None:
        await repository.start_collector_run(
            CollectorRun(
                run_id=run_id,
                exchange=exchange,
                symbol=symbol,
                started_at=started_at,
                stopped_at=None,
                status=STATUS_RUNNING,
                interval_seconds=args.interval_seconds,
                depth_limit=args.limit,
                collected_count=0,
                failure_count=0,
                last_snapshot_at=None,
                stop_reason=None,
            )
        )

    collected = 0
    total_failures = 0
    consecutive_failures = 0
    first_snapshot_at = None
    last_snapshot_at = None
    last_heartbeat_at = started_at
    status = STATUS_RUNNING
    stop_reason: str | None = None

    async def _persist(run_status: str, *, stopped_at=None, reason: str | None = None) -> None:
        if repository is not None:
            await repository.update_collector_run(
                run_id=run_id,
                status=run_status,
                collected_count=collected,
                failure_count=total_failures,
                last_snapshot_at=last_snapshot_at,
                stopped_at=stopped_at,
                stop_reason=reason,
            )

    try:
        while args.max_snapshots is None or collected < args.max_snapshots:
            try:
                ok = await _collect_once(
                    client=client, repository=repository, symbol=symbol, exchange=exchange, limit=args.limit
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001 - keep collecting through transient errors
                ok = False
                logger.warning(
                    "order_book_poll_failed",
                    run_id=run_id,
                    error=str(exc),
                    consecutive_failures=consecutive_failures + 1,
                )

            if ok:
                collected += 1
                consecutive_failures = 0
                now = utc_now()
                if first_snapshot_at is None:
                    first_snapshot_at = now
                last_snapshot_at = now
                if should_emit(collected, args.stats_every_snapshots):
                    logger.info(
                        "order_book_collector_stats",
                        run_id=run_id,
                        collected=collected,
                        failures=total_failures,
                        avg_interval_seconds=_round(
                            average_interval_seconds(first_snapshot_at, last_snapshot_at, collected)
                        ),
                    )
            else:
                total_failures += 1
                consecutive_failures += 1
                if should_stop_on_failures(consecutive_failures, args.max_failures):
                    status = STATUS_FAILED
                    stop_reason = f"max_consecutive_failures_reached:{consecutive_failures}"
                    logger.error(
                        "order_book_collector_failing",
                        run_id=run_id,
                        consecutive_failures=consecutive_failures,
                        max_failures=args.max_failures,
                    )
                    break

            now = utc_now()
            if args.heartbeat_every_seconds > 0 and (now - last_heartbeat_at).total_seconds() >= args.heartbeat_every_seconds:
                last_heartbeat_at = now
                table_total = (
                    await repository.count_order_book_snapshots(exchange=exchange, symbol=symbol)
                    if repository is not None
                    else None
                )
                logger.info(
                    "order_book_collector_heartbeat",
                    run_id=run_id,
                    collected=collected,
                    failures=total_failures,
                    last_snapshot_at=None if last_snapshot_at is None else last_snapshot_at.isoformat(),
                    avg_interval_seconds=_round(
                        average_interval_seconds(first_snapshot_at, last_snapshot_at, collected)
                    ),
                    table_total=table_total,
                )
                await _persist(STATUS_RUNNING)

            if args.max_snapshots is not None and collected >= args.max_snapshots:
                break

            if consecutive_failures > 0:
                backoff = compute_backoff_seconds(
                    consecutive_failures,
                    initial=args.backoff_initial_seconds,
                    maximum=args.backoff_max_seconds,
                )
                sleep_for = apply_jitter(backoff, random.random()) if args.jitter else backoff
            else:
                sleep_for = args.interval_seconds
            await asyncio.sleep(sleep_for)

        if status == STATUS_RUNNING:
            status = STATUS_STOPPED
            stop_reason = stop_reason or (
                "max_snapshots_reached" if args.max_snapshots is not None else "loop_ended"
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        status = STATUS_INTERRUPTED
        stop_reason = "keyboard_interrupt"
        logger.info("order_book_collector_interrupted", run_id=run_id)
    finally:
        await client.close()
        stopped_at = utc_now()
        if repository is not None and db is not None:
            await _persist(status, stopped_at=stopped_at, reason=stop_reason)
            total = await repository.count_order_book_snapshots(exchange=exchange, symbol=symbol)
            await db.close()
            logger.info(
                "order_book_collector_stopped",
                run_id=run_id,
                status=status,
                stop_reason=stop_reason,
                collected=collected,
                failures=total_failures,
                table_total=total,
            )
        else:
            logger.info(
                "order_book_collector_stopped",
                run_id=run_id,
                status=status,
                stop_reason=stop_reason,
                collected=collected,
                failures=total_failures,
                dry_run=True,
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("order_book_collector_keyboard_interrupt")
