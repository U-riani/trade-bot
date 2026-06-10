"""V23 Phase 2: live Binance order-book feature collector.

Polls Binance /api/v3/depth on an interval, derives spread + per-depth imbalance
(top 5/10/20), and appends each observation to order_book_snapshots. This is
DATA COLLECTION ONLY: no trading, no signals. Binance has no historical depth,
so this dataset must accumulate forward in time before any analysis is
meaningful.

Example:
    python -m scripts.collect_order_book_features --symbol BTCUSDT --interval-seconds 5 --limit 100

Dry-run (no DB writes), stop after 3 snapshots:
    python -m scripts.collect_order_book_features --dry-run --max-snapshots 3
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
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
    parser.add_argument("--dry-run", action="store_true", help="Compute and log only; do not write to the DB.")
    parser.add_argument("--exchange", default=None, help="Exchange id override. Default from --market-data-source.")
    parser.add_argument(
        "--market-data-source",
        choices=("production", "testnet"),
        default=None,
        help="Resolves the exchange id and which Binance host to poll. Default HISTORICAL_MARKET_DATA_SOURCE.",
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


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        raise SystemExit("--max-snapshots must be positive when set")

    _market_data_source, use_testnet_data, default_exchange = _resolve_market_data_source(args.market_data_source)
    exchange = args.exchange or default_exchange
    symbol = args.symbol.upper().strip()

    db: Database | None = None
    repository: TradingRepository | None = None
    if not args.dry_run:
        db = Database(settings.database_url)
        await db.connect()
        repository = TradingRepository(db)

    client = BinanceRestClient(testnet=use_testnet_data)

    logger.info(
        "order_book_collector_started",
        symbol=symbol,
        exchange=exchange,
        interval_seconds=args.interval_seconds,
        depth_limit=args.limit,
        max_snapshots=args.max_snapshots,
        dry_run=args.dry_run,
    )

    collected = 0
    failures = 0
    try:
        while args.max_snapshots is None or collected < args.max_snapshots:
            try:
                ok = await _collect_once(
                    client=client,
                    repository=repository,
                    symbol=symbol,
                    exchange=exchange,
                    limit=args.limit,
                )
                if ok:
                    collected += 1
                else:
                    failures += 1
            except Exception as exc:  # noqa: BLE001 - keep collecting through transient errors
                failures += 1
                logger.warning("order_book_poll_failed", error=str(exc))

            if args.max_snapshots is not None and collected >= args.max_snapshots:
                break
            await asyncio.sleep(args.interval_seconds)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("order_book_collector_interrupted")
    finally:
        await client.close()
        if db is not None:
            total = await repository.count_order_book_snapshots(exchange=exchange, symbol=symbol)
            await db.close()
            logger.info("order_book_collector_stopped", collected=collected, failures=failures, table_total=total)
        else:
            logger.info("order_book_collector_stopped", collected=collected, failures=failures, dry_run=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("order_book_collector_keyboard_interrupt")
