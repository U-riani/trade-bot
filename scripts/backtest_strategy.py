from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import BacktestResult
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.strategy.ema_rsi import EmaRsiStrategy

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local paper backtest for the configured EMA/RSI strategy.")
    parser.add_argument("--limit", type=int, default=500, help="Number of recent candles to use.")
    parser.add_argument(
        "--source",
        choices=("auto", "db", "rest"),
        default="auto",
        help="Candle source. auto tries DB first, then REST.",
    )
    parser.add_argument("--show-trades", action="store_true", help="Print executed round trips.")
    parser.add_argument(
        "--market-data-source",
        choices=("production", "testnet"),
        default=None,
        help=(
            "Historical candle source for DB exchange key and REST fallback. "
            "Defaults to HISTORICAL_MARKET_DATA_SOURCE. Use production for realistic research."
        ),
    )
    parser.add_argument(
        "--fee-rate-pct",
        type=Decimal,
        default=None,
        help="Trading fee percent per side. Defaults to BACKTEST_FEE_RATE_PCT.",
    )
    parser.add_argument(
        "--slippage-pct",
        type=Decimal,
        default=None,
        help="Simulated slippage percent per order. Defaults to BACKTEST_SLIPPAGE_PCT.",
    )
    parser.add_argument("--export-json", type=Path, default=None, help="Export backtest result as JSON.")
    parser.add_argument("--export-csv", type=Path, default=None, help="Export completed trades as CSV.")
    return parser


def _resolve_market_data_source(source_override: str | None = None) -> tuple[str, bool, str]:
    settings = get_settings()
    market_data_source = source_override or settings.historical_market_data_source.value
    use_testnet_data = market_data_source == "testnet"
    exchange_id = "binance_testnet" if use_testnet_data else "binance_spot"
    return market_data_source, use_testnet_data, exchange_id


async def _load_from_db(limit: int, market_data_source_override: str | None = None):
    settings = get_settings()
    if not settings.database_enabled:
        logger.warning("backtest_db_skipped", reason="DATABASE_ENABLED=false")
        return []

    db = Database(settings.database_url)
    await db.connect()
    try:
        repository = TradingRepository(db)
        market_data_source, _use_testnet_data, exchange_id = _resolve_market_data_source(
            market_data_source_override
        )
        candles = await repository.load_recent_candles(
            exchange=exchange_id,
            symbol=settings.normalized_symbol,
            timeframe=settings.timeframe,
            limit=limit,
        )
        logger.info(
            "backtest_db_candles_loaded",
            count=len(candles),
            limit=limit,
            market_data_source=market_data_source,
            exchange_id=exchange_id,
        )
        return candles
    finally:
        await db.close()


async def _load_from_rest(limit: int, market_data_source_override: str | None = None):
    settings = get_settings()
    market_data_source, use_testnet_data, exchange_id = _resolve_market_data_source(
        market_data_source_override
    )
    client = BinanceRestClient(testnet=use_testnet_data)
    try:
        candles = await client.get_historical_closed_candles(
            symbol=settings.normalized_symbol,
            timeframe=settings.timeframe,
            limit=limit,
            exchange=exchange_id,
        )
        logger.info(
            "backtest_rest_candles_loaded",
            count=len(candles),
            limit=limit,
            market_data_source=market_data_source,
            exchange_id=exchange_id,
        )
        return candles
    finally:
        await client.close()


async def _load_candles(source: str, limit: int, market_data_source_override: str | None = None):
    if source == "db":
        return await _load_from_db(limit, market_data_source_override)
    if source == "rest":
        return await _load_from_rest(limit, market_data_source_override)

    candles = await _load_from_db(limit, market_data_source_override)
    if candles:
        return candles
    return await _load_from_rest(limit, market_data_source_override)


def _decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _result_payload(result: BacktestResult) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "metrics": {
            "candles_processed": metrics.candles_processed,
            "executed_orders": metrics.executed_orders,
            "round_trips": metrics.round_trips,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": metrics.win_rate,
            "realized_pnl": _decimal_to_str(metrics.realized_pnl),
            "unrealized_pnl": _decimal_to_str(metrics.unrealized_pnl),
            "total_fees": _decimal_to_str(metrics.total_fees),
            "max_drawdown": _decimal_to_str(metrics.max_drawdown),
            "initial_equity": _decimal_to_str(metrics.initial_equity),
            "final_equity": _decimal_to_str(metrics.final_equity),
            "return_pct": _decimal_to_str(metrics.return_pct),
            "has_open_position": metrics.has_open_position,
            "open_position_quantity": _decimal_to_str(metrics.open_position_quantity),
            "open_position_avg_entry_price": _decimal_to_str(metrics.open_position_avg_entry_price),
            "last_price": _decimal_to_str(metrics.last_price),
        },
        "trades": [
            {
                "symbol": trade.symbol,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "entry_price": _decimal_to_str(trade.entry_price),
                "exit_price": _decimal_to_str(trade.exit_price),
                "quantity": _decimal_to_str(trade.quantity),
                "quote_amount": _decimal_to_str(trade.quote_amount),
                "entry_fee": _decimal_to_str(trade.entry_fee),
                "exit_fee": _decimal_to_str(trade.exit_fee),
                "total_fees": _decimal_to_str(trade.total_fees),
                "pnl": _decimal_to_str(trade.pnl),
                "return_pct": _decimal_to_str(trade.return_pct),
                "entry_reason": trade.entry_reason,
                "exit_reason": trade.exit_reason,
            }
            for trade in result.trades
        ],
    }


def _export_json(path: Path, result: BacktestResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_result_payload(result), indent=2), encoding="utf-8")
    logger.info("backtest_json_exported", path=str(path))


def _export_csv(path: Path, result: BacktestResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "symbol",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "quote_amount",
                "entry_fee",
                "exit_fee",
                "total_fees",
                "pnl",
                "return_pct",
                "entry_reason",
                "exit_reason",
            ],
        )
        writer.writeheader()
        for trade in result.trades:
            writer.writerow(
                {
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "entry_price": str(trade.entry_price),
                    "exit_price": str(trade.exit_price),
                    "quantity": str(trade.quantity),
                    "quote_amount": str(trade.quote_amount),
                    "entry_fee": str(trade.entry_fee),
                    "exit_fee": str(trade.exit_fee),
                    "total_fees": str(trade.total_fees),
                    "pnl": str(trade.pnl),
                    "return_pct": str(trade.return_pct),
                    "entry_reason": trade.entry_reason,
                    "exit_reason": trade.exit_reason,
                }
            )
    logger.info("backtest_csv_exported", path=str(path), trades=len(result.trades))


def _run_backtest(candles, args: argparse.Namespace) -> None:
    settings = get_settings()
    strategy = EmaRsiStrategy(
        fast_period=settings.ema_fast_period,
        slow_period=settings.ema_slow_period,
        rsi_period=settings.rsi_period,
        rsi_buy_min=settings.rsi_buy_min,
        rsi_buy_max=settings.rsi_buy_max,
        rsi_sell_min=settings.rsi_sell_min,
        suggested_quote_amount=settings.max_order_usdt,
        trend_ema_period=settings.trend_ema_period,
        min_ema_gap_pct=settings.min_ema_gap_pct,
        atr_period=settings.atr_period,
        min_atr_pct=settings.min_atr_pct,
    )
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    engine = BacktestEngine(
        strategy=strategy,
        symbol=settings.normalized_symbol,
        initial_quote_balance=settings.initial_quote_balance,
        max_order_usdt=settings.max_order_usdt,
        max_position_usdt=settings.max_position_usdt,
        stop_loss_pct=settings.stop_loss_pct,
        take_profit_pct=settings.take_profit_pct,
        allow_only_one_open_position=settings.allow_only_one_open_position,
        fee_rate_pct=fee_rate_pct,
        slippage_pct=slippage_pct,
    )
    result = engine.run(candles)
    metrics = result.metrics

    logger.info(
        "backtest_result",
        candles_processed=metrics.candles_processed,
        executed_orders=metrics.executed_orders,
        round_trips=metrics.round_trips,
        winning_trades=metrics.winning_trades,
        losing_trades=metrics.losing_trades,
        win_rate=round(metrics.win_rate, 4),
        realized_pnl=str(metrics.realized_pnl),
        unrealized_pnl=str(metrics.unrealized_pnl),
        total_fees=str(metrics.total_fees),
        max_drawdown=str(metrics.max_drawdown),
        initial_equity=str(metrics.initial_equity),
        final_equity=str(metrics.final_equity),
        return_pct=str(metrics.return_pct),
        has_open_position=metrics.has_open_position,
        open_position_quantity=str(metrics.open_position_quantity),
        open_position_avg_entry_price=str(metrics.open_position_avg_entry_price),
        last_price=str(metrics.last_price),
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
    )

    if args.show_trades:
        for index, trade in enumerate(result.trades, start=1):
            logger.info(
                "backtest_trade",
                index=index,
                symbol=trade.symbol,
                entry_time=trade.entry_time.isoformat(),
                exit_time=trade.exit_time.isoformat(),
                entry_price=str(trade.entry_price),
                exit_price=str(trade.exit_price),
                quantity=str(trade.quantity),
                quote_amount=str(trade.quote_amount),
                entry_fee=str(trade.entry_fee),
                exit_fee=str(trade.exit_fee),
                total_fees=str(trade.total_fees),
                pnl=str(trade.pnl),
                return_pct=str(trade.return_pct),
                entry_reason=trade.entry_reason,
                exit_reason=trade.exit_reason,
            )

    if args.export_json:
        _export_json(args.export_json, result)
    if args.export_csv:
        _export_csv(args.export_csv, result)


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = _parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("--fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("--slippage-pct cannot be negative")

    candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not candles:
        raise SystemExit("No candles available for backtest")

    _run_backtest(candles, args)


if __name__ == "__main__":
    asyncio.run(main())
