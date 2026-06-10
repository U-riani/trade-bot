from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.backtesting.optimizer import (
    OptimizationResult,
    generate_parameter_sets,
    optimize_parameter_grid,
)
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from scripts.backtest_strategy import _decimal_to_str, _load_candles, _resolve_market_data_source

logger = get_logger(__name__)


DEFAULT_EMA_FAST_VALUES = "5,9,12"
DEFAULT_EMA_SLOW_VALUES = "21,34"
DEFAULT_RSI_PERIOD_VALUES = "14"
DEFAULT_RSI_BUY_MIN_VALUES = "45"
DEFAULT_RSI_BUY_MAX_VALUES = "65,70"
DEFAULT_RSI_SELL_MIN_VALUES = "70,75"
DEFAULT_STOP_LOSS_VALUES = "0.5,0.7,1.0"
DEFAULT_TAKE_PROFIT_VALUES = "0.8,1.2,1.8"
DEFAULT_TREND_EMA_VALUES = "0,200"
DEFAULT_MIN_EMA_GAP_VALUES = "0,0.03,0.06"
DEFAULT_ATR_PERIOD_VALUES = "0,14"
DEFAULT_MIN_ATR_PCT_VALUES = "0,0.08"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize EMA/RSI strategy parameters against local candles."
    )
    parser.add_argument("--limit", type=int, default=10000, help="Number of recent candles to use.")
    parser.add_argument(
        "--source",
        choices=("auto", "db", "rest"),
        default="db",
        help="Candle source. Use db after running scripts.backfill_candles.",
    )
    parser.add_argument(
        "--market-data-source",
        choices=("production", "testnet"),
        default=None,
        help="Historical candle source. Defaults to HISTORICAL_MARKET_DATA_SOURCE.",
    )
    parser.add_argument(
        "--ema-fast-values",
        default=DEFAULT_EMA_FAST_VALUES,
        help="Comma-separated EMA fast periods.",
    )
    parser.add_argument(
        "--ema-slow-values",
        default=DEFAULT_EMA_SLOW_VALUES,
        help="Comma-separated EMA slow periods.",
    )
    parser.add_argument(
        "--rsi-period-values",
        default=DEFAULT_RSI_PERIOD_VALUES,
        help="Comma-separated RSI periods.",
    )
    parser.add_argument(
        "--rsi-buy-min-values",
        default=DEFAULT_RSI_BUY_MIN_VALUES,
        help="Comma-separated RSI buy minimum values.",
    )
    parser.add_argument(
        "--rsi-buy-max-values",
        default=DEFAULT_RSI_BUY_MAX_VALUES,
        help="Comma-separated RSI buy maximum values.",
    )
    parser.add_argument(
        "--rsi-sell-min-values",
        default=DEFAULT_RSI_SELL_MIN_VALUES,
        help="Comma-separated RSI sell threshold values.",
    )
    parser.add_argument(
        "--stop-loss-pct-values",
        default=DEFAULT_STOP_LOSS_VALUES,
        help="Comma-separated stop-loss percent values.",
    )
    parser.add_argument(
        "--take-profit-pct-values",
        default=DEFAULT_TAKE_PROFIT_VALUES,
        help="Comma-separated take-profit percent values.",
    )
    parser.add_argument(
        "--trend-ema-values",
        default=DEFAULT_TREND_EMA_VALUES,
        help="Comma-separated trend EMA periods. Use 0 to disable the filter.",
    )
    parser.add_argument(
        "--min-ema-gap-pct-values",
        default=DEFAULT_MIN_EMA_GAP_VALUES,
        help="Comma-separated minimum EMA gap percent values. Use 0 to disable the filter.",
    )
    parser.add_argument(
        "--atr-period-values",
        default=DEFAULT_ATR_PERIOD_VALUES,
        help="Comma-separated ATR periods. Use 0 to disable ATR filtering.",
    )
    parser.add_argument(
        "--min-atr-pct-values",
        default=DEFAULT_MIN_ATR_PCT_VALUES,
        help="Comma-separated minimum ATR percent values. Requires a non-zero ATR period.",
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
    parser.add_argument(
        "--min-round-trips",
        type=int,
        default=10,
        help="Discard parameter sets with fewer completed trades.",
    )
    parser.add_argument("--top", type=int, default=20, help="How many ranked results to log.")
    parser.add_argument("--export-json", type=Path, default=None, help="Export all results as JSON.")
    parser.add_argument("--export-csv", type=Path, default=None, help="Export all results as CSV.")
    return parser


def _split_values(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    return values


def _parse_int_list(raw: str, *, field_name: str) -> list[int]:
    try:
        values = [int(item) for item in _split_values(raw)]
    except ValueError as exc:
        raise SystemExit(f"{field_name} must be a comma-separated integer list") from exc

    if any(value <= 0 for value in values):
        raise SystemExit(f"{field_name} values must be positive")
    return sorted(set(values))


def _parse_float_list(raw: str, *, field_name: str) -> list[float]:
    try:
        values = [float(item) for item in _split_values(raw)]
    except ValueError as exc:
        raise SystemExit(f"{field_name} must be a comma-separated number list") from exc

    if any(value < 0 or value > 100 for value in values):
        raise SystemExit(f"{field_name} values must be between 0 and 100")
    return sorted(set(values))


def _parse_decimal_list(raw: str, *, field_name: str) -> list[Decimal]:
    try:
        values = [Decimal(item) for item in _split_values(raw)]
    except InvalidOperation as exc:
        raise SystemExit(f"{field_name} must be a comma-separated decimal list") from exc

    if any(value <= 0 for value in values):
        raise SystemExit(f"{field_name} values must be positive")
    return sorted(set(values))


def _parse_optional_int_list(raw: str, *, field_name: str) -> list[int | None]:
    try:
        values = [int(item) for item in _split_values(raw)]
    except ValueError as exc:
        raise SystemExit(f"{field_name} must be a comma-separated integer list") from exc

    if any(value < 0 for value in values):
        raise SystemExit(f"{field_name} values cannot be negative")
    return [None if value == 0 else value for value in sorted(set(values))]


def _parse_non_negative_decimal_list(raw: str, *, field_name: str) -> list[Decimal]:
    try:
        values = [Decimal(item) for item in _split_values(raw)]
    except InvalidOperation as exc:
        raise SystemExit(f"{field_name} must be a comma-separated decimal list") from exc

    if any(value < 0 for value in values):
        raise SystemExit(f"{field_name} values cannot be negative")
    return sorted(set(values))


def _result_payload(item: OptimizationResult) -> dict[str, Any]:
    metrics = item.metrics
    parameters = item.parameters
    return {
        "rank": item.rank,
        "score": _decimal_to_str(item.score),
        "parameters": {
            "key": parameters.key,
            "ema_fast_period": parameters.ema_fast_period,
            "ema_slow_period": parameters.ema_slow_period,
            "rsi_period": parameters.rsi_period,
            "rsi_buy_min": parameters.rsi_buy_min,
            "rsi_buy_max": parameters.rsi_buy_max,
            "rsi_sell_min": parameters.rsi_sell_min,
            "stop_loss_pct": _decimal_to_str(parameters.stop_loss_pct),
            "take_profit_pct": _decimal_to_str(parameters.take_profit_pct),
            "trend_ema_period": parameters.trend_ema_period,
            "min_ema_gap_pct": _decimal_to_str(parameters.min_ema_gap_pct),
            "atr_period": parameters.atr_period,
            "min_atr_pct": _decimal_to_str(parameters.min_atr_pct),
        },
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
            "open_position_avg_entry_price": _decimal_to_str(
                metrics.open_position_avg_entry_price
            ),
            "last_price": _decimal_to_str(metrics.last_price),
        },
    }


def _export_json(path: Path, results: list[OptimizationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([_result_payload(item) for item in results], indent=2),
        encoding="utf-8",
    )
    logger.info("strategy_optimization_json_exported", path=str(path), rows=len(results))


def _export_csv(path: Path, results: list[OptimizationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "rank",
            "score",
            "key",
            "ema_fast_period",
            "ema_slow_period",
            "rsi_period",
            "rsi_buy_min",
            "rsi_buy_max",
            "rsi_sell_min",
            "stop_loss_pct",
            "take_profit_pct",
            "trend_ema_period",
            "min_ema_gap_pct",
            "atr_period",
            "min_atr_pct",
            "final_equity",
            "return_pct",
            "max_drawdown",
            "win_rate",
            "round_trips",
            "executed_orders",
            "winning_trades",
            "losing_trades",
            "total_fees",
            "realized_pnl",
            "unrealized_pnl",
            "has_open_position",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            parameters = item.parameters
            metrics = item.metrics
            writer.writerow(
                {
                    "rank": item.rank,
                    "score": str(item.score),
                    "key": parameters.key,
                    "ema_fast_period": parameters.ema_fast_period,
                    "ema_slow_period": parameters.ema_slow_period,
                    "rsi_period": parameters.rsi_period,
                    "rsi_buy_min": parameters.rsi_buy_min,
                    "rsi_buy_max": parameters.rsi_buy_max,
                    "rsi_sell_min": parameters.rsi_sell_min,
                    "stop_loss_pct": str(parameters.stop_loss_pct),
                    "take_profit_pct": str(parameters.take_profit_pct),
                    "trend_ema_period": parameters.trend_ema_period or 0,
                    "min_ema_gap_pct": str(parameters.min_ema_gap_pct),
                    "atr_period": parameters.atr_period or 0,
                    "min_atr_pct": str(parameters.min_atr_pct),
                    "final_equity": str(metrics.final_equity),
                    "return_pct": str(metrics.return_pct),
                    "max_drawdown": str(metrics.max_drawdown),
                    "win_rate": metrics.win_rate,
                    "round_trips": metrics.round_trips,
                    "executed_orders": metrics.executed_orders,
                    "winning_trades": metrics.winning_trades,
                    "losing_trades": metrics.losing_trades,
                    "total_fees": str(metrics.total_fees),
                    "realized_pnl": str(metrics.realized_pnl),
                    "unrealized_pnl": str(metrics.unrealized_pnl),
                    "has_open_position": metrics.has_open_position,
                }
            )
    logger.info("strategy_optimization_csv_exported", path=str(path), rows=len(results))


def _log_result(item: OptimizationResult) -> None:
    parameters = item.parameters
    metrics = item.metrics
    logger.info(
        "strategy_optimization_result",
        rank=item.rank,
        score=str(item.score),
        key=parameters.key,
        ema_fast_period=parameters.ema_fast_period,
        ema_slow_period=parameters.ema_slow_period,
        rsi_period=parameters.rsi_period,
        rsi_buy_min=parameters.rsi_buy_min,
        rsi_buy_max=parameters.rsi_buy_max,
        rsi_sell_min=parameters.rsi_sell_min,
        stop_loss_pct=str(parameters.stop_loss_pct),
        take_profit_pct=str(parameters.take_profit_pct),
        trend_ema_period=parameters.trend_ema_period or 0,
        min_ema_gap_pct=str(parameters.min_ema_gap_pct),
        atr_period=parameters.atr_period or 0,
        min_atr_pct=str(parameters.min_atr_pct),
        final_equity=str(metrics.final_equity),
        return_pct=str(metrics.return_pct),
        max_drawdown=str(metrics.max_drawdown),
        win_rate=round(metrics.win_rate, 4),
        round_trips=metrics.round_trips,
        total_fees=str(metrics.total_fees),
        has_open_position=metrics.has_open_position,
    )


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = _parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if args.min_round_trips < 0:
        raise SystemExit("--min-round-trips cannot be negative")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("--fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("--slippage-pct cannot be negative")

    parameter_sets = generate_parameter_sets(
        ema_fast_values=_parse_int_list(args.ema_fast_values, field_name="--ema-fast-values"),
        ema_slow_values=_parse_int_list(args.ema_slow_values, field_name="--ema-slow-values"),
        rsi_period_values=_parse_int_list(
            args.rsi_period_values,
            field_name="--rsi-period-values",
        ),
        rsi_buy_min_values=_parse_float_list(
            args.rsi_buy_min_values,
            field_name="--rsi-buy-min-values",
        ),
        rsi_buy_max_values=_parse_float_list(
            args.rsi_buy_max_values,
            field_name="--rsi-buy-max-values",
        ),
        rsi_sell_min_values=_parse_float_list(
            args.rsi_sell_min_values,
            field_name="--rsi-sell-min-values",
        ),
        stop_loss_pct_values=_parse_decimal_list(
            args.stop_loss_pct_values,
            field_name="--stop-loss-pct-values",
        ),
        take_profit_pct_values=_parse_decimal_list(
            args.take_profit_pct_values,
            field_name="--take-profit-pct-values",
        ),
        trend_ema_period_values=_parse_optional_int_list(
            args.trend_ema_values,
            field_name="--trend-ema-values",
        ),
        min_ema_gap_pct_values=_parse_non_negative_decimal_list(
            args.min_ema_gap_pct_values,
            field_name="--min-ema-gap-pct-values",
        ),
        atr_period_values=_parse_optional_int_list(
            args.atr_period_values,
            field_name="--atr-period-values",
        ),
        min_atr_pct_values=_parse_non_negative_decimal_list(
            args.min_atr_pct_values,
            field_name="--min-atr-pct-values",
        ),
    )
    if not parameter_sets:
        raise SystemExit("No valid parameter combinations generated")

    candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not candles:
        raise SystemExit("No candles available for optimization")

    market_data_source, _use_testnet_data, exchange_id = _resolve_market_data_source(
        args.market_data_source
    )
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "strategy_optimization_started",
        candles=len(candles),
        parameter_sets=len(parameter_sets),
        source=args.source,
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        min_round_trips=args.min_round_trips,
    )

    results = optimize_parameter_grid(
        candles=candles,
        symbol=settings.normalized_symbol,
        parameter_sets=parameter_sets,
        initial_quote_balance=settings.initial_quote_balance,
        max_order_usdt=settings.max_order_usdt,
        max_position_usdt=settings.max_position_usdt,
        allow_only_one_open_position=settings.allow_only_one_open_position,
        fee_rate_pct=fee_rate_pct,
        slippage_pct=slippage_pct,
        min_round_trips=args.min_round_trips,
    )

    logger.info(
        "strategy_optimization_finished",
        evaluated_parameter_sets=len(parameter_sets),
        kept_parameter_sets=len(results),
    )

    for item in results[: args.top]:
        _log_result(item)

    if args.export_json:
        _export_json(args.export_json, results)
    if args.export_csv:
        _export_csv(args.export_csv, results)


if __name__ == "__main__":
    asyncio.run(main())
