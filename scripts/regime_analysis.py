"""Per-regime strategy edge analysis (V21 research tool).

Aggregate backtests showed no edge in any strategy family. This script asks the
sharper question: does any family have an edge *inside the regime it is built
for*? It labels every candle bullish / bearish / sideways with the causal V20
regime detector, runs each strategy, and reports expectancy / profit factor /
win rate bucketed by the regime active at each trade's entry.

Read it like this:
- mean_reversion is interesting only if it has profit factor > 1 in SIDEWAYS.
- breakout / ema_rsi are interesting only if profit factor > 1 in BULLISH
  (the only trending regime a long-only spot bot can ride).
A positive bucket is a *lead* to gate with a regime filter, not a finished
strategy. If every bucket is still profit_factor < 1, the family is simply out
of ideas and no amount of regime slicing will resurrect it.
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

from app.backtesting.analytics import TradeStatistics
from app.backtesting.engine import BacktestEngine
from app.backtesting.regime_analysis import (
    RegimeBucket,
    build_regime_buckets,
    label_regimes,
    regime_candle_counts,
)
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.models import Candle
from app.strategy.base import Strategy
from app.strategy.breakout_momentum import BreakoutMomentumStrategy
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.market_regime import MarketRegime
from app.strategy.mean_reversion import MeanReversionStrategy
from scripts.backtest_strategy import _decimal_to_str, _load_candles, _resolve_market_data_source

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class StrategyRegimeRow:
    timeframe: str
    strategy_name: str
    bucket: RegimeBucket


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Break down each strategy's trade edge by market regime (bullish/bearish/sideways)."
    )
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--timeframes", default="5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--min-candles-per-timeframe", type=int, default=300)
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
    parser.add_argument("--stop-loss-pct", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--take-profit-pct", type=Decimal, default=Decimal("0.8"))
    # EMA/RSI baseline = best V17 region.
    parser.add_argument("--ema-fast-period", type=int, default=12)
    parser.add_argument("--ema-slow-period", type=int, default=34)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-buy-min", type=float, default=45.0)
    parser.add_argument("--rsi-buy-max", type=float, default=60.0)
    parser.add_argument("--rsi-sell-min", type=float, default=70.0)
    parser.add_argument("--trend-ema-period", type=int, default=200)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-atr-pct", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--min-ema-gap-pct", type=Decimal, default=Decimal("0"))
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--breakout-exit-lookback", type=int, default=10)
    parser.add_argument("--mr-lookback", type=int, default=20)
    parser.add_argument("--mr-entry-z", type=Decimal, default=Decimal("2.0"))
    parser.add_argument("--mr-exit-z", type=Decimal, default=Decimal("0.0"))
    parser.add_argument("--mr-rsi-buy-max", type=float, default=35.0)
    # Regime detector knobs (mirror compare_strategies / V20 defaults).
    parser.add_argument("--regime-fast-ema-period", type=int, default=50)
    parser.add_argument("--regime-slow-ema-period", type=int, default=200)
    parser.add_argument("--regime-slope-lookback", type=int, default=20)
    parser.add_argument("--regime-min-slope-pct", type=Decimal, default=Decimal("0.03"))
    parser.add_argument("--regime-min-ema-gap-pct", type=Decimal, default=Decimal("0.05"))
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _build_strategies(args: argparse.Namespace, suggested_quote_amount: Decimal) -> list[Strategy]:
    trend_ema = args.trend_ema_period if args.trend_ema_period > 0 else None
    atr_period = args.atr_period if args.atr_period > 0 else None
    return [
        EmaRsiStrategy(
            fast_period=args.ema_fast_period,
            slow_period=args.ema_slow_period,
            rsi_period=args.rsi_period,
            rsi_buy_min=args.rsi_buy_min,
            rsi_buy_max=args.rsi_buy_max,
            rsi_sell_min=args.rsi_sell_min,
            suggested_quote_amount=suggested_quote_amount,
            trend_ema_period=trend_ema,
            min_ema_gap_pct=args.min_ema_gap_pct,
            atr_period=atr_period,
            min_atr_pct=args.min_atr_pct,
        ),
        BreakoutMomentumStrategy(
            breakout_lookback=args.breakout_lookback,
            exit_lookback=args.breakout_exit_lookback,
            suggested_quote_amount=suggested_quote_amount,
            trend_ema_period=trend_ema,
            atr_period=atr_period,
            min_atr_pct=args.min_atr_pct,
        ),
        MeanReversionStrategy(
            lookback=args.mr_lookback,
            entry_z=args.mr_entry_z,
            exit_z=args.mr_exit_z,
            rsi_period=args.rsi_period,
            rsi_buy_max=args.mr_rsi_buy_max if args.mr_rsi_buy_max > 0 else None,
            trend_ema_period=trend_ema,
            atr_period=atr_period,
            min_atr_pct=args.min_atr_pct,
            suggested_quote_amount=suggested_quote_amount,
        ),
    ]


def _run_engine(strategy: Strategy, candles: list[Candle], args: argparse.Namespace, settings, fee_rate_pct, slippage_pct):
    engine = BacktestEngine(
        strategy=strategy,
        symbol=settings.normalized_symbol,
        initial_quote_balance=settings.initial_quote_balance,
        max_order_usdt=settings.max_order_usdt,
        max_position_usdt=settings.max_position_usdt,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        allow_only_one_open_position=settings.allow_only_one_open_position,
        fee_rate_pct=fee_rate_pct,
        slippage_pct=slippage_pct,
    )
    return engine.run(candles)


def _bucket_payload(timeframe: str, strategy_name: str, bucket: RegimeBucket) -> dict[str, Any]:
    stats: TradeStatistics = bucket.stats
    return {
        "timeframe": timeframe,
        "strategy_name": strategy_name,
        "regime": bucket.regime.value,
        "candle_count": bucket.candle_count,
        "num_trades": stats.num_trades,
        "win_rate": _decimal_to_str(stats.win_rate),
        "profit_factor": _decimal_to_str(stats.profit_factor),
        "expectancy": _decimal_to_str(stats.expectancy),
        "expectancy_r": _decimal_to_str(stats.expectancy_r),
        "total_pnl": _decimal_to_str(bucket.total_pnl),
        "max_consecutive_losses": stats.max_consecutive_losses,
    }


def _export_json(path: Path, rows: list[StrategyRegimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_bucket_payload(row.timeframe, row.strategy_name, row.bucket) for row in rows]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("regime_analysis_json_exported", path=str(path), rows=len(payload))


def _export_csv(path: Path, rows: list[StrategyRegimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timeframe",
        "strategy_name",
        "regime",
        "candle_count",
        "num_trades",
        "win_rate",
        "profit_factor",
        "expectancy",
        "expectancy_r",
        "total_pnl",
        "max_consecutive_losses",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_bucket_payload(row.timeframe, row.strategy_name, row.bucket))
    logger.info("regime_analysis_csv_exported", path=str(path), rows=len(rows))


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
        raise SystemExit("No candles available for regime analysis")

    market_data_source, _use_testnet, exchange_id = _resolve_market_data_source(args.market_data_source)
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "regime_analysis_started",
        source=args.source,
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        source_candles=len(source_candles),
        timeframes=args.timeframes,
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
    )

    regime_kwargs = dict(
        fast_ema_period=args.regime_fast_ema_period,
        slow_ema_period=args.regime_slow_ema_period,
        slope_lookback=args.regime_slope_lookback,
        min_slope_pct=args.regime_min_slope_pct,
        min_ema_gap_pct=args.regime_min_ema_gap_pct,
    )

    rows: list[StrategyRegimeRow] = []
    for timeframe in _parse_timeframes(args.timeframes):
        candles = resample_candles(source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe)
        if len(candles) < args.min_candles_per_timeframe:
            logger.warning("regime_analysis_timeframe_skipped", timeframe=timeframe, candles=len(candles))
            continue

        labels = label_regimes(candles, **regime_kwargs)
        counts = regime_candle_counts(labels)
        logger.info(
            "regime_distribution",
            timeframe=timeframe,
            candles=len(candles),
            bullish=counts.get(MarketRegime.BULLISH, 0),
            bearish=counts.get(MarketRegime.BEARISH, 0),
            sideways=counts.get(MarketRegime.SIDEWAYS, 0),
            unknown=counts.get(MarketRegime.UNKNOWN, 0),
        )

        for strategy in _build_strategies(args, settings.max_order_usdt):
            result = _run_engine(strategy, candles, args, settings, fee_rate_pct, slippage_pct)
            buckets = build_regime_buckets(result.trades, labels)
            for bucket in buckets:
                rows.append(StrategyRegimeRow(timeframe=timeframe, strategy_name=strategy.name, bucket=bucket))
                if bucket.stats.num_trades == 0:
                    continue
                logger.info(
                    "regime_edge",
                    timeframe=timeframe,
                    strategy_name=strategy.name,
                    regime=bucket.regime.value,
                    candle_count=bucket.candle_count,
                    num_trades=bucket.stats.num_trades,
                    win_rate=round(float(bucket.stats.win_rate), 4),
                    profit_factor=_decimal_to_str(bucket.stats.profit_factor),
                    expectancy=str(bucket.stats.expectancy),
                    total_pnl=str(bucket.total_pnl),
                )

    if not rows:
        raise SystemExit("No timeframe had enough candles for regime analysis")

    if args.export_json:
        _export_json(args.export_json, rows)
    if args.export_csv:
        _export_csv(args.export_csv, rows)

    logger.info("regime_analysis_finished", rows=len(rows))


if __name__ == "__main__":
    asyncio.run(main())
