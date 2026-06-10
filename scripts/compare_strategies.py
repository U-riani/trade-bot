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

from app.backtesting.analytics import (
    EquityStatistics,
    TradeStatistics,
    bars_per_year_for_timeframe,
    compute_equity_statistics,
    compute_trade_statistics,
)
from app.backtesting.benchmarks import (
    buy_and_hold_order_sized_benchmark,
    no_trade_benchmark,
    split_walk_forward,
)
from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import BacktestMetrics, BacktestResult
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.models import Candle
from app.strategy.base import Strategy
from app.strategy.breakout_momentum import BreakoutMomentumStrategy
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.market_regime import MarketRegimeFilteredStrategy
from scripts.backtest_strategy import _decimal_to_str, _load_candles, _resolve_market_data_source

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class StrategyComparisonRow:
    timeframe: str
    segment: str
    strategy_name: str
    category: str
    result: BacktestResult
    notes: str

    @property
    def score(self) -> Decimal:
        metrics = self.result.metrics
        return metrics.final_equity - (metrics.max_drawdown * Decimal("0.05")) - (
            metrics.total_fees * Decimal("0.05")
        )

    @property
    def trade_stats(self) -> TradeStatistics:
        return compute_trade_statistics(self.result.trades)

    @property
    def equity_stats(self) -> EquityStatistics:
        return compute_equity_statistics(
            self.result.equity_curve,
            bars_per_year=bars_per_year_for_timeframe(self.timeframe),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare EMA/RSI, breakout/momentum, no-trade, and buy-and-hold benchmarks "
            "on full and walk-forward candle segments."
        )
    )
    parser.add_argument("--limit", type=int, default=10000, help="Number of recent source candles to use.")
    parser.add_argument(
        "--timeframes",
        default="1m,5m,15m",
        help="Comma-separated timeframes to compare. V20 keeps V19 default: 1m,5m,15m.",
    )
    parser.add_argument(
        "--source-timeframe",
        default="1m",
        help="Timeframe of loaded candles before resampling. Default: 1m.",
    )
    parser.add_argument(
        "--min-candles-per-timeframe",
        type=int,
        default=200,
        help="Skip a timeframe if resampling leaves fewer candles than this.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "db", "rest"),
        default="db",
        help="Candle source. Use db after scripts.backfill_candles.",
    )
    parser.add_argument(
        "--market-data-source",
        choices=("production", "testnet"),
        default=None,
        help="Historical candle source. Defaults to HISTORICAL_MARKET_DATA_SOURCE.",
    )
    parser.add_argument(
        "--train-ratio",
        type=Decimal,
        default=Decimal("0.7"),
        help="Walk-forward train split ratio, e.g. 0.7.",
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
        "--ema-fast-period",
        type=int,
        default=12,
        help="EMA/RSI baseline fast EMA. Default uses the best V17 region.",
    )
    parser.add_argument("--ema-slow-period", type=int, default=34)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-buy-min", type=float, default=45.0)
    parser.add_argument("--rsi-buy-max", type=float, default=60.0)
    parser.add_argument("--rsi-sell-min", type=float, default=70.0)
    parser.add_argument("--trend-ema-period", type=int, default=200)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--min-atr-pct", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--min-ema-gap-pct", type=Decimal, default=Decimal("0"))
    parser.add_argument("--stop-loss-pct", type=Decimal, default=Decimal("0.5"))
    parser.add_argument("--take-profit-pct", type=Decimal, default=Decimal("0.8"))
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--breakout-exit-lookback", type=int, default=10)
    parser.add_argument("--breakout-min-pct", type=Decimal, default=Decimal("0"))
    parser.add_argument(
        "--regime-fast-ema-period",
        type=int,
        default=50,
        help="V20 market-regime fast EMA. Used by regime-filtered strategy.",
    )
    parser.add_argument(
        "--regime-slow-ema-period",
        type=int,
        default=200,
        help="V20 market-regime slow EMA. Used by regime-filtered strategy.",
    )
    parser.add_argument(
        "--regime-slope-lookback",
        type=int,
        default=20,
        help="How many candles back to measure slow EMA slope for regime filter.",
    )
    parser.add_argument(
        "--regime-min-slope-pct",
        type=Decimal,
        default=Decimal("0.03"),
        help="Minimum slow EMA slope percent required for bullish regime.",
    )
    parser.add_argument(
        "--regime-min-ema-gap-pct",
        type=Decimal,
        default=Decimal("0.05"),
        help="Minimum fast-vs-slow EMA gap percent required for bullish regime.",
    )
    parser.add_argument("--top", type=int, default=20, help="How many rows to log.")
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _run_strategy(
    *,
    candles: list[Candle],
    strategy: Strategy,
    symbol: str,
    initial_quote_balance: Decimal,
    max_order_usdt: Decimal,
    max_position_usdt: Decimal,
    stop_loss_pct: Decimal,
    take_profit_pct: Decimal,
    allow_only_one_open_position: bool,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> BacktestResult:
    engine = BacktestEngine(
        strategy=strategy,
        symbol=symbol,
        initial_quote_balance=initial_quote_balance,
        max_order_usdt=max_order_usdt,
        max_position_usdt=max_position_usdt,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        allow_only_one_open_position=allow_only_one_open_position,
        fee_rate_pct=fee_rate_pct,
        slippage_pct=slippage_pct,
    )
    return engine.run(candles)


def _comparison_rows_for_segment(
    *,
    timeframe: str,
    segment: str,
    candles: list[Candle],
    args: argparse.Namespace,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> list[StrategyComparisonRow]:
    settings = get_settings()
    ema_strategy = EmaRsiStrategy(
        fast_period=args.ema_fast_period,
        slow_period=args.ema_slow_period,
        rsi_period=args.rsi_period,
        rsi_buy_min=args.rsi_buy_min,
        rsi_buy_max=args.rsi_buy_max,
        rsi_sell_min=args.rsi_sell_min,
        suggested_quote_amount=settings.max_order_usdt,
        trend_ema_period=args.trend_ema_period if args.trend_ema_period > 0 else None,
        min_ema_gap_pct=args.min_ema_gap_pct,
        atr_period=args.atr_period if args.atr_period > 0 else None,
        min_atr_pct=args.min_atr_pct,
    )
    regime_filtered_ema_strategy = MarketRegimeFilteredStrategy(
        base_strategy=EmaRsiStrategy(
            fast_period=args.ema_fast_period,
            slow_period=args.ema_slow_period,
            rsi_period=args.rsi_period,
            rsi_buy_min=args.rsi_buy_min,
            rsi_buy_max=args.rsi_buy_max,
            rsi_sell_min=args.rsi_sell_min,
            suggested_quote_amount=settings.max_order_usdt,
            trend_ema_period=args.trend_ema_period if args.trend_ema_period > 0 else None,
            min_ema_gap_pct=args.min_ema_gap_pct,
            atr_period=args.atr_period if args.atr_period > 0 else None,
            min_atr_pct=args.min_atr_pct,
        ),
        fast_ema_period=args.regime_fast_ema_period,
        slow_ema_period=args.regime_slow_ema_period,
        slope_lookback=args.regime_slope_lookback,
        min_slope_pct=args.regime_min_slope_pct,
        min_ema_gap_pct=args.regime_min_ema_gap_pct,
        name="regime_filtered_ema_rsi_v20",
    )
    breakout_strategy = BreakoutMomentumStrategy(
        breakout_lookback=args.breakout_lookback,
        exit_lookback=args.breakout_exit_lookback,
        suggested_quote_amount=settings.max_order_usdt,
        trend_ema_period=args.trend_ema_period if args.trend_ema_period > 0 else None,
        atr_period=args.atr_period if args.atr_period > 0 else None,
        min_atr_pct=args.min_atr_pct,
        min_breakout_pct=args.breakout_min_pct,
    )

    common = {
        "candles": candles,
        "symbol": settings.normalized_symbol,
        "initial_quote_balance": settings.initial_quote_balance,
        "max_order_usdt": settings.max_order_usdt,
        "max_position_usdt": settings.max_position_usdt,
        "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct,
        "allow_only_one_open_position": settings.allow_only_one_open_position,
        "fee_rate_pct": fee_rate_pct,
        "slippage_pct": slippage_pct,
    }

    return [
        StrategyComparisonRow(
            timeframe=timeframe,
            segment=segment,
            strategy_name="no_trade",
            category="benchmark",
            result=no_trade_benchmark(
                candles=candles,
                initial_quote_balance=settings.initial_quote_balance,
            ),
            notes="Do nothing benchmark. This humiliates many strategies by existing.",
        ),
        StrategyComparisonRow(
            timeframe=timeframe,
            segment=segment,
            strategy_name="buy_hold_order_sized",
            category="benchmark",
            result=buy_and_hold_order_sized_benchmark(
                candles=candles,
                initial_quote_balance=settings.initial_quote_balance,
                quote_amount=settings.max_order_usdt,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            ),
            notes="Buy once with MAX_ORDER_USDT and hold to the last candle.",
        ),
        StrategyComparisonRow(
            timeframe=timeframe,
            segment=segment,
            strategy_name="ema_rsi_v17_best_region",
            category="strategy",
            result=_run_strategy(strategy=ema_strategy, **common),
            notes="Filtered EMA/RSI baseline from the best V17 region.",
        ),
        StrategyComparisonRow(
            timeframe=timeframe,
            segment=segment,
            strategy_name="regime_filtered_ema_rsi_v20",
            category="strategy_candidate",
            result=_run_strategy(strategy=regime_filtered_ema_strategy, **common),
            notes="V20 EMA/RSI with bullish-regime gate. BUY only when market context agrees.",
        ),
        StrategyComparisonRow(
            timeframe=timeframe,
            segment=segment,
            strategy_name="breakout_momentum_v1",
            category="strategy_candidate",
            result=_run_strategy(strategy=breakout_strategy, **common),
            notes="Breakout above prior high with trend/ATR filters and low-break exit.",
        ),
    ]


def _row_payload(row: StrategyComparisonRow, rank: int) -> dict[str, Any]:
    metrics = row.result.metrics
    return {
        "rank": rank,
        "timeframe": row.timeframe,
        "segment": row.segment,
        "strategy_name": row.strategy_name,
        "category": row.category,
        "score": _decimal_to_str(row.score),
        "notes": row.notes,
        "metrics": _metrics_payload(metrics),
        "edge": _analytics_payload(row.trade_stats, row.equity_stats),
    }


def _analytics_payload(trade_stats: TradeStatistics, equity_stats: EquityStatistics) -> dict[str, Any]:
    """Edge-quality block: the numbers that separate a real edge from a lucky run."""
    return {
        "profit_factor": _decimal_to_str(trade_stats.profit_factor),
        "expectancy": _decimal_to_str(trade_stats.expectancy),
        "expectancy_r": _decimal_to_str(trade_stats.expectancy_r),
        "payoff_ratio": _decimal_to_str(trade_stats.payoff_ratio),
        "avg_win": _decimal_to_str(trade_stats.avg_win),
        "avg_loss": _decimal_to_str(trade_stats.avg_loss),
        "avg_return_pct": _decimal_to_str(trade_stats.avg_return_pct),
        "largest_win": _decimal_to_str(trade_stats.largest_win),
        "largest_loss": _decimal_to_str(trade_stats.largest_loss),
        "max_consecutive_wins": trade_stats.max_consecutive_wins,
        "max_consecutive_losses": trade_stats.max_consecutive_losses,
        "trade_return_sharpe": _decimal_to_str(trade_stats.trade_return_sharpe),
        "sharpe_annualized": _decimal_to_str(equity_stats.sharpe),
        "sortino_annualized": _decimal_to_str(equity_stats.sortino),
        "max_drawdown_pct": _decimal_to_str(equity_stats.max_drawdown_pct),
    }


def _metrics_payload(metrics: BacktestMetrics) -> dict[str, Any]:
    return {
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
    }


def _rank_rows(rows: list[StrategyComparisonRow]) -> list[tuple[int, StrategyComparisonRow]]:
    segment_order = {"full": 0, "train": 1, "validation": 2}
    ranked: list[tuple[int, StrategyComparisonRow]] = []

    timeframe_order = {"1m": 0, "3m": 1, "5m": 2, "15m": 3, "1h": 4}

    groups = sorted(
        {(row.timeframe, row.segment) for row in rows},
        key=lambda item: (timeframe_order.get(item[0], 99), item[0], segment_order.get(item[1], 99)),
    )
    for timeframe, segment in groups:
        segment_rows = [row for row in rows if row.timeframe == timeframe and row.segment == segment]
        sorted_segment_rows = sorted(
            segment_rows,
            key=lambda item: (
                item.score,
                item.result.metrics.final_equity,
                -item.result.metrics.max_drawdown,
                -item.result.metrics.round_trips,
            ),
            reverse=True,
        )
        ranked.extend((index, row) for index, row in enumerate(sorted_segment_rows, start=1))

    return ranked


def _log_overfit_report(rows: list[StrategyComparisonRow]) -> None:
    """Flag strategies that look good on train data but fall apart on validation.

    The single most common way a backtest lies: a parameter set that fits the
    in-sample noise. We catch it by comparing the same strategy's return on the
    train segment versus the held-out validation segment. Benchmarks are skipped
    because "do nothing" cannot overfit.
    """
    by_key: dict[tuple[str, str], dict[str, StrategyComparisonRow]] = {}
    for row in rows:
        if row.category == "benchmark":
            continue
        key = (row.timeframe, row.strategy_name)
        by_key.setdefault(key, {})[row.segment] = row

    for (timeframe, strategy_name), segments in sorted(by_key.items()):
        train = segments.get("train")
        validation = segments.get("validation")
        if train is None or validation is None:
            continue

        train_return = train.result.metrics.return_pct
        validation_return = validation.result.metrics.return_pct
        degradation = train_return - validation_return

        # Positive in-sample but non-positive out-of-sample is the textbook tell.
        overfit_suspected = train_return > 0 >= validation_return
        verdict = "overfit_suspected" if overfit_suspected else "ok"

        logger.info(
            "strategy_overfit_check",
            timeframe=timeframe,
            strategy_name=strategy_name,
            verdict=verdict,
            train_return_pct=str(train_return),
            validation_return_pct=str(validation_return),
            degradation_pct=str(degradation),
            validation_round_trips=validation.result.metrics.round_trips,
        )


def _log_row(rank: int, row: StrategyComparisonRow) -> None:
    metrics = row.result.metrics
    trade_stats = row.trade_stats
    equity_stats = row.equity_stats
    logger.info(
        "strategy_comparison_result",
        rank=rank,
        timeframe=row.timeframe,
        segment=row.segment,
        strategy_name=row.strategy_name,
        category=row.category,
        score=str(row.score),
        final_equity=str(metrics.final_equity),
        return_pct=str(metrics.return_pct),
        max_drawdown=str(metrics.max_drawdown),
        round_trips=metrics.round_trips,
        win_rate=round(metrics.win_rate, 4),
        profit_factor=_decimal_to_str(trade_stats.profit_factor),
        expectancy=str(trade_stats.expectancy),
        expectancy_r=_decimal_to_str(trade_stats.expectancy_r),
        sharpe_annualized=_decimal_to_str(equity_stats.sharpe),
        max_consecutive_losses=trade_stats.max_consecutive_losses,
        total_fees=str(metrics.total_fees),
        has_open_position=metrics.has_open_position,
    )


def _export_json(path: Path, ranked_rows: list[tuple[int, StrategyComparisonRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_row_payload(row, rank) for rank, row in ranked_rows]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("strategy_comparison_json_exported", path=str(path), rows=len(payload))


def _export_csv(path: Path, ranked_rows: list[tuple[int, StrategyComparisonRow]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "timeframe",
        "segment",
        "strategy_name",
        "category",
        "score",
        "final_equity",
        "return_pct",
        "max_drawdown",
        "round_trips",
        "executed_orders",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "profit_factor",
        "expectancy",
        "expectancy_r",
        "payoff_ratio",
        "sharpe_annualized",
        "sortino_annualized",
        "max_drawdown_pct",
        "max_consecutive_losses",
        "total_fees",
        "realized_pnl",
        "unrealized_pnl",
        "has_open_position",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in ranked_rows:
            metrics = row.result.metrics
            trade_stats = row.trade_stats
            equity_stats = row.equity_stats
            writer.writerow(
                {
                    "rank": rank,
                    "timeframe": row.timeframe,
                    "segment": row.segment,
                    "strategy_name": row.strategy_name,
                    "category": row.category,
                    "score": str(row.score),
                    "final_equity": str(metrics.final_equity),
                    "return_pct": str(metrics.return_pct),
                    "max_drawdown": str(metrics.max_drawdown),
                    "round_trips": metrics.round_trips,
                    "executed_orders": metrics.executed_orders,
                    "winning_trades": metrics.winning_trades,
                    "losing_trades": metrics.losing_trades,
                    "win_rate": metrics.win_rate,
                    "profit_factor": _decimal_to_str(trade_stats.profit_factor),
                    "expectancy": str(trade_stats.expectancy),
                    "expectancy_r": _decimal_to_str(trade_stats.expectancy_r),
                    "payoff_ratio": _decimal_to_str(trade_stats.payoff_ratio),
                    "sharpe_annualized": _decimal_to_str(equity_stats.sharpe),
                    "sortino_annualized": _decimal_to_str(equity_stats.sortino),
                    "max_drawdown_pct": str(equity_stats.max_drawdown_pct),
                    "max_consecutive_losses": trade_stats.max_consecutive_losses,
                    "total_fees": str(metrics.total_fees),
                    "realized_pnl": str(metrics.realized_pnl),
                    "unrealized_pnl": str(metrics.unrealized_pnl),
                    "has_open_position": metrics.has_open_position,
                    "notes": row.notes,
                }
            )
    logger.info("strategy_comparison_csv_exported", path=str(path), rows=len(ranked_rows))


def _parse_timeframes(value: str) -> list[str]:
    timeframes: list[str] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item not in timeframes:
            timeframes.append(item)
    if not timeframes:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return timeframes


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = _parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("--fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("--slippage-pct cannot be negative")
    if args.regime_fast_ema_period <= 0:
        raise SystemExit("--regime-fast-ema-period must be positive")
    if args.regime_slow_ema_period <= 0:
        raise SystemExit("--regime-slow-ema-period must be positive")
    if args.regime_fast_ema_period >= args.regime_slow_ema_period:
        raise SystemExit("--regime-fast-ema-period must be smaller than --regime-slow-ema-period")
    if args.regime_slope_lookback <= 0:
        raise SystemExit("--regime-slope-lookback must be positive")
    if args.regime_min_slope_pct < 0:
        raise SystemExit("--regime-min-slope-pct cannot be negative")
    if args.regime_min_ema_gap_pct < 0:
        raise SystemExit("--regime-min-ema-gap-pct cannot be negative")

    source_candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not source_candles:
        raise SystemExit("No candles available for strategy comparison")

    requested_timeframes = _parse_timeframes(args.timeframes)
    market_data_source, _use_testnet_data, exchange_id = _resolve_market_data_source(
        args.market_data_source
    )
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "strategy_comparison_started",
        source=args.source,
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        source_candles=len(source_candles),
        source_timeframe=args.source_timeframe,
        requested_timeframes=",".join(requested_timeframes),
        train_ratio=str(args.train_ratio),
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        regime_fast_ema_period=args.regime_fast_ema_period,
        regime_slow_ema_period=args.regime_slow_ema_period,
        regime_slope_lookback=args.regime_slope_lookback,
        regime_min_slope_pct=str(args.regime_min_slope_pct),
        regime_min_ema_gap_pct=str(args.regime_min_ema_gap_pct),
    )

    rows: list[StrategyComparisonRow] = []
    for timeframe in requested_timeframes:
        candles = resample_candles(
            source_candles,
            target_timeframe=timeframe,
            source_timeframe=args.source_timeframe,
        )
        if len(candles) < args.min_candles_per_timeframe:
            logger.warning(
                "strategy_comparison_timeframe_skipped",
                timeframe=timeframe,
                candles=len(candles),
                min_candles=args.min_candles_per_timeframe,
            )
            continue

        train_candles, validation_candles = split_walk_forward(candles, train_ratio=args.train_ratio)
        logger.info(
            "strategy_comparison_timeframe_ready",
            timeframe=timeframe,
            candles=len(candles),
            train_candles=len(train_candles),
            validation_candles=len(validation_candles),
        )

        rows.extend(
            _comparison_rows_for_segment(
                timeframe=timeframe,
                segment="full",
                candles=candles,
                args=args,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
        )
        rows.extend(
            _comparison_rows_for_segment(
                timeframe=timeframe,
                segment="train",
                candles=train_candles,
                args=args,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
        )
        rows.extend(
            _comparison_rows_for_segment(
                timeframe=timeframe,
                segment="validation",
                candles=validation_candles,
                args=args,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
        )

    if not rows:
        raise SystemExit("No timeframe had enough candles to compare")

    ranked_rows = _rank_rows(rows)
    logger.info("strategy_comparison_finished", rows=len(ranked_rows))
    for rank, row in ranked_rows:
        if rank <= args.top:
            _log_row(rank, row)

    _log_overfit_report(rows)

    if args.export_json:
        _export_json(args.export_json, ranked_rows)
    if args.export_csv:
        _export_csv(args.export_csv, ranked_rows)


if __name__ == "__main__":
    asyncio.run(main())
