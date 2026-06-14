"""V27 research-only backtest for order-book imbalance threshold candidates.

This script exists to answer one practical question: is the V26/V26.1 order-book
signal interesting enough to keep developing the bot? It does not trade, does not
emit live signals, and does not declare victory because one tiny sample got cute.
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

from app.backtesting.analytics import (
    EquityStatistics,
    TradeStatistics,
    bars_per_year_for_timeframe,
    compute_equity_statistics,
    compute_trade_statistics,
)
from app.backtesting.order_book_strategy import (
    OrderBookThresholdConfig,
    buy_and_hold_feature_rows,
    quantile_threshold,
    rows_with_feature,
    run_order_book_threshold_backtest,
    split_feature_rows,
)
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import MarketFeatures
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.backtest_strategy import _decimal_to_str, _resolve_market_data_source

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class OrderBookStrategyRow:
    timeframe: str
    segment: str
    strategy_name: str
    feature: str
    horizon_bars: int
    entry_quantile: Decimal
    entry_threshold: float
    sample_size: int
    strategy_result: object
    buy_hold_result: object
    notes: str

    @property
    def trade_stats(self) -> TradeStatistics:
        return compute_trade_statistics(self.strategy_result.trades)  # type: ignore[attr-defined]

    @property
    def equity_stats(self) -> EquityStatistics:
        return compute_equity_statistics(
            self.strategy_result.equity_curve,  # type: ignore[attr-defined]
            bars_per_year=bars_per_year_for_timeframe(self.timeframe),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest research-only order-book imbalance threshold candidates. "
            "No live signals, no trading, no profit claims."
        )
    )
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--timeframe", default="5m", help="Feature/candle timeframe to test. Default: 5m.")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument(
        "--features",
        default="imbalance_top_20,order_book_imbalance,imbalance_top_10,imbalance_top_5",
        help="Comma-separated market_features columns to test.",
    )
    parser.add_argument("--horizons", default="1,3,6", help="Comma-separated fixed holding periods in bars.")
    parser.add_argument("--entry-quantiles", default="0.6,0.7,0.8", help="Train-set quantiles used as entry thresholds.")
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--train-ratio", type=Decimal, default=Decimal("0.7"))
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _parse_csv_list(value: str) -> list[str]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if item and item not in result:
            result.append(item)
    if not result:
        raise SystemExit("CSV argument must contain at least one item")
    return result


def _parse_ints(value: str) -> list[int]:
    parsed: list[int] = []
    for item in _parse_csv_list(value):
        number = int(item)
        if number <= 0:
            raise SystemExit("horizons must be positive integers")
        parsed.append(number)
    return parsed


def _parse_quantiles(value: str) -> list[Decimal]:
    quantiles: list[Decimal] = []
    for item in _parse_csv_list(value):
        quantile = Decimal(item)
        if quantile <= 0 or quantile >= 1:
            raise SystemExit("entry quantiles must be between 0 and 1")
        quantiles.append(quantile)
    return quantiles


def _feature_sample_size(rows: list[MarketFeatures], feature: str) -> int:
    return len(rows_with_feature(rows, feature))


def _make_strategy_name(feature: str, quantile: Decimal, horizon: int) -> str:
    q_label = str(quantile).replace(".", "p")
    return f"order_book_threshold_{feature}_q{q_label}_h{horizon}_v27"


def _verdict(row: OrderBookStrategyRow) -> str:
    metrics = row.strategy_result.metrics  # type: ignore[attr-defined]
    trade_stats = row.trade_stats
    buy_hold_return = row.buy_hold_result.metrics.return_pct  # type: ignore[attr-defined]
    if metrics.round_trips < 3:
        return "not_enough_trades"
    if metrics.return_pct <= 0:
        return "failed_positive_return_check"
    if metrics.return_pct <= buy_hold_return:
        return "does_not_beat_order_sized_buy_hold"
    if trade_stats.profit_factor is not None and trade_stats.profit_factor <= Decimal("1"):
        return "profit_factor_not_above_1"
    return "promising_research_only"


def _row_payload(row: OrderBookStrategyRow, rank: int) -> dict[str, Any]:
    metrics = row.strategy_result.metrics  # type: ignore[attr-defined]
    buy_hold_metrics = row.buy_hold_result.metrics  # type: ignore[attr-defined]
    trade_stats = row.trade_stats
    equity_stats = row.equity_stats
    return {
        "rank": rank,
        "timeframe": row.timeframe,
        "segment": row.segment,
        "strategy_name": row.strategy_name,
        "feature": row.feature,
        "horizon_bars": row.horizon_bars,
        "entry_quantile": str(row.entry_quantile),
        "entry_threshold": row.entry_threshold,
        "sample_size": row.sample_size,
        "verdict": _verdict(row),
        "notes": row.notes,
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
        },
        "edge": {
            "profit_factor": _decimal_to_str(trade_stats.profit_factor),
            "expectancy": _decimal_to_str(trade_stats.expectancy),
            "expectancy_r": _decimal_to_str(trade_stats.expectancy_r),
            "payoff_ratio": _decimal_to_str(trade_stats.payoff_ratio),
            "avg_return_pct": _decimal_to_str(trade_stats.avg_return_pct),
            "max_consecutive_losses": trade_stats.max_consecutive_losses,
            "trade_return_sharpe": _decimal_to_str(trade_stats.trade_return_sharpe),
            "sharpe_annualized": _decimal_to_str(equity_stats.sharpe),
            "sortino_annualized": _decimal_to_str(equity_stats.sortino),
            "max_drawdown_pct": _decimal_to_str(equity_stats.max_drawdown_pct),
        },
        "benchmark": {
            "buy_hold_order_sized_final_equity": _decimal_to_str(buy_hold_metrics.final_equity),
            "buy_hold_order_sized_return_pct": _decimal_to_str(buy_hold_metrics.return_pct),
            "beats_no_trade": metrics.return_pct > 0,
            "beats_order_sized_buy_hold": metrics.return_pct > buy_hold_metrics.return_pct,
        },
    }


def _sort_key(row: OrderBookStrategyRow) -> tuple[Decimal, Decimal, int]:
    metrics = row.strategy_result.metrics  # type: ignore[attr-defined]
    trade_stats = row.trade_stats
    profit_factor = trade_stats.profit_factor or Decimal("0")
    return (metrics.return_pct, profit_factor, metrics.round_trips)


def _export_json(path: Path, rows: list[OrderBookStrategyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_row_payload(row, rank) for rank, row in enumerate(rows, start=1)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("order_book_strategy_json_exported", path=str(path), rows=len(payload))


def _export_csv(path: Path, rows: list[OrderBookStrategyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "timeframe",
        "segment",
        "strategy_name",
        "feature",
        "horizon_bars",
        "entry_quantile",
        "entry_threshold",
        "sample_size",
        "verdict",
        "return_pct",
        "final_equity",
        "round_trips",
        "win_rate",
        "profit_factor",
        "expectancy",
        "max_drawdown_pct",
        "buy_hold_order_sized_return_pct",
        "beats_no_trade",
        "beats_order_sized_buy_hold",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            payload = _row_payload(row, rank)
            writer.writerow(
                {
                    "rank": rank,
                    "timeframe": row.timeframe,
                    "segment": row.segment,
                    "strategy_name": row.strategy_name,
                    "feature": row.feature,
                    "horizon_bars": row.horizon_bars,
                    "entry_quantile": str(row.entry_quantile),
                    "entry_threshold": row.entry_threshold,
                    "sample_size": row.sample_size,
                    "verdict": payload["verdict"],
                    "return_pct": payload["metrics"]["return_pct"],
                    "final_equity": payload["metrics"]["final_equity"],
                    "round_trips": payload["metrics"]["round_trips"],
                    "win_rate": payload["metrics"]["win_rate"],
                    "profit_factor": payload["edge"]["profit_factor"],
                    "expectancy": payload["edge"]["expectancy"],
                    "max_drawdown_pct": payload["edge"]["max_drawdown_pct"],
                    "buy_hold_order_sized_return_pct": payload["benchmark"]["buy_hold_order_sized_return_pct"],
                    "beats_no_trade": payload["benchmark"]["beats_no_trade"],
                    "beats_order_sized_buy_hold": payload["benchmark"]["beats_order_sized_buy_hold"],
                    "notes": row.notes,
                }
            )
    logger.info("order_book_strategy_csv_exported", path=str(path), rows=len(rows))


async def _load_feature_rows(args: argparse.Namespace) -> tuple[list[MarketFeatures], str]:
    settings = get_settings()
    market_data_source, _use_testnet, exchange_id = _resolve_market_data_source(args.market_data_source)
    db = Database(settings.database_url)
    await db.connect()
    try:
        repository = TradingRepository(db)
        rows = await repository.load_market_features(
            exchange=exchange_id,
            symbol=settings.normalized_symbol,
            timeframe=args.timeframe,
            limit=args.limit,
        )
    finally:
        await db.close()
    logger.info(
        "order_book_strategy_features_loaded",
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        symbol=settings.normalized_symbol,
        timeframe=args.timeframe,
        rows=len(rows),
    )
    return rows, settings.normalized_symbol


def _build_rows_for_feature(
    *,
    feature_rows: list[MarketFeatures],
    feature: str,
    horizons: list[int],
    quantiles: list[Decimal],
    args: argparse.Namespace,
    symbol: str,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> list[OrderBookStrategyRow]:
    settings = get_settings()
    available = rows_with_feature(feature_rows, feature)
    if len(available) < args.min_feature_samples:
        logger.info(
            "order_book_strategy_feature_skipped",
            feature=feature,
            sample_size=len(available),
            min_feature_samples=args.min_feature_samples,
            reason="not_enough_samples",
        )
        return []

    train_rows, validation_rows = split_feature_rows(available, train_ratio=args.train_ratio)
    segments = {
        "full": available,
        "train": train_rows,
        "validation": validation_rows,
    }
    output: list[OrderBookStrategyRow] = []
    for quantile in quantiles:
        threshold = quantile_threshold(train_rows, feature, float(quantile))
        for horizon in horizons:
            config = OrderBookThresholdConfig(
                feature_name=feature,
                entry_threshold=threshold,
                horizon_bars=horizon,
                strategy_name=_make_strategy_name(feature, quantile, horizon),
            )
            for segment, rows in segments.items():
                if len(rows) < args.min_feature_samples and segment != "validation":
                    continue
                result = run_order_book_threshold_backtest(
                    rows=rows,
                    config=config,
                    symbol=symbol,
                    initial_quote_balance=settings.initial_quote_balance,
                    quote_amount=settings.max_order_usdt,
                    fee_rate_pct=fee_rate_pct,
                    slippage_pct=slippage_pct,
                )
                buy_hold = buy_and_hold_feature_rows(
                    rows=rows,
                    symbol=symbol,
                    initial_quote_balance=settings.initial_quote_balance,
                    quote_amount=settings.max_order_usdt,
                    fee_rate_pct=fee_rate_pct,
                    slippage_pct=slippage_pct,
                )
                output.append(
                    OrderBookStrategyRow(
                        timeframe=args.timeframe,
                        segment=segment,
                        strategy_name=config.strategy_name,
                        feature=feature,
                        horizon_bars=horizon,
                        entry_quantile=quantile,
                        entry_threshold=threshold,
                        sample_size=len(rows),
                        strategy_result=result,
                        buy_hold_result=buy_hold,
                        notes=(
                            "Research-only fixed-horizon long. Feature at candle i triggers entry at candle i+1; "
                            "threshold learned from train rows only."
                        ),
                    )
                )
    return output


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.min_feature_samples <= 0:
        raise SystemExit("--min-feature-samples must be positive")
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise SystemExit("--train-ratio must be greater than 0 and smaller than 1")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("--fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("--slippage-pct cannot be negative")

    features = _parse_csv_list(args.features)
    horizons = _parse_ints(args.horizons)
    quantiles = _parse_quantiles(args.entry_quantiles)
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct
    feature_rows, symbol = await _load_feature_rows(args)

    logger.info(
        "order_book_strategy_backtest_started",
        timeframe=args.timeframe,
        features=",".join(features),
        horizons=",".join(str(item) for item in horizons),
        entry_quantiles=",".join(str(item) for item in quantiles),
        min_feature_samples=args.min_feature_samples,
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        note="research only; no live trading; no profit claim",
    )

    rows: list[OrderBookStrategyRow] = []
    for feature in features:
        rows.extend(
            _build_rows_for_feature(
                feature_rows=feature_rows,
                feature=feature,
                horizons=horizons,
                quantiles=quantiles,
                args=args,
                symbol=symbol,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
        )

    if not rows:
        raise SystemExit("No order-book strategy rows produced. Check feature coverage / min samples.")

    ranked = sorted(rows, key=_sort_key, reverse=True)
    logger.info("order_book_strategy_backtest_finished", rows=len(ranked), note="research only")
    for rank, row in enumerate(ranked[: args.top], start=1):
        payload = _row_payload(row, rank)
        logger.info(
            "order_book_strategy_result",
            rank=rank,
            timeframe=row.timeframe,
            segment=row.segment,
            strategy_name=row.strategy_name,
            feature=row.feature,
            horizon_bars=row.horizon_bars,
            entry_quantile=str(row.entry_quantile),
            entry_threshold=round(row.entry_threshold, 6),
            sample_size=row.sample_size,
            verdict=payload["verdict"],
            return_pct=payload["metrics"]["return_pct"],
            final_equity=payload["metrics"]["final_equity"],
            round_trips=payload["metrics"]["round_trips"],
            win_rate=round(payload["metrics"]["win_rate"], 4),
            profit_factor=payload["edge"]["profit_factor"],
            buy_hold_return_pct=payload["benchmark"]["buy_hold_order_sized_return_pct"],
            beats_buy_hold=payload["benchmark"]["beats_order_sized_buy_hold"],
        )

    if args.export_json:
        _export_json(args.export_json, ranked)
    if args.export_csv:
        _export_csv(args.export_csv, ranked)


if __name__ == "__main__":
    asyncio.run(main())
