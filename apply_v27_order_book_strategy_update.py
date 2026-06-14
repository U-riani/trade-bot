from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()

FILES: dict[str, str] = {}

FILES["app/backtesting/order_book_strategy.py"] = r'''"""V27 research-only order-book threshold strategy backtester.

This module intentionally does not place orders, emit live signals, or claim an
edge. It turns the V26/V26.1 feature research into a small, brutally simple
question:

    If a live order-book imbalance feature is high at a candle close, does a
    fixed-horizon long trade beat no-trade / order-sized buy-and-hold after fees
    and slippage?

The design is deliberately conservative about time:
- Feature value at row i is considered known only after that candle closes.
- Entry happens on row i+1, never on the same row that produced the signal.
- Exit happens after a fixed number of future bars.

So yes, we do the boring no-lookahead thing, because reality already charges
fees and does not need help cheating.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import ceil

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.market.features import MarketFeatures


@dataclass(slots=True, frozen=True)
class OrderBookThresholdConfig:
    """Config for one research-only long threshold strategy."""

    feature_name: str
    entry_threshold: float
    horizon_bars: int
    strategy_name: str


@dataclass(slots=True)
class _OpenPosition:
    entry_index: int
    exit_index: int
    entry_time: object
    entry_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    entry_reason: str


def feature_value(row: MarketFeatures, feature_name: str) -> float | None:
    """Return a numeric feature value from a MarketFeatures row.

    Unknown feature names raise early instead of quietly returning None and making
    a strategy look like it responsibly decided not to trade. Computers love that
    kind of passive-aggressive failure.
    """

    if not hasattr(row, feature_name):
        raise ValueError(f"Unknown market feature: {feature_name}")
    value = getattr(row, feature_name)
    if value is None:
        return None
    return float(value)


def rows_with_feature(rows: list[MarketFeatures], feature_name: str) -> list[MarketFeatures]:
    """Rows sorted by close_time where the requested feature is present."""

    return sorted((row for row in rows if feature_value(row, feature_name) is not None), key=lambda row: row.close_time)


def quantile_threshold(rows: list[MarketFeatures], feature_name: str, quantile: float) -> float:
    """Nearest-rank quantile threshold for a feature.

    The threshold is intended to be learned from train rows and reused for
    validation/full runs. That prevents the classic "I optimized on the future"
    backtest circus.
    """

    if quantile <= 0 or quantile >= 1:
        raise ValueError("quantile must be between 0 and 1")
    values = sorted(value for row in rows if (value := feature_value(row, feature_name)) is not None)
    if not values:
        raise ValueError(f"No rows with feature: {feature_name}")
    index = max(0, min(len(values) - 1, ceil(len(values) * quantile) - 1))
    return values[index]


def split_feature_rows(rows: list[MarketFeatures], *, train_ratio: Decimal) -> tuple[list[MarketFeatures], list[MarketFeatures]]:
    """Chronological train/validation split for feature rows."""

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be greater than 0 and smaller than 1")
    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    split_index = int(len(sorted_rows) * float(train_ratio))
    if split_index <= 0 or split_index >= len(sorted_rows):
        raise ValueError("not enough rows to split train/validation sets")
    return sorted_rows[:split_index], sorted_rows[split_index:]


def run_order_book_threshold_backtest(
    *,
    rows: list[MarketFeatures],
    config: OrderBookThresholdConfig,
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestResult:
    """Run one fixed-horizon long threshold strategy on feature rows.

    Feature row i triggers a BUY scheduled for row i+1. This is important: the
    feature is known only after row i closes, so buying at the same row's close is
    lookahead-ish optimism with nicer shoes.
    """

    if config.horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if initial_quote_balance <= 0:
        raise ValueError("initial_quote_balance must be positive")
    if quote_amount <= 0:
        raise ValueError("quote_amount must be positive")
    if fee_rate_pct < 0:
        raise ValueError("fee_rate_pct cannot be negative")
    if slippage_pct < 0:
        raise ValueError("slippage_pct cannot be negative")

    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    if not sorted_rows:
        return _empty_result(initial_quote_balance=initial_quote_balance)

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")

    quote_balance = initial_quote_balance
    realized_pnl = Decimal("0")
    total_fees = Decimal("0")
    position: _OpenPosition | None = None
    trades: list[BacktestTrade] = []
    equity_curve: list[Decimal] = []
    equity_peak = initial_quote_balance
    max_drawdown = Decimal("0")
    scheduled_buy_index: int | None = None
    scheduled_reason = ""
    executed_orders = 0

    for index, row in enumerate(sorted_rows):
        close_price = Decimal(str(row.close_price))

        if position is None and scheduled_buy_index == index:
            spend = _spend_amount(quote_balance=quote_balance, requested=quote_amount, fee_rate=fee_rate)
            if spend > 0 and close_price > 0:
                entry_price = close_price * (Decimal("1") + slippage_rate)
                entry_fee = spend * fee_rate
                quantity = spend / entry_price
                quote_balance -= spend + entry_fee
                realized_pnl -= entry_fee
                total_fees += entry_fee
                executed_orders += 1
                position = _OpenPosition(
                    entry_index=index,
                    exit_index=min(index + config.horizon_bars, len(sorted_rows) - 1),
                    entry_time=row.close_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    quote_amount=spend,
                    entry_fee=entry_fee,
                    entry_reason=scheduled_reason or f"{config.feature_name}>={config.entry_threshold}",
                )
            scheduled_buy_index = None
            scheduled_reason = ""

        if position is not None and index >= position.exit_index and index > position.entry_index:
            exit_price = close_price * (Decimal("1") - slippage_rate)
            gross_quote = position.quantity * exit_price
            exit_fee = gross_quote * fee_rate
            net_quote = gross_quote - exit_fee
            cost_basis = position.quantity * position.entry_price
            pnl = net_quote - cost_basis - position.entry_fee

            quote_balance += net_quote
            realized_pnl += gross_quote - cost_basis - exit_fee
            total_fees += exit_fee
            executed_orders += 1
            trades.append(
                BacktestTrade(
                    symbol=symbol,
                    entry_time=position.entry_time,  # type: ignore[arg-type]
                    exit_time=row.close_time,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    quote_amount=position.quote_amount,
                    entry_fee=position.entry_fee,
                    exit_fee=exit_fee,
                    pnl=pnl,
                    entry_reason=position.entry_reason,
                    exit_reason=f"fixed_horizon_{config.horizon_bars}_bars",
                )
            )
            position = None

        if position is None and scheduled_buy_index is None and index + 1 < len(sorted_rows):
            value = feature_value(row, config.feature_name)
            if value is not None and value >= config.entry_threshold:
                scheduled_buy_index = index + 1
                scheduled_reason = f"{config.feature_name}={value:.6f} >= threshold={config.entry_threshold:.6f}"

        current_equity = quote_balance + ((position.quantity * close_price) if position is not None else Decimal("0"))
        equity_curve.append(current_equity)
        if current_equity > equity_peak:
            equity_peak = current_equity
        drawdown = equity_peak - current_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    last_price = Decimal(str(sorted_rows[-1].close_price))
    open_qty = position.quantity if position is not None else Decimal("0")
    open_avg = position.entry_price if position is not None else Decimal("0")
    unrealized_pnl = (last_price - open_avg) * open_qty if position is not None else Decimal("0")
    final_equity = quote_balance + (open_qty * last_price)

    metrics = BacktestMetrics(
        candles_processed=len(sorted_rows),
        executed_orders=executed_orders,
        round_trips=len(trades),
        winning_trades=sum(1 for trade in trades if trade.pnl > 0),
        losing_trades=sum(1 for trade in trades if trade.pnl < 0),
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_fees=total_fees,
        max_drawdown=max_drawdown,
        initial_equity=initial_quote_balance,
        final_equity=final_equity,
        open_position_quantity=open_qty,
        open_position_avg_entry_price=open_avg,
        last_price=last_price,
    )
    return BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve)


def buy_and_hold_feature_rows(
    *,
    rows: list[MarketFeatures],
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestResult:
    """Order-sized buy-and-hold benchmark over the same feature-row window."""

    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    if not sorted_rows:
        return _empty_result(initial_quote_balance=initial_quote_balance)

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    spend = _spend_amount(quote_balance=initial_quote_balance, requested=quote_amount, fee_rate=fee_rate)
    first = sorted_rows[0]
    last = sorted_rows[-1]
    entry_price = Decimal(str(first.close_price)) * (Decimal("1") + slippage_rate)
    last_price = Decimal(str(last.close_price))
    entry_fee = spend * fee_rate
    quantity = spend / entry_price if entry_price > 0 else Decimal("0")
    quote_balance = initial_quote_balance - spend - entry_fee
    final_equity = quote_balance + (quantity * last_price)
    unrealized = (last_price - entry_price) * quantity
    equity_curve = [quote_balance + (quantity * Decimal(str(row.close_price))) for row in sorted_rows]
    max_drawdown = _absolute_max_drawdown(equity_curve, initial_quote_balance)

    metrics = BacktestMetrics(
        candles_processed=len(sorted_rows),
        executed_orders=1 if quantity > 0 else 0,
        round_trips=0,
        winning_trades=0,
        losing_trades=0,
        realized_pnl=-entry_fee,
        unrealized_pnl=unrealized,
        total_fees=entry_fee,
        max_drawdown=max_drawdown,
        initial_equity=initial_quote_balance,
        final_equity=final_equity,
        open_position_quantity=quantity,
        open_position_avg_entry_price=entry_price,
        last_price=last_price,
    )
    return BacktestResult(metrics=metrics, trades=[], equity_curve=equity_curve)


def _spend_amount(*, quote_balance: Decimal, requested: Decimal, fee_rate: Decimal) -> Decimal:
    max_spend = quote_balance / (Decimal("1") + fee_rate) if fee_rate > 0 else quote_balance
    return min(requested, max_spend)


def _empty_result(*, initial_quote_balance: Decimal) -> BacktestResult:
    metrics = BacktestMetrics(
        candles_processed=0,
        executed_orders=0,
        round_trips=0,
        winning_trades=0,
        losing_trades=0,
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        total_fees=Decimal("0"),
        max_drawdown=Decimal("0"),
        initial_equity=initial_quote_balance,
        final_equity=initial_quote_balance,
        open_position_quantity=Decimal("0"),
        open_position_avg_entry_price=Decimal("0"),
        last_price=None,
    )
    return BacktestResult(metrics=metrics, trades=[], equity_curve=[])


def _absolute_max_drawdown(equity_curve: list[Decimal], fallback_peak: Decimal) -> Decimal:
    if not equity_curve:
        return Decimal("0")
    peak = max(fallback_peak, equity_curve[0])
    max_drawdown = Decimal("0")
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown
'''

FILES["scripts/backtest_order_book_strategy.py"] = r'''"""V27 research-only backtest for order-book imbalance threshold candidates.

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
'''

FILES["tests/test_order_book_strategy.py"] = r'''from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.backtesting.order_book_strategy import (
    OrderBookThresholdConfig,
    feature_value,
    quantile_threshold,
    rows_with_feature,
    run_order_book_threshold_backtest,
    split_feature_rows,
)
from app.market.features import MarketFeatures


def _row(index: int, *, close: float = 100.0, imbalance: float | None = None) -> MarketFeatures:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * index)
    return MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="5m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=5) - timedelta(milliseconds=1),
        close_price=close,
        volume=1.0,
        imbalance_top_20=imbalance,
    )


def test_quantile_threshold_nearest_rank() -> None:
    rows = [_row(i, imbalance=value) for i, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    assert quantile_threshold(rows, "imbalance_top_20", 0.8) == pytest.approx(0.4)


def test_rows_with_feature_filters_missing_and_sorts() -> None:
    rows = [_row(2, imbalance=0.3), _row(1, imbalance=None), _row(0, imbalance=0.1)]
    filtered = rows_with_feature(rows, "imbalance_top_20")
    assert [row.close_price for row in filtered] == [100.0, 100.0]
    assert [feature_value(row, "imbalance_top_20") for row in filtered] == [0.1, 0.3]


def test_unknown_feature_raises() -> None:
    with pytest.raises(ValueError, match="Unknown market feature"):
        feature_value(_row(0, imbalance=0.1), "definitely_not_real")


def test_strategy_enters_next_row_not_signal_row_and_exits_after_horizon() -> None:
    rows = [
        _row(0, close=100, imbalance=0.9),  # signal only
        _row(1, close=101, imbalance=0.1),  # entry here
        _row(2, close=102, imbalance=0.1),
        _row(3, close=104, imbalance=0.1),  # exit here with horizon 2
    ]
    result = run_order_book_threshold_backtest(
        rows=rows,
        config=OrderBookThresholdConfig(
            feature_name="imbalance_top_20",
            entry_threshold=0.8,
            horizon_bars=2,
            strategy_name="test_ob_threshold",
        ),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )
    assert result.metrics.round_trips == 1
    trade = result.trades[0]
    assert trade.entry_time == rows[1].close_time
    assert trade.exit_time == rows[3].close_time
    assert trade.entry_price == Decimal("101")
    assert trade.exit_price == Decimal("104")
    assert trade.pnl > 0


def test_strategy_applies_fee_and_slippage() -> None:
    rows = [_row(0, close=100, imbalance=0.9), _row(1, close=100, imbalance=0.1), _row(2, close=100, imbalance=0.1)]
    result = run_order_book_threshold_backtest(
        rows=rows,
        config=OrderBookThresholdConfig(
            feature_name="imbalance_top_20",
            entry_threshold=0.8,
            horizon_bars=1,
            strategy_name="test_ob_threshold",
        ),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0.1"),
        slippage_pct=Decimal("0.1"),
    )
    assert result.metrics.round_trips == 1
    assert result.metrics.total_fees > 0
    assert result.trades[0].pnl < 0


def test_split_feature_rows_chronological() -> None:
    rows = [_row(i, imbalance=0.1) for i in range(10)]
    train, validation = split_feature_rows(rows, train_ratio=Decimal("0.7"))
    assert len(train) == 7
    assert len(validation) == 3
    assert train[-1].close_time < validation[0].close_time
'''

README_MARKER = "## V27 order-book threshold strategy research prototype"
README_APPEND = r'''

## V27 order-book threshold strategy research prototype

V27 turns the V26/V26.1 feature candidates into a deliberately small
research-only strategy prototype. It does not place live orders, does not emit
live trading signals, and does not claim profitability. The goal is to answer a
narrow business question: is the order-book signal interesting enough to keep
building the bot?

The first candidate is based on the V26.1 report where 5m `imbalance_top_20`
was the strongest current feature candidate. The prototype uses a fixed-horizon
long rule:

1. Learn an entry threshold from the train portion of the available feature rows.
2. If a feature value at candle `i` is above the threshold, schedule a BUY at
   candle `i+1`.
3. Exit after a fixed number of future bars.
4. Compare against no-trade and order-sized buy-and-hold after fees/slippage.

Example workflow:

```powershell
python -m scripts.backfill_candles --market-data-source production --symbol BTCUSDT --timeframe 1m --limit 5000

python -m scripts.aggregate_order_book_features --market-data-source production --source db --symbol BTCUSDT --candle-limit 50000 --snapshot-limit 2000000 --timeframes 1m,5m,15m

python -m scripts.backtest_order_book_strategy --market-data-source production --timeframe 5m --limit 50000 --features imbalance_top_20,order_book_imbalance,imbalance_top_10,imbalance_top_5 --horizons 1,3,6 --entry-quantiles 0.6,0.7,0.8 --min-feature-samples 100 --export-json reports/order_book_strategy_v27.json --export-csv reports/order_book_strategy_v27.csv
```

Interpretation rule: a good V27 result is not permission to trade. It only means
we should continue to V27.1/V28 with stability checks, more samples, and stricter
walk-forward validation. A bad result means the current order-book idea probably
is not worth turning into execution logic yet.
'''

FILES["README.md"] = None  # handled specially


def write_file(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"updated {path}")


def update_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if README_MARKER not in text:
        text = text.rstrip() + README_APPEND + "\n"
        path.write_text(text, encoding="utf-8")
        print("updated README.md")
    else:
        print("README.md already has V27 section")


def main() -> None:
    for path, content in FILES.items():
        if path == "README.md":
            continue
        assert content is not None
        write_file(path, content)
    update_readme()
    print("V27 order-book strategy research update applied.")
    print("Run: python -m pytest -q")
    print(
        "Run: python -m scripts.backtest_order_book_strategy --market-data-source production --timeframe 5m --limit 50000 "
        "--features imbalance_top_20,order_book_imbalance,imbalance_top_10,imbalance_top_5 --horizons 1,3,6 "
        "--entry-quantiles 0.6,0.7,0.8 --min-feature-samples 100 "
        "--export-json reports/order_book_strategy_v27.json --export-csv reports/order_book_strategy_v27.csv"
    )


if __name__ == "__main__":
    main()
