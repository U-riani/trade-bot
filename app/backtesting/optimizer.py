from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import product

from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import BacktestMetrics
from app.market.models import Candle
from app.strategy.ema_rsi import EmaRsiStrategy


@dataclass(slots=True, frozen=True)
class StrategyParameterSet:
    ema_fast_period: int
    ema_slow_period: int
    rsi_period: int
    rsi_buy_min: float
    rsi_buy_max: float
    rsi_sell_min: float
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    trend_ema_period: int | None = None
    min_ema_gap_pct: Decimal = Decimal("0")
    atr_period: int | None = None
    min_atr_pct: Decimal = Decimal("0")

    @property
    def key(self) -> str:
        trend = f"trend{self.trend_ema_period}" if self.trend_ema_period else "trend0"
        atr_filter = f"atr{self.atr_period}_{self.min_atr_pct}" if self.atr_period else "atr0_0"
        return (
            f"ema{self.ema_fast_period}_{self.ema_slow_period}__"
            f"rsi{self.rsi_period}_{self.rsi_buy_min:g}-{self.rsi_buy_max:g}_sell{self.rsi_sell_min:g}__"
            f"sl{self.stop_loss_pct}_tp{self.take_profit_pct}__"
            f"{trend}_gap{self.min_ema_gap_pct}_{atr_filter}"
        )


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    rank: int
    parameters: StrategyParameterSet
    metrics: BacktestMetrics

    @property
    def score(self) -> Decimal:
        """Default score used for sorting optimization results.

        Final equity is the primary target. Drawdown, fees, and excessive trade
        count get small penalties so the optimizer prefers quieter parameter
        sets when equity is similar. Otherwise it happily discovers a machine
        that trades itself to death, because apparently computers enjoy fees too.
        """
        return (
            self.metrics.final_equity
            - (self.metrics.max_drawdown * Decimal("0.05"))
            - (self.metrics.total_fees * Decimal("0.05"))
            - (Decimal(self.metrics.round_trips) * Decimal("0.001"))
        )


def generate_parameter_sets(
    *,
    ema_fast_values: list[int],
    ema_slow_values: list[int],
    rsi_period_values: list[int],
    rsi_buy_min_values: list[float],
    rsi_buy_max_values: list[float],
    rsi_sell_min_values: list[float],
    stop_loss_pct_values: list[Decimal],
    take_profit_pct_values: list[Decimal],
    trend_ema_period_values: list[int | None] | None = None,
    min_ema_gap_pct_values: list[Decimal] | None = None,
    atr_period_values: list[int | None] | None = None,
    min_atr_pct_values: list[Decimal] | None = None,
) -> list[StrategyParameterSet]:
    parameter_sets: list[StrategyParameterSet] = []
    trend_values = trend_ema_period_values or [None]
    gap_values = min_ema_gap_pct_values or [Decimal("0")]
    atr_values = atr_period_values or [None]
    min_atr_values = min_atr_pct_values or [Decimal("0")]

    for (
        fast,
        slow,
        rsi_period,
        buy_min,
        buy_max,
        sell_min,
        stop_loss,
        take_profit,
        trend_ema_period,
        min_ema_gap_pct,
        atr_period,
        min_atr_pct,
    ) in product(
        ema_fast_values,
        ema_slow_values,
        rsi_period_values,
        rsi_buy_min_values,
        rsi_buy_max_values,
        rsi_sell_min_values,
        stop_loss_pct_values,
        take_profit_pct_values,
        trend_values,
        gap_values,
        atr_values,
        min_atr_values,
    ):
        if fast >= slow:
            continue
        if buy_min >= buy_max:
            continue
        if not 0 <= buy_min <= 100 or not 0 <= buy_max <= 100 or not 0 <= sell_min <= 100:
            continue
        if stop_loss <= 0 or take_profit <= 0:
            continue
        if trend_ema_period is not None and trend_ema_period <= slow:
            continue
        if min_ema_gap_pct < 0:
            continue
        if atr_period is None and min_atr_pct > 0:
            continue
        if atr_period is not None and atr_period <= 0:
            continue
        if min_atr_pct < 0:
            continue

        parameter_sets.append(
            StrategyParameterSet(
                ema_fast_period=fast,
                ema_slow_period=slow,
                rsi_period=rsi_period,
                rsi_buy_min=buy_min,
                rsi_buy_max=buy_max,
                rsi_sell_min=sell_min,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                trend_ema_period=trend_ema_period,
                min_ema_gap_pct=min_ema_gap_pct,
                atr_period=atr_period,
                min_atr_pct=min_atr_pct,
            )
        )

    return parameter_sets


def optimize_parameter_grid(
    *,
    candles: list[Candle],
    symbol: str,
    parameter_sets: list[StrategyParameterSet],
    initial_quote_balance: Decimal,
    max_order_usdt: Decimal,
    max_position_usdt: Decimal,
    allow_only_one_open_position: bool,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
    min_round_trips: int = 0,
) -> list[OptimizationResult]:
    ranked_candidates: list[OptimizationResult] = []

    for parameters in parameter_sets:
        strategy = EmaRsiStrategy(
            fast_period=parameters.ema_fast_period,
            slow_period=parameters.ema_slow_period,
            rsi_period=parameters.rsi_period,
            rsi_buy_min=parameters.rsi_buy_min,
            rsi_buy_max=parameters.rsi_buy_max,
            rsi_sell_min=parameters.rsi_sell_min,
            suggested_quote_amount=max_order_usdt,
            trend_ema_period=parameters.trend_ema_period,
            min_ema_gap_pct=parameters.min_ema_gap_pct,
            atr_period=parameters.atr_period,
            min_atr_pct=parameters.min_atr_pct,
        )
        engine = BacktestEngine(
            strategy=strategy,
            symbol=symbol,
            initial_quote_balance=initial_quote_balance,
            max_order_usdt=max_order_usdt,
            max_position_usdt=max_position_usdt,
            stop_loss_pct=parameters.stop_loss_pct,
            take_profit_pct=parameters.take_profit_pct,
            allow_only_one_open_position=allow_only_one_open_position,
            fee_rate_pct=fee_rate_pct,
            slippage_pct=slippage_pct,
        )
        result = engine.run(candles)
        if result.metrics.round_trips < min_round_trips:
            continue

        ranked_candidates.append(
            OptimizationResult(
                rank=0,
                parameters=parameters,
                metrics=result.metrics,
            )
        )

    sorted_candidates = sorted(
        ranked_candidates,
        key=lambda item: (
            item.score,
            item.metrics.final_equity,
            -item.metrics.max_drawdown,
            -item.metrics.round_trips,
            item.metrics.win_rate,
        ),
        reverse=True,
    )

    return [
        OptimizationResult(rank=index, parameters=item.parameters, metrics=item.metrics)
        for index, item in enumerate(sorted_candidates, start=1)
    ]
