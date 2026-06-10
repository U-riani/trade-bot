from __future__ import annotations

from decimal import Decimal

from app.market.indicators import atr, ema, rsi
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


class EmaRsiStrategy(Strategy):
    name = "ema_rsi_v2_filtered"

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        rsi_period: int = 14,
        rsi_buy_min: float = 45,
        rsi_buy_max: float = 70,
        rsi_sell_min: float = 75,
        suggested_quote_amount: Decimal | None = None,
        trend_ema_period: int | None = None,
        min_ema_gap_pct: Decimal = Decimal("0"),
        atr_period: int | None = None,
        min_atr_pct: Decimal = Decimal("0"),
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        if trend_ema_period is not None and trend_ema_period <= slow_period:
            raise ValueError("trend_ema_period must be greater than slow_period when enabled")
        if min_ema_gap_pct < 0:
            raise ValueError("min_ema_gap_pct cannot be negative")
        if atr_period is not None and atr_period <= 0:
            raise ValueError("atr_period must be positive when enabled")
        if min_atr_pct < 0:
            raise ValueError("min_atr_pct cannot be negative")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_buy_min = rsi_buy_min
        self.rsi_buy_max = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min
        self.suggested_quote_amount = suggested_quote_amount
        self.trend_ema_period = trend_ema_period
        self.min_ema_gap_pct = min_ema_gap_pct
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        candles = list(market_state.candles)
        closes = market_state.get_closes()
        minimum = max(
            self.slow_period,
            self.rsi_period,
            self.trend_ema_period or 0,
            self.atr_period or 0,
        ) + 2
        if len(closes) < minimum:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=market_state.symbol,
                reason=f"not_enough_candles: have={len(closes)}, need={minimum}",
            )

        fast = ema(closes, self.fast_period)
        slow = ema(closes, self.slow_period)
        current_rsi = rsi(closes, self.rsi_period)

        if current_rsi is None:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=market_state.symbol,
                reason="rsi_not_available",
            )

        prev_fast, current_fast = fast[-2], fast[-1]
        prev_slow, current_slow = slow[-2], slow[-1]
        current_close = closes[-1]

        crossed_above = prev_fast <= prev_slow and current_fast > current_slow
        crossed_below = prev_fast >= prev_slow and current_fast < current_slow
        ema_gap_pct = self._percentage_gap(current_fast, current_slow)

        if crossed_above and self.rsi_buy_min <= current_rsi <= self.rsi_buy_max:
            blocked_reason = self._buy_filter_block_reason(
                closes=closes,
                candles=candles,
                current_close=current_close,
                current_fast=current_fast,
                current_slow=current_slow,
                ema_gap_pct=ema_gap_pct,
            )
            if blocked_reason is not None:
                return TradeSignal.hold(
                    strategy_name=self.name,
                    symbol=market_state.symbol,
                    reason=blocked_reason,
                )

            return TradeSignal(
                strategy_name=self.name,
                symbol=market_state.symbol,
                side=SignalSide.BUY,
                confidence=0.72,
                reason=(
                    f"ema_cross_above: ema{self.fast_period}={current_fast:.6f}, "
                    f"ema{self.slow_period}={current_slow:.6f}, rsi={current_rsi:.2f}, "
                    f"ema_gap_pct={ema_gap_pct:.4f}"
                ),
                created_at=utc_now(),
                suggested_quote_amount=self.suggested_quote_amount,
            )

        if crossed_below or current_rsi >= self.rsi_sell_min:
            return TradeSignal(
                strategy_name=self.name,
                symbol=market_state.symbol,
                side=SignalSide.SELL,
                confidence=0.66,
                reason=(
                    f"sell_condition: crossed_below={crossed_below}, "
                    f"rsi={current_rsi:.2f}, ema{self.fast_period}={current_fast:.6f}, "
                    f"ema{self.slow_period}={current_slow:.6f}"
                ),
                created_at=utc_now(),
            )

        return TradeSignal.hold(
            strategy_name=self.name,
            symbol=market_state.symbol,
            reason=(
                f"no_signal: ema{self.fast_period}={current_fast:.6f}, "
                f"ema{self.slow_period}={current_slow:.6f}, rsi={current_rsi:.2f}, "
                f"ema_gap_pct={ema_gap_pct:.4f}"
            ),
        )

    def _buy_filter_block_reason(
        self,
        *,
        closes: list[float],
        candles: list,
        current_close: float,
        current_fast: float,
        current_slow: float,
        ema_gap_pct: float,
    ) -> str | None:
        if self.min_ema_gap_pct > 0 and Decimal(str(ema_gap_pct)) < self.min_ema_gap_pct:
            return (
                f"buy_filtered_min_ema_gap: gap_pct={ema_gap_pct:.4f}, "
                f"required={self.min_ema_gap_pct}"
            )

        if self.trend_ema_period is not None:
            trend = ema(closes, self.trend_ema_period)[-1]
            if current_close <= trend:
                return (
                    f"buy_filtered_trend: close={current_close:.6f}, "
                    f"ema{self.trend_ema_period}={trend:.6f}"
                )
            if current_slow <= trend:
                return (
                    f"buy_filtered_slow_below_trend: ema{self.slow_period}={current_slow:.6f}, "
                    f"ema{self.trend_ema_period}={trend:.6f}"
                )

        if self.atr_period is not None and self.min_atr_pct > 0:
            current_atr = atr(candles, self.atr_period)
            if current_atr is None:
                return "buy_filtered_atr_not_available"
            atr_pct = (Decimal(str(current_atr)) / Decimal(str(current_close))) * Decimal("100")
            if atr_pct < self.min_atr_pct:
                return f"buy_filtered_min_atr: atr_pct={atr_pct:.4f}, required={self.min_atr_pct}"

        return None

    @staticmethod
    def _percentage_gap(current_fast: float, current_slow: float) -> float:
        if current_slow == 0:
            return 0.0
        return abs(current_fast - current_slow) / abs(current_slow) * 100
