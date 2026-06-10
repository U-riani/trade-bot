from __future__ import annotations

from decimal import Decimal

from app.market.indicators import atr, ema
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


class BreakoutMomentumStrategy(Strategy):
    """Simple long-only breakout strategy for V18 research comparisons.

    The strategy buys when the latest close breaks above the previous N-candle
    high, optionally filtered by a long trend EMA and minimum ATR percent. It
    exits when the latest close breaks below the previous exit-lookback low.

    This is intentionally boring. Boring is good. Boring at least has the decency
    to fail in a way we can measure instead of mystically optimizing EMA/RSI into
    a decorative spreadsheet bonfire.
    """

    name = "breakout_momentum_v1"

    def __init__(
        self,
        *,
        breakout_lookback: int = 20,
        exit_lookback: int = 10,
        suggested_quote_amount: Decimal | None = None,
        trend_ema_period: int | None = 200,
        atr_period: int | None = 14,
        min_atr_pct: Decimal = Decimal("0.08"),
        min_breakout_pct: Decimal = Decimal("0"),
    ) -> None:
        if breakout_lookback <= 1:
            raise ValueError("breakout_lookback must be greater than 1")
        if exit_lookback <= 1:
            raise ValueError("exit_lookback must be greater than 1")
        if trend_ema_period is not None and trend_ema_period <= breakout_lookback:
            raise ValueError("trend_ema_period must be greater than breakout_lookback when enabled")
        if atr_period is not None and atr_period <= 0:
            raise ValueError("atr_period must be positive when enabled")
        if min_atr_pct < 0:
            raise ValueError("min_atr_pct cannot be negative")
        if min_breakout_pct < 0:
            raise ValueError("min_breakout_pct cannot be negative")

        self.breakout_lookback = breakout_lookback
        self.exit_lookback = exit_lookback
        self.suggested_quote_amount = suggested_quote_amount
        self.trend_ema_period = trend_ema_period
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct
        self.min_breakout_pct = min_breakout_pct

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        candles = list(market_state.candles)
        closes = market_state.get_closes()
        minimum = max(
            self.breakout_lookback,
            self.exit_lookback,
            self.trend_ema_period or 0,
            self.atr_period or 0,
        ) + 1
        if len(candles) < minimum:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=market_state.symbol,
                reason=f"not_enough_candles: have={len(candles)}, need={minimum}",
            )

        current = candles[-1]
        current_close = Decimal(str(current.close))
        previous_breakout_window = candles[-(self.breakout_lookback + 1) : -1]
        previous_exit_window = candles[-(self.exit_lookback + 1) : -1]
        previous_high = max(Decimal(str(candle.high)) for candle in previous_breakout_window)
        previous_low = min(Decimal(str(candle.low)) for candle in previous_exit_window)

        breakout_threshold = previous_high * (Decimal("1") + self.min_breakout_pct / Decimal("100"))
        if current_close > breakout_threshold:
            blocked_reason = self._buy_filter_block_reason(
                candles=candles,
                closes=closes,
                current_close=current_close,
            )
            if blocked_reason is not None:
                return TradeSignal.hold(
                    strategy_name=self.name,
                    symbol=market_state.symbol,
                    reason=blocked_reason,
                )

            breakout_pct = ((current_close - previous_high) / previous_high) * Decimal("100")
            return TradeSignal(
                strategy_name=self.name,
                symbol=market_state.symbol,
                side=SignalSide.BUY,
                confidence=0.68,
                reason=(
                    f"breakout_buy: close={current_close}, previous_high={previous_high}, "
                    f"breakout_pct={breakout_pct:.4f}"
                ),
                created_at=utc_now(),
                suggested_quote_amount=self.suggested_quote_amount,
            )

        if current_close < previous_low:
            breakdown_pct = ((previous_low - current_close) / previous_low) * Decimal("100")
            return TradeSignal(
                strategy_name=self.name,
                symbol=market_state.symbol,
                side=SignalSide.SELL,
                confidence=0.62,
                reason=(
                    f"breakout_exit_low_break: close={current_close}, previous_low={previous_low}, "
                    f"breakdown_pct={breakdown_pct:.4f}"
                ),
                created_at=utc_now(),
            )

        return TradeSignal.hold(
            strategy_name=self.name,
            symbol=market_state.symbol,
            reason=(
                f"no_breakout: close={current_close}, previous_high={previous_high}, "
                f"previous_low={previous_low}"
            ),
        )

    def _buy_filter_block_reason(
        self,
        *,
        candles: list,
        closes: list[float],
        current_close: Decimal,
    ) -> str | None:
        if self.trend_ema_period is not None:
            trend = Decimal(str(ema(closes, self.trend_ema_period)[-1]))
            if current_close <= trend:
                return f"buy_filtered_trend: close={current_close}, ema{self.trend_ema_period}={trend}"

        if self.atr_period is not None and self.min_atr_pct > 0:
            current_atr = atr(candles, self.atr_period)
            if current_atr is None:
                return "buy_filtered_atr_not_available"
            atr_pct = (Decimal(str(current_atr)) / current_close) * Decimal("100")
            if atr_pct < self.min_atr_pct:
                return f"buy_filtered_min_atr: atr_pct={atr_pct:.4f}, required={self.min_atr_pct}"

        return None
