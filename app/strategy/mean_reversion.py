from __future__ import annotations

from decimal import Decimal
from math import sqrt

from app.market.indicators import atr, ema, rsi
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


class MeanReversionStrategy(Strategy):
    """Long-only Bollinger/z-score mean-reversion strategy.

    Different signal *family* from the EMA/RSI and breakout strategies, which
    are both trend-following. Trend strategies buy strength and sell weakness;
    this one does the opposite: it buys statistically stretched dips and sells
    the reversion back toward the mean. On short BTC timeframes, price spends a
    lot of time chopping rather than trending, and that is exactly the regime
    where trend-following bleeds fees and mean-reversion has a fighting chance.

    Entry (BUY): the latest close is at least ``entry_z`` rolling standard
    deviations below the ``lookback`` mean, optionally confirmed by an oversold
    RSI, a minimum-volatility (ATR) floor, and a "dip inside an uptrend" trend
    filter so we are not just catching a falling knife with our face.

    Exit (SELL): the latest close has reverted back up to ``exit_z`` standard
    deviations of the mean (``exit_z=0`` means "back to the mean"). The risk
    manager's stop-loss/take-profit guard still runs on top of this.

    Nothing here is guaranteed to be profitable. It is guaranteed to fail
    differently from the trend strategies, which is the entire point of testing
    a second signal family instead of optimizing the first one into folklore.
    """

    name = "mean_reversion_v1"

    def __init__(
        self,
        *,
        lookback: int = 20,
        entry_z: Decimal = Decimal("2.0"),
        exit_z: Decimal = Decimal("0.0"),
        rsi_period: int | None = 14,
        rsi_buy_max: float | None = 35.0,
        trend_ema_period: int | None = None,
        atr_period: int | None = 14,
        min_atr_pct: Decimal = Decimal("0"),
        suggested_quote_amount: Decimal | None = None,
    ) -> None:
        if lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        if entry_z <= 0:
            raise ValueError("entry_z must be positive (how far below the mean to buy)")
        if exit_z < -entry_z:
            raise ValueError("exit_z must be greater than -entry_z")
        if rsi_period is not None and rsi_period <= 0:
            raise ValueError("rsi_period must be positive when enabled")
        if rsi_buy_max is not None and not 0 < rsi_buy_max < 100:
            raise ValueError("rsi_buy_max must be between 0 and 100 when enabled")
        if trend_ema_period is not None and trend_ema_period <= 1:
            raise ValueError("trend_ema_period must be greater than 1 when enabled")
        if atr_period is not None and atr_period <= 0:
            raise ValueError("atr_period must be positive when enabled")
        if min_atr_pct < 0:
            raise ValueError("min_atr_pct cannot be negative")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.rsi_period = rsi_period
        self.rsi_buy_max = rsi_buy_max
        self.trend_ema_period = trend_ema_period
        self.atr_period = atr_period
        self.min_atr_pct = min_atr_pct
        self.suggested_quote_amount = suggested_quote_amount

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        candles = list(market_state.candles)
        closes = market_state.get_closes()

        minimum = max(
            self.lookback,
            (self.rsi_period + 1) if self.rsi_period is not None else 0,
            self.trend_ema_period or 0,
            (self.atr_period + 1) if self.atr_period is not None else 0,
        )
        if len(candles) < minimum:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=market_state.symbol,
                reason=f"not_enough_candles: have={len(candles)}, need={minimum}",
            )

        window = [float(value) for value in closes[-self.lookback :]]
        mean = sum(window) / len(window)
        variance = sum((value - mean) ** 2 for value in window) / (len(window) - 1)
        std = sqrt(variance)

        current_close = Decimal(str(candles[-1].close))

        if std <= 0:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=market_state.symbol,
                reason=f"flat_window_no_volatility: mean={mean}",
            )

        z_score = (float(current_close) - mean) / std

        # Exit first: if price has reverted back to/above the exit band, close.
        if z_score >= float(self.exit_z):
            return TradeSignal(
                strategy_name=self.name,
                symbol=market_state.symbol,
                side=SignalSide.SELL,
                confidence=0.6,
                reason=f"mean_reversion_exit: z={z_score:.3f}, exit_z={self.exit_z}",
                created_at=utc_now(),
            )

        # Entry: price stretched at least entry_z std-devs below the mean.
        if z_score <= -float(self.entry_z):
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

            return TradeSignal(
                strategy_name=self.name,
                symbol=market_state.symbol,
                side=SignalSide.BUY,
                confidence=0.66,
                reason=(
                    f"mean_reversion_buy: z={z_score:.3f}, entry_z=-{self.entry_z}, "
                    f"close={current_close}, mean={mean:.2f}"
                ),
                created_at=utc_now(),
                suggested_quote_amount=self.suggested_quote_amount,
            )

        return TradeSignal.hold(
            strategy_name=self.name,
            symbol=market_state.symbol,
            reason=f"no_reversion_setup: z={z_score:.3f}",
        )

    def _buy_filter_block_reason(
        self,
        *,
        candles: list,
        closes: list[float],
        current_close: Decimal,
    ) -> str | None:
        if self.rsi_buy_max is not None and self.rsi_period is not None:
            current_rsi = rsi(closes, self.rsi_period)
            if current_rsi is None:
                return "buy_filtered_rsi_not_available"
            if current_rsi > self.rsi_buy_max:
                return f"buy_filtered_rsi_not_oversold: rsi={current_rsi:.2f}, max={self.rsi_buy_max}"

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
