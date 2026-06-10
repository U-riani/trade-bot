from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.market.indicators import ema
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal


class MarketRegime(StrEnum):
    UNKNOWN = "unknown"
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass(slots=True, frozen=True)
class MarketRegimeSnapshot:
    regime: MarketRegime
    reason: str
    close: Decimal | None = None
    fast_ema: Decimal | None = None
    slow_ema: Decimal | None = None
    slow_ema_past: Decimal | None = None
    ema_gap_pct: Decimal | None = None
    slow_slope_pct: Decimal | None = None


def classify_market_regime(
    market_state: MarketState,
    *,
    fast_ema_period: int = 50,
    slow_ema_period: int = 200,
    slope_lookback: int = 20,
    min_slope_pct: Decimal = Decimal("0.03"),
    min_ema_gap_pct: Decimal = Decimal("0.05"),
) -> MarketRegimeSnapshot:
    """Classify the current market as bullish, bearish, sideways, or unknown.

    V20 uses this as a boring but useful guardrail: in spot long-only mode, the
    bot should not keep trying to buy when the larger context is bearish. This
    will not magically print money, because sadly the universe has terms and
    conditions, but it should reduce obviously dumb long entries.
    """
    if fast_ema_period <= 0:
        raise ValueError("fast_ema_period must be positive")
    if slow_ema_period <= 0:
        raise ValueError("slow_ema_period must be positive")
    if fast_ema_period >= slow_ema_period:
        raise ValueError("fast_ema_period must be smaller than slow_ema_period")
    if slope_lookback <= 0:
        raise ValueError("slope_lookback must be positive")
    if min_slope_pct < 0:
        raise ValueError("min_slope_pct cannot be negative")
    if min_ema_gap_pct < 0:
        raise ValueError("min_ema_gap_pct cannot be negative")

    closes = market_state.get_closes()
    required = slow_ema_period + slope_lookback + 1
    if len(closes) < required:
        return MarketRegimeSnapshot(
            regime=MarketRegime.UNKNOWN,
            reason=f"not_enough_candles_for_regime: have={len(closes)}, need={required}",
        )

    fast_values = ema(closes, fast_ema_period)
    slow_values = ema(closes, slow_ema_period)
    close = Decimal(str(closes[-1]))
    fast = Decimal(str(fast_values[-1]))
    slow = Decimal(str(slow_values[-1]))
    slow_past = Decimal(str(slow_values[-(slope_lookback + 1)]))

    if slow == 0 or slow_past == 0:
        return MarketRegimeSnapshot(regime=MarketRegime.UNKNOWN, reason="zero_ema_value")

    ema_gap_pct = ((fast - slow) / slow) * Decimal("100")
    slow_slope_pct = ((slow - slow_past) / slow_past) * Decimal("100")

    common = {
        "close": close,
        "fast_ema": fast,
        "slow_ema": slow,
        "slow_ema_past": slow_past,
        "ema_gap_pct": ema_gap_pct,
        "slow_slope_pct": slow_slope_pct,
    }

    if close > slow and fast > slow and ema_gap_pct >= min_ema_gap_pct and slow_slope_pct >= min_slope_pct:
        return MarketRegimeSnapshot(
            regime=MarketRegime.BULLISH,
            reason=(
                f"bullish_regime: close>{slow_ema_period}ema, "
                f"ema_gap_pct={ema_gap_pct:.4f}, slow_slope_pct={slow_slope_pct:.4f}"
            ),
            **common,
        )

    if close < slow and fast < slow and ema_gap_pct <= -min_ema_gap_pct and slow_slope_pct <= -min_slope_pct:
        return MarketRegimeSnapshot(
            regime=MarketRegime.BEARISH,
            reason=(
                f"bearish_regime: close<{slow_ema_period}ema, "
                f"ema_gap_pct={ema_gap_pct:.4f}, slow_slope_pct={slow_slope_pct:.4f}"
            ),
            **common,
        )

    return MarketRegimeSnapshot(
        regime=MarketRegime.SIDEWAYS,
        reason=(
            f"sideways_regime: ema_gap_pct={ema_gap_pct:.4f}, "
            f"slow_slope_pct={slow_slope_pct:.4f}"
        ),
        **common,
    )


class MarketRegimeFilteredStrategy(Strategy):
    """Wrap another strategy and allow BUY only in bullish market regimes."""

    def __init__(
        self,
        *,
        base_strategy: Strategy,
        fast_ema_period: int = 50,
        slow_ema_period: int = 200,
        slope_lookback: int = 20,
        min_slope_pct: Decimal = Decimal("0.03"),
        min_ema_gap_pct: Decimal = Decimal("0.05"),
        name: str | None = None,
    ) -> None:
        self.base_strategy = base_strategy
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.slope_lookback = slope_lookback
        self.min_slope_pct = min_slope_pct
        self.min_ema_gap_pct = min_ema_gap_pct
        self.name = name or f"regime_filtered_{base_strategy.name}"

        # Validate eagerly instead of waiting for a backtest to explode halfway
        # through, because apparently computers require explicit instructions not
        # to waste everyone's evening.
        classify_market_regime(
            MarketState(symbol="VALIDATION_DUMMY", max_candles=1),
            fast_ema_period=fast_ema_period,
            slow_ema_period=slow_ema_period,
            slope_lookback=slope_lookback,
            min_slope_pct=min_slope_pct,
            min_ema_gap_pct=min_ema_gap_pct,
        )

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        base_signal = self.base_strategy.on_market_state(market_state)
        if base_signal.side != SignalSide.BUY:
            if base_signal.side == SignalSide.HOLD:
                return TradeSignal.hold(
                    strategy_name=self.name,
                    symbol=base_signal.symbol,
                    reason=f"base_hold: {base_signal.reason}",
                )
            return TradeSignal(
                strategy_name=self.name,
                symbol=base_signal.symbol,
                side=base_signal.side,
                confidence=base_signal.confidence,
                reason=f"base_exit_passthrough: {base_signal.reason}",
                created_at=base_signal.created_at,
                suggested_quote_amount=base_signal.suggested_quote_amount,
                is_protective_exit=base_signal.is_protective_exit,
            )

        snapshot = classify_market_regime(
            market_state,
            fast_ema_period=self.fast_ema_period,
            slow_ema_period=self.slow_ema_period,
            slope_lookback=self.slope_lookback,
            min_slope_pct=self.min_slope_pct,
            min_ema_gap_pct=self.min_ema_gap_pct,
        )
        if snapshot.regime != MarketRegime.BULLISH:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=base_signal.symbol,
                reason=f"buy_blocked_by_{snapshot.regime}: {snapshot.reason}; base={base_signal.reason}",
            )

        return TradeSignal(
            strategy_name=self.name,
            symbol=base_signal.symbol,
            side=SignalSide.BUY,
            confidence=min(0.95, base_signal.confidence + 0.05),
            reason=f"regime_confirmed_bullish: {snapshot.reason}; base={base_signal.reason}",
            created_at=base_signal.created_at,
            suggested_quote_amount=base_signal.suggested_quote_amount,
            is_protective_exit=base_signal.is_protective_exit,
        )
