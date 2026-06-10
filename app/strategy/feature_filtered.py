"""V22 feature-filtered strategy wrapper.

Wraps any base strategy and lets a BUY through only when configured market-
feature rules pass. Four rules are supported:

  * volume spike      : latest volume / recent average >= min_volume_spike_ratio
  * taker_buy_ratio   : >= min_taker_buy_ratio
  * order_book_imbalance : >= min_order_book_imbalance
  * spread            : <= max_spread_pct

The volume-spike rule is computed from candles, so it always works historically.
The other three depend on the market_features table. Their honesty contract:

  * If a feature filter is configured *optional* and its data is missing for the
    current candle, the filter is SKIPPED (it cannot block, and it does not
    pretend the data exists).
  * If configured *required* and the data is missing, the wrapper RAISES a clear
    FeatureUnavailableError instead of silently passing or blocking.

This is the whole point of V22: order-book features do not exist historically, so
a backtest must not quietly invent them. Requiring them on historical data is a
configuration error and is treated as one.
"""

from __future__ import annotations

from datetime import datetime

from app.market.features import MarketFeatures, volume_spike_ratios
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal


class FeatureUnavailableError(RuntimeError):
    """Raised when a REQUIRED feature filter has no data for the current candle."""


class FeatureFilteredStrategy(Strategy):
    def __init__(
        self,
        *,
        base_strategy: Strategy,
        volume_spike_lookback: int = 20,
        min_volume_spike_ratio: float | None = None,
        min_taker_buy_ratio: float | None = None,
        min_order_book_imbalance: float | None = None,
        max_spread_pct: float | None = None,
        require_taker: bool = False,
        require_order_book: bool = False,
        require_spread: bool = False,
        features_by_close_time: dict[datetime, MarketFeatures] | None = None,
        name: str | None = None,
    ) -> None:
        if volume_spike_lookback <= 0:
            raise ValueError("volume_spike_lookback must be positive")

        # A required feature filter with no data source can never be satisfied;
        # fail fast at construction instead of mid-backtest.
        required_with_threshold = (
            (require_taker and min_taker_buy_ratio is not None)
            or (require_order_book and min_order_book_imbalance is not None)
            or (require_spread and max_spread_pct is not None)
        )
        if required_with_threshold and features_by_close_time is None:
            raise ValueError(
                "a required feature filter is configured but no features_by_close_time "
                "was provided; required feature data must have a source"
            )

        self.base_strategy = base_strategy
        self.volume_spike_lookback = volume_spike_lookback
        self.min_volume_spike_ratio = min_volume_spike_ratio
        self.min_taker_buy_ratio = min_taker_buy_ratio
        self.min_order_book_imbalance = min_order_book_imbalance
        self.max_spread_pct = max_spread_pct
        self.require_taker = require_taker
        self.require_order_book = require_order_book
        self.require_spread = require_spread
        self.features_by_close_time = features_by_close_time or {}
        self.name = name or f"feature_filtered_{base_strategy.name}"

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        base_signal = self.base_strategy.on_market_state(market_state)

        # Only BUY signals are gated. Exits/holds pass through unchanged so the
        # wrapper can never trap an open position.
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

        block_reason = self._buy_block_reason(market_state)
        if block_reason is not None:
            return TradeSignal.hold(
                strategy_name=self.name,
                symbol=base_signal.symbol,
                reason=f"feature_filtered: {block_reason}; base={base_signal.reason}",
            )

        return TradeSignal(
            strategy_name=self.name,
            symbol=base_signal.symbol,
            side=SignalSide.BUY,
            confidence=min(0.95, base_signal.confidence + 0.02),
            reason=f"feature_filters_passed; base={base_signal.reason}",
            created_at=base_signal.created_at,
            suggested_quote_amount=base_signal.suggested_quote_amount,
            is_protective_exit=base_signal.is_protective_exit,
        )

    def _buy_block_reason(self, market_state: MarketState) -> str | None:
        candles = list(market_state.candles)
        if not candles:
            return "no_candles"
        latest = candles[-1]

        # Volume spike (price-derived, always available historically).
        if self.min_volume_spike_ratio is not None:
            ratios = volume_spike_ratios(candles, self.volume_spike_lookback)
            ratio = ratios[-1]
            if ratio is None:
                return f"volume_spike_insufficient_history: need>{self.volume_spike_lookback}"
            if ratio < self.min_volume_spike_ratio:
                return f"volume_spike_too_low: ratio={ratio:.3f}, min={self.min_volume_spike_ratio}"

        row = self.features_by_close_time.get(latest.close_time)

        taker_block = self._feature_min_block(
            value=row.taker_buy_ratio if row is not None else None,
            threshold=self.min_taker_buy_ratio,
            required=self.require_taker,
            name="taker_buy_ratio",
        )
        if taker_block is not None:
            return taker_block

        imbalance_block = self._feature_min_block(
            value=row.order_book_imbalance if row is not None else None,
            threshold=self.min_order_book_imbalance,
            required=self.require_order_book,
            name="order_book_imbalance",
        )
        if imbalance_block is not None:
            return imbalance_block

        spread_block = self._feature_max_block(
            value=row.spread_pct if row is not None else None,
            threshold=self.max_spread_pct,
            required=self.require_spread,
            name="spread_pct",
        )
        if spread_block is not None:
            return spread_block

        return None

    def _feature_min_block(
        self,
        *,
        value: float | None,
        threshold: float | None,
        required: bool,
        name: str,
    ) -> str | None:
        """Block when value < threshold. Handle missing data honestly."""
        if threshold is None:
            return None
        if value is None:
            if required:
                raise FeatureUnavailableError(
                    f"required feature {name!r} is unavailable for the current candle; "
                    "this feature has no historical data and cannot be required in a backtest"
                )
            return None  # optional filter, no data -> skip without pretending
        if value < threshold:
            return f"{name}_too_low: value={value:.5f}, min={threshold}"
        return None

    def _feature_max_block(
        self,
        *,
        value: float | None,
        threshold: float | None,
        required: bool,
        name: str,
    ) -> str | None:
        """Block when value > threshold. Handle missing data honestly."""
        if threshold is None:
            return None
        if value is None:
            if required:
                raise FeatureUnavailableError(
                    f"required feature {name!r} is unavailable for the current candle; "
                    "this feature has no historical data and cannot be required in a backtest"
                )
            return None
        if value > threshold:
            return f"{name}_too_high: value={value:.5f}, max={threshold}"
        return None
