"""Per-regime attribution of backtest trades.

The aggregate backtest hides a question we actually care about: *when* does a
strategy make or lose money? A mean-reversion strategy is supposed to work in
ranging (sideways) markets and bleed in strong trends; a breakout strategy is
the opposite. If we only look at the blended result, a real conditional edge
gets averaged into mush against the regime where the strategy never had a
chance.

This module labels every candle with a market regime (reusing the V20
``classify_market_regime`` detector) and then buckets each completed trade by
the regime that was active at its entry. Per-bucket we reuse
``compute_trade_statistics`` so expectancy / profit factor / win rate are
directly comparable across regimes.

Important honesty note: a per-regime edge is still only a *lead*, not a
strategy. It only becomes tradeable if a regime detector can identify the good
regime in real time without peeking at the future. The detector used here is
causal (it only ever looks at candles up to and including the current one), so
a positive bucket here is at least not cheating on time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.backtesting.analytics import TradeStatistics, compute_trade_statistics
from app.backtesting.metrics import BacktestTrade
from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.market_regime import MarketRegime, classify_market_regime


@dataclass(slots=True, frozen=True)
class RegimeBucket:
    regime: MarketRegime
    candle_count: int  # how many candles sat in this regime (exposure)
    stats: TradeStatistics
    total_pnl: Decimal


def label_regimes(
    candles: list[Candle],
    *,
    max_candles: int = 500,
    fast_ema_period: int = 50,
    slow_ema_period: int = 200,
    slope_lookback: int = 20,
    min_slope_pct: Decimal = Decimal("0.03"),
    min_ema_gap_pct: Decimal = Decimal("0.05"),
) -> dict[datetime, MarketRegime]:
    """Causally label each candle's regime, keyed by close_time.

    The candle is walked through a ``MarketState`` exactly the way the backtest
    engine does (same ``max_candles`` window, same duplicate/out-of-order
    rejection) so the regime seen at a trade's entry matches what the strategy
    would have seen live. The classifier never looks past the current candle.
    """
    if not candles:
        return {}

    sorted_candles = sorted(candles, key=lambda item: item.open_time)
    state = MarketState(symbol=sorted_candles[0].symbol, max_candles=max_candles)
    labels: dict[datetime, MarketRegime] = {}

    for candle in sorted_candles:
        if not state.add_candle(candle):
            continue
        snapshot = classify_market_regime(
            state,
            fast_ema_period=fast_ema_period,
            slow_ema_period=slow_ema_period,
            slope_lookback=slope_lookback,
            min_slope_pct=min_slope_pct,
            min_ema_gap_pct=min_ema_gap_pct,
        )
        labels[candle.close_time] = snapshot.regime

    return labels


def regime_candle_counts(labels: dict[datetime, MarketRegime]) -> dict[MarketRegime, int]:
    """How many candles fell in each regime (the strategy's exposure budget)."""
    counts: dict[MarketRegime, int] = {regime: 0 for regime in MarketRegime}
    for regime in labels.values():
        counts[regime] += 1
    return counts


def attribute_trades_by_regime(
    trades: list[BacktestTrade],
    labels: dict[datetime, MarketRegime],
) -> dict[MarketRegime, list[BacktestTrade]]:
    """Group trades by the regime active at their entry candle.

    A trade whose entry_time has no label (e.g. it entered before the regime
    detector had enough history) is bucketed under ``MarketRegime.UNKNOWN``.
    """
    buckets: dict[MarketRegime, list[BacktestTrade]] = {regime: [] for regime in MarketRegime}
    for trade in trades:
        regime = labels.get(trade.entry_time, MarketRegime.UNKNOWN)
        buckets[regime].append(trade)
    return buckets


def build_regime_buckets(
    trades: list[BacktestTrade],
    labels: dict[datetime, MarketRegime],
) -> list[RegimeBucket]:
    """Full per-regime breakdown: exposure, trade stats, and total pnl."""
    counts = regime_candle_counts(labels)
    grouped = attribute_trades_by_regime(trades, labels)

    buckets: list[RegimeBucket] = []
    for regime in MarketRegime:
        regime_trades = grouped[regime]
        buckets.append(
            RegimeBucket(
                regime=regime,
                candle_count=counts.get(regime, 0),
                stats=compute_trade_statistics(regime_trades),
                total_pnl=sum((trade.pnl for trade in regime_trades), Decimal("0")),
            )
        )
    return buckets
