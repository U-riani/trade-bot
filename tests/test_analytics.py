from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from app.backtesting.analytics import (
    bars_per_year_for_timeframe,
    compute_equity_statistics,
    compute_trade_statistics,
)
from app.backtesting.metrics import BacktestTrade


def _trade(pnl: str, *, quote_amount: str = "100", entry_fee: str = "0") -> BacktestTrade:
    """Build a minimal trade; only pnl / quote_amount / entry_fee drive the stats."""
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return BacktestTrade(
        symbol="BTCUSDT",
        entry_time=ts,
        exit_time=ts,
        entry_price=Decimal("100"),
        exit_price=Decimal("100"),
        quantity=Decimal("1"),
        quote_amount=Decimal(quote_amount),
        entry_fee=Decimal(entry_fee),
        exit_fee=Decimal("0"),
        pnl=Decimal(pnl),
        entry_reason="test",
        exit_reason="test",
    )


def test_empty_trades_return_zeroed_stats() -> None:
    stats = compute_trade_statistics([])
    assert stats.num_trades == 0
    assert stats.expectancy == Decimal("0")
    assert stats.profit_factor is None
    assert stats.payoff_ratio is None
    assert stats.expectancy_r is None
    assert stats.trade_return_sharpe is None
    assert stats.max_consecutive_losses == 0


def test_trade_statistics_known_set() -> None:
    # Two +10 wins and one -5 loss, all on 100 quote with no fees.
    trades = [_trade("10"), _trade("10"), _trade("-5")]
    stats = compute_trade_statistics(trades)

    assert stats.num_trades == 3
    assert stats.num_wins == 2
    assert stats.num_losses == 1
    assert stats.gross_profit == Decimal("20")
    assert stats.gross_loss == Decimal("5")
    assert stats.profit_factor == Decimal("4")
    assert stats.avg_win == Decimal("10")
    assert stats.avg_loss == Decimal("5")
    assert stats.payoff_ratio == Decimal("2")
    # expectancy = (2/3 * 10) - (1/3 * 5) = 5 (Decimal rounding leaves a tiny tail)
    assert abs(stats.expectancy - Decimal("5")) < Decimal("0.0001")
    assert stats.expectancy_r is not None
    assert abs(stats.expectancy_r - Decimal("1")) < Decimal("0.0001")
    assert stats.largest_win == Decimal("10")
    assert stats.largest_loss == Decimal("-5")
    assert stats.max_consecutive_wins == 2
    assert stats.max_consecutive_losses == 1


def test_profit_factor_none_when_no_losses() -> None:
    stats = compute_trade_statistics([_trade("3"), _trade("4")])
    assert stats.profit_factor is None  # undefined / infinite with zero losses
    assert stats.payoff_ratio is None
    assert stats.expectancy > 0


def test_consecutive_losses_streak() -> None:
    trades = [_trade("5"), _trade("-1"), _trade("-1"), _trade("-1"), _trade("2")]
    stats = compute_trade_statistics(trades)
    assert stats.max_consecutive_losses == 3
    assert stats.max_consecutive_wins == 1


def test_equity_statistics_drawdown_and_sharpe() -> None:
    curve = [Decimal("100"), Decimal("110"), Decimal("121"), Decimal("108.9")]
    stats = compute_equity_statistics(curve)

    assert stats.bars == 4
    # Peak 121 down to 108.9 => 10% drawdown.
    assert abs(stats.max_drawdown_pct - Decimal("10")) < Decimal("0.0001")
    assert stats.sharpe is not None
    assert stats.sharpe > 0
    assert stats.sortino is not None
    assert stats.annualized is False


def test_equity_statistics_flat_curve_has_no_sharpe() -> None:
    stats = compute_equity_statistics([Decimal("100"), Decimal("100"), Decimal("100")])
    assert stats.sharpe is None  # zero variance => undefined
    assert stats.max_drawdown_pct == Decimal("0")


def test_equity_statistics_short_curve() -> None:
    stats = compute_equity_statistics([Decimal("100")])
    assert stats.bars == 1
    assert stats.sharpe is None
    assert stats.max_drawdown_pct == Decimal("0")


def test_annualization_scales_sharpe() -> None:
    curve = [Decimal("100"), Decimal("110"), Decimal("121"), Decimal("108.9")]
    raw = compute_equity_statistics(curve)
    annualized = compute_equity_statistics(curve, bars_per_year=bars_per_year_for_timeframe("5m"))
    assert annualized.annualized is True
    assert raw.sharpe is not None and annualized.sharpe is not None
    # Annualized Sharpe must be the raw ratio scaled up by sqrt(bars_per_year) > 1.
    assert annualized.sharpe > raw.sharpe


def test_bars_per_year_for_timeframe() -> None:
    # A year has 365*24*60 = 525600 one-minute candles.
    assert bars_per_year_for_timeframe("1m") == Decimal("525600")
    assert bars_per_year_for_timeframe("5m") == Decimal("105120")
    assert bars_per_year_for_timeframe("1h") == Decimal("8760")
