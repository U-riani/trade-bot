"""Edge-quality analytics for backtest results.

The base :mod:`app.backtesting.metrics` answers "how much money did this make".
These analytics answer the harder question "is there an actual edge, or did the
backtest get lucky". They are pure functions over the trade list and the per-bar
equity curve, so they are trivial to unit test and never touch I/O.

Why these specific numbers:

- profit_factor / expectancy / payoff_ratio describe the *per trade* edge. A
  strategy can have a 30% win rate and still be excellent if the wins are large
  enough, or a 70% win rate and still bleed out if the losses are large enough.
  Win rate alone is the most over-read and least informative trading metric.
- Sharpe / Sortino describe risk-adjusted *path* quality from the equity curve.
- max_consecutive_losses is the number that decides whether a human can actually
  hold the strategy without turning it off at the worst possible moment.

Nothing here makes a strategy profitable. It only makes a fake edge easier to
catch before real money finds out the expensive way.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.metrics import BacktestTrade
from app.utils.timeframe import timeframe_to_seconds

_SECONDS_PER_YEAR = Decimal(365 * 24 * 60 * 60)


@dataclass(slots=True, frozen=True)
class TradeStatistics:
    """Per-trade edge statistics derived purely from completed round trips."""

    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: Decimal
    gross_profit: Decimal
    gross_loss: Decimal  # reported as a positive magnitude
    profit_factor: Decimal | None  # None => no losing trades (undefined / infinite)
    avg_win: Decimal
    avg_loss: Decimal  # positive magnitude
    payoff_ratio: Decimal | None  # avg_win / avg_loss; None when no losses
    expectancy: Decimal  # expected pnl per trade, in quote currency
    expectancy_r: Decimal | None  # expectancy expressed in average-loss (R) units
    avg_return_pct: Decimal
    largest_win: Decimal  # signed (>= 0)
    largest_loss: Decimal  # signed (<= 0)
    max_consecutive_wins: int
    max_consecutive_losses: int
    trade_return_sharpe: Decimal | None  # mean/stdev of per-trade return %


@dataclass(slots=True, frozen=True)
class EquityStatistics:
    """Risk-adjusted statistics derived from the per-bar equity curve."""

    bars: int
    sharpe: Decimal | None
    sortino: Decimal | None
    max_drawdown_pct: Decimal
    annualized: bool


def _mean(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


def _sample_stdev(values: list[Decimal], mean: Decimal) -> Decimal:
    if len(values) < 2:
        return Decimal("0")
    variance = sum(((value - mean) ** 2 for value in values), Decimal("0")) / Decimal(len(values) - 1)
    if variance <= 0:
        return Decimal("0")
    return variance.sqrt()


def _downside_stdev(values: list[Decimal], target: Decimal) -> Decimal:
    """Standard deviation of only the below-target returns (Sortino denominator)."""
    downside = [value - target for value in values if value < target]
    if not downside:
        return Decimal("0")
    # Sortino convention: divide by total observation count, not the downside count.
    variance = sum((diff**2 for diff in downside), Decimal("0")) / Decimal(len(values))
    if variance <= 0:
        return Decimal("0")
    return variance.sqrt()


def compute_trade_statistics(trades: list[BacktestTrade]) -> TradeStatistics:
    """Summarize the per-trade edge of a finished backtest.

    An empty trade list returns an all-zero record rather than raising, so the
    caller can rank a strategy that never traded against ones that did.
    """
    num_trades = len(trades)
    if num_trades == 0:
        zero = Decimal("0")
        return TradeStatistics(
            num_trades=0,
            num_wins=0,
            num_losses=0,
            win_rate=zero,
            gross_profit=zero,
            gross_loss=zero,
            profit_factor=None,
            avg_win=zero,
            avg_loss=zero,
            payoff_ratio=None,
            expectancy=zero,
            expectancy_r=None,
            avg_return_pct=zero,
            largest_win=zero,
            largest_loss=zero,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            trade_return_sharpe=None,
        )

    wins = [trade.pnl for trade in trades if trade.pnl > 0]
    losses = [trade.pnl for trade in trades if trade.pnl < 0]
    num_wins = len(wins)
    num_losses = len(losses)

    gross_profit = sum(wins, Decimal("0"))
    gross_loss = -sum(losses, Decimal("0"))  # positive magnitude

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    avg_win = gross_profit / Decimal(num_wins) if num_wins else Decimal("0")
    avg_loss = gross_loss / Decimal(num_losses) if num_losses else Decimal("0")
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else None

    win_rate = Decimal(num_wins) / Decimal(num_trades)
    loss_rate = Decimal(num_losses) / Decimal(num_trades)
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    expectancy_r = expectancy / avg_loss if avg_loss > 0 else None

    returns_pct = [trade.return_pct for trade in trades]
    avg_return_pct = _mean(returns_pct)
    return_stdev = _sample_stdev(returns_pct, avg_return_pct)
    trade_return_sharpe = avg_return_pct / return_stdev if return_stdev > 0 else None

    largest_win = max((trade.pnl for trade in trades), default=Decimal("0"))
    largest_loss = min((trade.pnl for trade in trades), default=Decimal("0"))
    if largest_win < 0:
        largest_win = Decimal("0")
    if largest_loss > 0:
        largest_loss = Decimal("0")

    max_consecutive_wins, max_consecutive_losses = _max_streaks(trades)

    return TradeStatistics(
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        expectancy_r=expectancy_r,
        avg_return_pct=avg_return_pct,
        largest_win=largest_win,
        largest_loss=largest_loss,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        trade_return_sharpe=trade_return_sharpe,
    )


def _max_streaks(trades: list[BacktestTrade]) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0
    for trade in trades:
        if trade.pnl > 0:
            cur_wins += 1
            cur_losses = 0
        elif trade.pnl < 0:
            cur_losses += 1
            cur_wins = 0
        else:  # break-even trade resets both streaks
            cur_wins = 0
            cur_losses = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def bars_per_year_for_timeframe(timeframe: str) -> Decimal:
    """How many candles of ``timeframe`` fit in a calendar year (annualization factor)."""
    seconds = Decimal(timeframe_to_seconds(timeframe))
    return _SECONDS_PER_YEAR / seconds


def compute_equity_statistics(
    equity_curve: list[Decimal],
    *,
    bars_per_year: Decimal | None = None,
) -> EquityStatistics:
    """Sharpe, Sortino, and percent max-drawdown from a per-bar equity curve.

    ``bars_per_year`` annualizes the Sharpe/Sortino (multiply by ``sqrt(N)``);
    pass ``None`` to get the raw per-bar ratio. Sparse strategies that sit in
    cash most of the time will show many zero-return bars, which deflates these
    ratios on purpose: idle capital is not risk-adjusted return.
    """
    bars = len(equity_curve)
    if bars < 2:
        return EquityStatistics(
            bars=bars,
            sharpe=None,
            sortino=None,
            max_drawdown_pct=Decimal("0"),
            annualized=bars_per_year is not None,
        )

    bar_returns: list[Decimal] = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        if previous > 0:
            bar_returns.append((current - previous) / previous)
        else:
            bar_returns.append(Decimal("0"))

    mean_return = _mean(bar_returns)
    stdev = _sample_stdev(bar_returns, mean_return)
    downside = _downside_stdev(bar_returns, Decimal("0"))

    scale = bars_per_year.sqrt() if bars_per_year is not None and bars_per_year > 0 else Decimal("1")
    sharpe = (mean_return / stdev) * scale if stdev > 0 else None
    sortino = (mean_return / downside) * scale if downside > 0 else None

    max_drawdown_pct = _max_drawdown_pct(equity_curve)

    return EquityStatistics(
        bars=bars,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_drawdown_pct,
        annualized=bars_per_year is not None,
    )


def _max_drawdown_pct(equity_curve: list[Decimal]) -> Decimal:
    peak = equity_curve[0]
    max_dd = Decimal("0")
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            drawdown = (peak - equity) / peak * Decimal("100")
            if drawdown > max_dd:
                max_dd = drawdown
    return max_dd
