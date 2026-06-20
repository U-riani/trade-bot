from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.walk_forward_price import (
    PriceCandidate,
    build_walk_forward_folds,
    simulate_fixed_horizon,
)
from app.market.models import Candle


def _candles(count: int, *, gap_at: int | None = None) -> list[Candle]:
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    output: list[Candle] = []
    for index in range(count):
        extra = 5 if gap_at is not None and index >= gap_at else 0
        open_time = origin + timedelta(minutes=index + extra)
        price = 100.0 + index
        output.append(
            Candle(
                exchange="binance_spot",
                symbol="BTCUSDT",
                timeframe="1m",
                open_time=open_time,
                close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
                volume=1.0,
            )
        )
    return output


def _candidate(horizon: int = 1) -> PriceCandidate:
    return PriceCandidate(name="test", family="breakout", horizon_bars=horizon, breakout_lookback=20)


def test_signal_enters_next_bar_open_and_exits_fixed_horizon_close() -> None:
    candles = _candles(6)
    result = simulate_fixed_horizon(
        candles,
        candidate=_candidate(horizon=2),
        signal_indexes=(1,),
        start_index=0,
        end_index=len(candles),
        initial_equity=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )
    assert result.round_trips == 1
    trade = result.trades[0]
    assert trade.entry_time == candles[2].open_time
    assert trade.exit_time == candles[3].close_time
    assert trade.entry_price == Decimal(str(candles[2].open))
    assert trade.exit_price == Decimal(str(candles[3].close))


def test_gap_crossing_trade_is_rejected() -> None:
    candles = _candles(6, gap_at=3)
    result = simulate_fixed_horizon(
        candles,
        candidate=_candidate(horizon=3),
        signal_indexes=(1,),
        start_index=0,
        end_index=len(candles),
        initial_equity=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )
    assert result.round_trips == 0
    assert result.skipped_gap_signals == 1


def test_overlapping_signals_do_not_create_stacked_positions() -> None:
    candles = _candles(10)
    result = simulate_fixed_horizon(
        candles,
        candidate=_candidate(horizon=3),
        signal_indexes=(1, 2, 3),
        start_index=0,
        end_index=len(candles),
        initial_equity=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )
    assert result.round_trips == 1
    assert result.skipped_overlap_signals == 2


def test_walk_forward_folds_keep_train_before_validation() -> None:
    folds = build_walk_forward_folds(100, train_bars=40, validation_bars=20, step_bars=20)
    assert len(folds) == 3
    assert [(fold.train_start, fold.train_end, fold.validation_start, fold.validation_end) for fold in folds] == [
        (0, 40, 40, 60),
        (20, 60, 60, 80),
        (40, 80, 80, 100),
    ]
    assert all(fold.train_end == fold.validation_start for fold in folds)
