from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.walk_forward_diagnostics import (
    aggregate_candidate_diagnostics,
    diagnose_train_fold,
    rank_diagnostics,
    train_rejection_reasons,
)
from app.backtesting.walk_forward_price import PriceCandidate, SimulationResult, WalkForwardFold
from app.market.models import Candle


def _result(*, pnl: str, trades: int, profit: str, loss: str) -> SimulationResult:
    # Directly construct a result without trades because only aggregate properties
    # are tested through a tiny synthetic subclass-like fixture below.
    class _TestResult(SimulationResult):
        pass

    # Use genuine simulated trades in diagnose_train_fold tests; this helper only
    # verifies rejection logic with a minimal object compatible with the protocol.
    raise RuntimeError("helper is intentionally unused")


def _candles(count: int) -> list[Candle]:
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    output: list[Candle] = []
    for index in range(count):
        open_time = origin + timedelta(minutes=index)
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


def test_diagnostics_mark_profitable_candidate_as_eligible() -> None:
    candles = _candles(80)
    candidate = PriceCandidate(name="candidate", family="breakout", horizon_bars=2, breakout_lookback=20)
    fold = WalkForwardFold(fold_number=1, train_start=0, train_end=60, validation_start=60, validation_end=80)
    diagnostics = diagnose_train_fold(
        candles,
        fold=fold,
        candidates=(candidate,),
        signal_index_map={candidate.name: (10, 20, 30, 40, 50)},
        initial_equity=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
        min_train_trades=5,
    )
    assert diagnostics[0].eligible
    assert diagnostics[0].rejection_reasons == ()


def test_diagnostics_expose_all_rejection_reasons() -> None:
    candles = _candles(30)
    candidate = PriceCandidate(name="candidate", family="breakout", horizon_bars=2, breakout_lookback=20)
    fold = WalkForwardFold(fold_number=1, train_start=0, train_end=25, validation_start=25, validation_end=30)
    diagnostics = diagnose_train_fold(
        candles,
        fold=fold,
        candidates=(candidate,),
        signal_index_map={candidate.name: (10,)},
        initial_equity=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("1"),
        slippage_pct=Decimal("0"),
        min_train_trades=5,
    )
    assert diagnostics[0].eligible is False
    assert "not_enough_train_trades" in diagnostics[0].rejection_reasons
    assert "non_positive_train_net_pnl_after_costs" in diagnostics[0].rejection_reasons
    assert "profit_factor_not_above_1" in diagnostics[0].rejection_reasons


def test_rank_places_eligible_candidate_first_and_aggregate_counts_it() -> None:
    candles = _candles(100)
    good = PriceCandidate(name="good", family="breakout", horizon_bars=2, breakout_lookback=20)
    bad = PriceCandidate(name="bad", family="breakout", horizon_bars=2, breakout_lookback=20)
    fold = WalkForwardFold(fold_number=1, train_start=0, train_end=70, validation_start=70, validation_end=100)
    diagnostics = diagnose_train_fold(
        candles,
        fold=fold,
        candidates=(bad, good),
        signal_index_map={bad.name: (10,), good.name: (10, 20, 30, 40, 50)},
        initial_equity=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
        min_train_trades=5,
    )
    ranked = rank_diagnostics(diagnostics)
    assert ranked[0].candidate.name == "good"
    aggregate = aggregate_candidate_diagnostics(ranked)
    assert aggregate[0]["candidate"]["name"] == "good"
    assert aggregate[0]["eligible_folds"] == 1
