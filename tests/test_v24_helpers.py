from __future__ import annotations

from app.backtesting.feature_analysis import (
    REASON_NO_COLLECTED_ORDER_BOOK,
    REASON_NO_FEATURE_ROWS,
    REASON_NO_FORWARD_RETURNS,
    REASON_NOT_ENOUGH_SAMPLES,
    classify_feature_data_reason,
)
from scripts.aggregate_order_book_features import resolve_symbol
from scripts.run_order_book_research_cycle import build_cycle_commands


# --- analyzer no-data reason classification ---

def test_classify_no_collected_order_book() -> None:
    reason = classify_feature_data_reason(
        feature_name="order_book_imbalance", present_count=0, sample_size=0, min_samples=100
    )
    assert reason == REASON_NO_COLLECTED_ORDER_BOOK


def test_classify_no_feature_rows_for_non_order_book() -> None:
    reason = classify_feature_data_reason(
        feature_name="taker_buy_ratio", present_count=0, sample_size=0, min_samples=100
    )
    assert reason == REASON_NO_FEATURE_ROWS


def test_classify_no_forward_returns() -> None:
    reason = classify_feature_data_reason(
        feature_name="imbalance_top_5", present_count=3, sample_size=0, min_samples=100
    )
    assert reason == REASON_NO_FORWARD_RETURNS


def test_classify_not_enough_samples() -> None:
    reason = classify_feature_data_reason(
        feature_name="spread_pct", present_count=50, sample_size=50, min_samples=100
    )
    assert reason == REASON_NOT_ENOUGH_SAMPLES


def test_classify_enough_samples_returns_none() -> None:
    reason = classify_feature_data_reason(
        feature_name="volume_spike_ratio", present_count=500, sample_size=500, min_samples=100
    )
    assert reason is None


# --- aggregation symbol override ---

def test_resolve_symbol_override_wins() -> None:
    assert resolve_symbol("ethusdt", "BTCUSDT") == "ETHUSDT"


def test_resolve_symbol_falls_back_to_default() -> None:
    assert resolve_symbol(None, "BTCUSDT") == "BTCUSDT"
    assert resolve_symbol("  ", "BTCUSDT") == "BTCUSDT"  # whitespace-only -> default


# --- research cycle command generation (dry-run) ---

def test_cycle_commands_without_analyze() -> None:
    commands = build_cycle_commands(
        market_data_source="production", symbol="BTCUSDT", timeframes="1m,5m,15m",
        backfill_limit=200, candle_limit=5000, min_feature_samples=100,
        analyze=False, analyze_limit=50000, python_exe="py",
    )
    labels = [label for label, _ in commands]
    assert labels == ["backfill_candles", "aggregate_order_book_features", "order_book_pipeline_status"]
    # symbol is threaded into the aggregate + status commands
    aggregate_argv = commands[1][1]
    assert "--symbol" in aggregate_argv
    assert "BTCUSDT" in aggregate_argv
    assert aggregate_argv[0] == "py"


def test_cycle_commands_with_analyze() -> None:
    commands = build_cycle_commands(
        market_data_source="production", symbol="ETHUSDT", timeframes="5m",
        backfill_limit=200, candle_limit=5000, min_feature_samples=250,
        analyze=True, analyze_limit=12345, python_exe="py",
    )
    labels = [label for label, _ in commands]
    assert labels[-1] == "analyze_market_features"
    analyze_argv = commands[-1][1]
    assert "--limit" in analyze_argv
    assert "12345" in analyze_argv
    assert "--min-feature-samples" in analyze_argv
    assert "250" in analyze_argv


def test_cycle_commands_backfill_first() -> None:
    commands = build_cycle_commands(
        market_data_source="production", symbol="BTCUSDT", timeframes="1m",
        backfill_limit=60, candle_limit=1000, min_feature_samples=100,
        analyze=False, analyze_limit=50000,
    )
    # Order matters: candles must be backfilled before aggregation can match them.
    assert commands[0][0] == "backfill_candles"
    assert commands[1][0] == "aggregate_order_book_features"
