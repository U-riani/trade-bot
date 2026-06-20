from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
SCRIPT = ROOT / "scripts" / "backtest_multitimeframe_pullback_strategy.py"
TEST = ROOT / "tests" / "test_multitimeframe_pullback_strategy.py"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if not SCRIPT.exists() or not TEST.exists():
        raise SystemExit("Run this script from the trade-bot repository root.")

    script = SCRIPT.read_text(encoding="utf-8")
    test = TEST.read_text(encoding="utf-8")

    script = replace_once(
        script,
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, replace\n",
        label="dataclasses import",
    )
    script = replace_once(
        script,
        "from app.backtesting.order_book_strategy import quantile_threshold, rows_with_feature\n",
        "from app.backtesting.order_book_strategy import quantile_threshold, rows_with_feature\nfrom app.backtesting.resample import resample_candles\n",
        label="resample import",
    )
    script = replace_once(
        script,
        "from app.market.features import MarketFeatures\n",
        "from app.market.features import MarketFeatures\nfrom app.market.models import Candle\n",
        label="candle import",
    )

    loader_pattern = re.compile(
        r"async def _load_rows\(timeframe: str, args: argparse\.Namespace\) -> tuple\[list\[MarketFeatures\], str\]:\n.*?\n\ndef _coverage_split",
        re.DOTALL,
    )
    replacement = '''def _price_feature_row(candle: Candle, observation: MarketFeatures | None = None) -> MarketFeatures:
    """Represent a complete candle while preserving any observed order-book fields.

    Price chronology comes from the candles table.  Live depth observations are
    joined by close time only, so absent depth does not erase a real candle.
    """

    if observation is not None:
        return replace(
            observation,
            timeframe=candle.timeframe,
            open_time=candle.open_time,
            close_time=candle.close_time,
            close_price=candle.close,
            volume=candle.volume,
        )
    return MarketFeatures(
        exchange=candle.exchange,
        symbol=candle.symbol,
        timeframe=candle.timeframe,
        open_time=candle.open_time,
        close_time=candle.close_time,
        close_price=candle.close,
        volume=candle.volume,
    )


def _merge_candles_with_observations(
    candles: list[Candle], observations: list[MarketFeatures]
) -> list[MarketFeatures]:
    """Return one row for every price candle, enriched only where depth was observed."""

    observed_by_close = {row.close_time: row for row in observations}
    return [_price_feature_row(candle, observed_by_close.get(candle.close_time)) for candle in candles]


async def _load_price_timelines(
    args: argparse.Namespace,
) -> tuple[list[MarketFeatures], list[MarketFeatures], list[MarketFeatures], str]:
    """Load complete 1m prices, then derive complete 5m/15m price timelines.

    V29 originally used sparse ``market_features`` rows for higher-timeframe
    indicator history.  Those rows exist only when aggregation produced a feature,
    so missing depth snapshots were incorrectly treated as missing candles.  The
    strategy now uses complete candle data for price logic and joins order-book
    observations solely onto the 1m entry timeline.
    """

    settings = get_settings()
    _source, _use_testnet, exchange = _resolve_market_data_source(args.market_data_source)
    db = Database(settings.database_url)
    await db.connect()
    try:
        repository = TradingRepository(db)
        candles_1m = await repository.load_recent_candles(
            exchange=exchange,
            symbol=settings.normalized_symbol,
            timeframe="1m",
            limit=args.limit,
        )
        observations_1m = await repository.load_market_features(
            exchange=exchange,
            symbol=settings.normalized_symbol,
            timeframe="1m",
            limit=args.limit,
        )
    finally:
        await db.close()

    if not candles_1m:
        raise SystemExit("No 1m candles available for V29.1")

    entry_rows = _merge_candles_with_observations(candles_1m, observations_1m)
    pullback_rows = [_price_feature_row(candle) for candle in resample_candles(candles_1m, target_timeframe="5m")]
    trend_rows = [_price_feature_row(candle) for candle in resample_candles(candles_1m, target_timeframe="15m")]

    logger.info(
        "v29_price_timelines_loaded",
        symbol=settings.normalized_symbol,
        candles_1m=len(candles_1m),
        observed_order_book_rows=len(observations_1m),
        entry_rows=len(entry_rows),
        pullback_rows=len(pullback_rows),
        trend_rows=len(trend_rows),
        note="price timelines come from complete candles; order-book values are joined only at 1m closes",
    )
    return entry_rows, pullback_rows, trend_rows, settings.normalized_symbol


def _coverage_split'''
    script, count = loader_pattern.subn(replacement, script, count=1)
    if count != 1:
        raise RuntimeError("Could not replace V29 feature-row loader")

    script = replace_once(
        script,
        '''    base_rows, symbol = await _load_rows("1m", args)
    pullback_rows, _ = await _load_rows("5m", args)
    trend_rows, _ = await _load_rows("15m", args)
''',
        '''    base_rows, pullback_rows, trend_rows, symbol = await _load_price_timelines(args)
''',
        label="V29 timeline load call",
    )

    if "test_v29_1_merge_preserves_complete_price_timeline" not in test:
        test = test.replace(
            "from decimal import Decimal\n",
            "from decimal import Decimal\n\nfrom app.market.models import Candle\n",
            1,
        )
        test = test.replace(
            "    run_multitimeframe_pullback_backtest,\n",
            "    run_multitimeframe_pullback_backtest,\n",
            1,
        )
        test += '''\n\ndef test_v29_1_merge_preserves_complete_price_timeline() -> None:\n    from scripts.backtest_multitimeframe_pullback_strategy import _merge_candles_with_observations\n\n    start = datetime(2026, 1, 1, tzinfo=UTC)\n    candles = [\n        Candle(\n            exchange="binance_spot", symbol="BTCUSDT", timeframe="1m",\n            open_time=start + timedelta(minutes=index),\n            close_time=start + timedelta(minutes=index + 1) - timedelta(milliseconds=1),\n            open=100 + index, high=101 + index, low=99 + index, close=100 + index, volume=1.0, is_closed=True,\n        )\n        for index in range(3)\n    ]\n    observation = _row(1, close=101, imbalance=0.42)\n    rows = _merge_candles_with_observations(candles, [observation])\n\n    assert len(rows) == 3\n    assert [row.close_price for row in rows] == [100, 101, 102]\n    assert rows[0].imbalance_top_20 is None\n    assert rows[1].imbalance_top_20 == 0.42\n    assert rows[2].imbalance_top_20 is None\n'''

    SCRIPT.write_text(script, encoding="utf-8")
    TEST.write_text(test, encoding="utf-8")
    print("updated scripts/backtest_multitimeframe_pullback_strategy.py")
    print("updated tests/test_multitimeframe_pullback_strategy.py")
    print("V29.1 candle-price-timeline correction applied.")
    print("Run: python -m pytest -q")
    print("Run: python -m scripts.backtest_multitimeframe_pullback_strategy --market-data-source production --limit 50000 --order-book-features imbalance_top_20,imbalance_top_5 --entry-quantiles 0.6,0.7,0.8 --horizons 5,10,15 --min-feature-samples 100 --min-trades 10 --export-json reports/multitimeframe_pullback_strategy_v29_1.json --export-csv reports/multitimeframe_pullback_strategy_v29_1.csv")


if __name__ == "__main__":
    main()
