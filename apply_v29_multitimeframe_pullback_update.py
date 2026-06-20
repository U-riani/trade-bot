from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path.cwd()
PAYLOAD = Path(__file__).resolve().parent / "v29_payload"
FILES = (
    "app/backtesting/multitimeframe_pullback_strategy.py",
    "scripts/backtest_multitimeframe_pullback_strategy.py",
    "tests/test_multitimeframe_pullback_strategy.py",
)


def main() -> None:
    if not (ROOT / "app").is_dir() or not (ROOT / "scripts").is_dir() or not (ROOT / "tests").is_dir():
        raise SystemExit("Run this script from the trade-bot repository root.")
    if not PAYLOAD.is_dir():
        raise SystemExit(f"Missing V29 payload directory: {PAYLOAD}")

    for relative in FILES:
        source = PAYLOAD / relative
        target = ROOT / relative
        if not source.is_file():
            raise SystemExit(f"Missing payload file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"updated {relative}")

    print("V29 multi-timeframe pullback/order-book-reversal research update applied.")
    print("Run: python -m pytest -q")
    print(
        "Run: python -m scripts.backtest_multitimeframe_pullback_strategy "
        "--market-data-source production --limit 50000 "
        "--order-book-features imbalance_top_20,imbalance_top_5 "
        "--entry-quantiles 0.6,0.7,0.8 --horizons 5,10,15 "
        "--min-feature-samples 100 --min-trades 10 "
        "--export-json reports/multitimeframe_pullback_strategy_v29.json "
        "--export-csv reports/multitimeframe_pullback_strategy_v29.csv"
    )


if __name__ == "__main__":
    main()
