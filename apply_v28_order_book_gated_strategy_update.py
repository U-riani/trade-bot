"""Apply the V28 order-book-gated price-strategy research update.

Run from the repository root after extracting this update archive there:
    python apply_v28_order_book_gated_strategy_update.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "v28_payload"
FILES = (
    Path("app/backtesting/order_book_gated_strategy.py"),
    Path("scripts/backtest_order_book_gated_strategies.py"),
    Path("tests/test_order_book_gated_strategy.py"),
)


def main() -> None:
    if not PAYLOAD.exists():
        raise SystemExit("V28 payload directory is missing. Extract the complete update archive into the repository root.")

    for relative_path in FILES:
        source = PAYLOAD / relative_path
        destination = ROOT / relative_path
        if not source.exists():
            raise SystemExit(f"Missing V28 payload file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"updated {relative_path.as_posix()}")

    shutil.rmtree(PAYLOAD)
    print("V28 order-book-gated strategy research update applied.")
    print("Run: python -m pytest -q")
    print(
        "Run: python -m scripts.backtest_order_book_gated_strategies "
        "--market-data-source production --timeframes 1m,5m,15m --limit 50000 "
        "--strategies ema_rsi_momentum,breakout_momentum,mean_reversion "
        "--order-book-features imbalance_top_20,imbalance_top_5 "
        "--entry-quantiles 0.6,0.7,0.8 --horizons 1,3,6 "
        "--min-feature-samples 100 --min-trades 5 "
        "--export-json reports/order_book_gated_strategy_v28.json "
        "--export-csv reports/order_book_gated_strategy_v28.csv"
    )


if __name__ == "__main__":
    main()
