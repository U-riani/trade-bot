"""Apply the V30 order-book-improvement research update to a trade-bot checkout."""

from __future__ import annotations

import shutil
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent / "v30_update"
FILES = (
    "app/backtesting/multitimeframe_pullback_delta_strategy.py",
    "scripts/backtest_multitimeframe_pullback_delta_strategy.py",
    "tests/test_multitimeframe_pullback_delta_strategy.py",
)


def main() -> None:
    project_root = Path.cwd()
    if not (project_root / "app").is_dir() or not (project_root / "scripts").is_dir():
        raise SystemExit("Run this script from the trade-bot project root.")
    if not PACKAGE_DIR.is_dir():
        raise SystemExit(f"Update package directory is missing: {PACKAGE_DIR}")

    for relative in FILES:
        source = PACKAGE_DIR / relative
        target = project_root / relative
        if not source.is_file():
            raise SystemExit(f"Update file is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"updated {relative}")

    print("V30 order-book-delta strategy update applied.")
    print("Run: python -m pytest -q")
    print(
        "Run: python -m scripts.backtest_multitimeframe_pullback_delta_strategy "
        "--market-data-source production --limit 50000 "
        "--order-book-features imbalance_top_20,imbalance_top_5 "
        "--delta-quantiles 0.5,0.6,0.7 --horizons 5,10,15 "
        "--min-current-imbalance 0.0 --min-feature-samples 100 --min-trades 10 "
        "--export-json reports/multitimeframe_pullback_delta_strategy_v30.json "
        "--export-csv reports/multitimeframe_pullback_delta_strategy_v30.csv"
    )


if __name__ == "__main__":
    main()
