"""Apply V30.1 order-book alignment audit files into an existing project root."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "v30_1_update"
TARGETS = (
    "app/backtesting/order_book_alignment.py",
    "scripts/audit_order_book_alignment.py",
    "tests/test_order_book_alignment.py",
)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit("v30_1_update directory is missing. Extract the whole archive first.")
    for relative in TARGETS:
        source = SOURCE / relative
        target = ROOT / relative
        if not source.exists():
            raise SystemExit(f"missing package file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"updated {relative}")
    print("V30.1 order-book alignment audit applied.")
    print("Run: python -m pytest -q")
    print("Run: python -m scripts.audit_order_book_alignment --market-data-source production --limit 50000 --order-book-features imbalance_top_20,imbalance_top_5 --max-age-seconds 30,60,120,180,300 --max-pair-gap-seconds 60,120,180,300 --min-feature-samples 100 --export-json reports/order_book_alignment_v30_1.json --export-csv reports/order_book_alignment_v30_1.csv")


if __name__ == "__main__":
    main()
