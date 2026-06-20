V27.3 coverage-aware split update

Fixes V27.2's incorrect 70/30 split across the entire historical candle table.
The new split is calculated inside the actual order-book feature coverage window,
while the backtest retains every candle inside that window and still skips any
trade path crossing a timestamp gap.

Apply from the repository root:
  python apply_v27_3_coverage_aware_split_update.py
