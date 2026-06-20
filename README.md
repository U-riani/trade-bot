# V30.1: Order-book Alignment Audit

V30 showed a data-alignment problem rather than a usable validation result:
price-only 15m-trend/5m-pullback setups existed, but none of the validation
setup timestamps had a usable exact 1m order-book delta pair.

This update does not add a new strategy. It adds a **freshness audit** that
answers whether an as-of order-book gate could be tested honestly.

For every price-only setup it reports:

- exact observed order-book value at the setup timestamp
- latest observed value at or before the setup and its age in seconds
- previous observed value, pair gap, and observed delta
- counts eligible under explicit max-age and max-pair-gap limits

No values are carried forward into backtests. This is measurement only.

## Apply

Extract the archive into the project root and run:

```powershell
python apply_v30_1_order_book_alignment_audit.py
python -m pytest -q
```

## Run the audit

```powershell
python -m scripts.audit_order_book_alignment `
  --market-data-source production `
  --limit 50000 `
  --order-book-features imbalance_top_20,imbalance_top_5 `
  --max-age-seconds 30,60,120,180,300 `
  --max-pair-gap-seconds 60,120,180,300 `
  --min-feature-samples 100 `
  --export-json reports/order_book_alignment_v30_1.json `
  --export-csv reports/order_book_alignment_v30_1.csv
```

Proceed to an as-of gate only if validation has at least 10 price setups with
both a recent observation and a recent observed pair under a conservative
predefined freshness rule, such as age <= 120 seconds and pair gap <= 120
seconds. Otherwise, continuous collection is required before further strategy
research.
