# V31.1 Candidate Training Diagnostics

## Purpose

V31 may correctly select no candidate in a fold. This diagnostic reports all
pre-registered V31 candidates on every **training** fold, with no validation
selection and no new strategy rules.

## What it measures

For every candidate and fold:

- signals available inside the training window;
- completed trades after next-1m-open entry and fixed-horizon exit;
- net PnL after the existing fee and slippage model;
- profit factor, max drawdown, fees, and skipped signals;
- all rejection reasons:
  - insufficient trade count;
  - non-positive post-cost PnL;
  - profit factor not above 1.

## Interpretation

This report is not a strategy selector and cannot authorize execution. It is
used to decide the *next research family*:

- **Trade scarcity dominates**: the entry condition is too rare for the tested
  horizon, so research should change market/holding regime, not lower standards.
- **Negative PnL dominates**: reject the family under this cost model.
- **Fees erase gross winners**: research longer holding horizons or a lower-turnover
  family, subject to a new pre-registered walk-forward test.
- **A stable near-miss appears**: define one narrowly-scoped follow-up family and
  retest it from scratch with separate validation windows.

The report never changes V31's pass criteria and never uses validation returns to
rank a candidate.
