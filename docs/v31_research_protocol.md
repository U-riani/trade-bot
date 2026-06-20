# V31 price-only walk-forward protocol

V31 evaluates a fixed catalog of three long-only price strategy families:

1. 15m bullish trend + 5m pullback.
2. 15m bullish trend + 5m mean reversion.
3. 15m bullish breakout.

It does **not** use live order-book data. That data is still being collected
and remains a future confirmation filter once its validation coverage is large
enough.

## Why this resembles NautilusTrader research discipline

NautilusTrader is used as a design reference, not a runtime dependency in this
update. Its event-driven research/live parity is a useful standard. V31 mirrors
that standard by using completed-bar signals, next-bar entry, explicit fees and
slippage, deterministic state transitions, and tests around timing/gaps.

The current project keeps its own runner because its PostgreSQL candle pipeline
and current reports need comparable semantics. A direct framework swap before a
candidate survives walk-forward validation would change several variables at
once and teach us nothing useful.

## Selection and validation

For each fold:

- Every candidate is evaluated only inside the train window.
- A candidate is eligible only if it has enough train trades, positive net PnL,
  and profit factor above 1 after modeled costs.
- The best eligible train candidate is selected.
- Only that selected candidate runs in the immediate following validation window.
- Validation windows are stitched with carried equity and no open positions.

## Pass criteria

The complete validation sequence must have:

- At least 10 completed trades.
- Positive cumulative net PnL after modeled fees/slippage.
- Profit factor above 1.
- At least two profitable validation folds.
- No single winner contributing more than 50% of gross profit.
- Better cumulative return than buy-and-hold across the same validation folds.

`promising_research_only` is not a live-trading approval. It only qualifies the
candidate for a later independent-engine parity run and forward paper-trading.
