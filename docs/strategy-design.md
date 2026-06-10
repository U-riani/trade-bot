# Strategy Design

Initial strategy: EMA + RSI.

BUY:

- EMA fast crosses above EMA slow
- RSI is inside the configured buy range

SELL:

- EMA fast crosses below EMA slow
- or RSI is above the sell threshold

The MVP uses closed candles only.
