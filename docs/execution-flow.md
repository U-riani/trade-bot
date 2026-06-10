# Execution Flow

1. Binance WebSocket sends a closed candle.
2. Market state stores the candle.
3. Strategy generates BUY / SELL / HOLD.
4. HOLD is ignored.
5. BUY / SELL goes to the risk manager.
6. Risk manager approves or rejects.
7. Approved orders go to executor.
8. Executor fills paper/testnet order.
9. Result is logged.

Strategy never sends orders directly.
