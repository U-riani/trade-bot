# Architecture

The bot is event-driven and splits market data, strategy, risk, and execution into separate layers.

```text
Exchange WebSocket
        ↓
Market Data Worker
        ↓
Market State
        ↓
Strategy Worker
        ↓
Risk Worker
        ↓
Execution Worker
```

The database is intentionally outside the hot decision path. Market decisions should use in-memory state.
