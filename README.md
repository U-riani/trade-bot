# Crypto Trading Bot

A local-first, performance-conscious crypto trading system designed to collect real-time crypto market data, analyze it quickly, generate trading signals, apply strict risk management rules, and execute buy/sell operations through exchange APIs.

This is not only a test/demo script. The codebase is structured so it can become a real working system. The first implementation intentionally starts with **paper trading** and **Binance Spot Testnet** support before live trading, because letting a newborn bot touch real money is how humans invent avoidable disasters.

## Current MVP Status

Implemented in this version:

- Python 3.12+ async application
- Binance Spot Testnet WebSocket market data listener
- BTCUSDT 1-minute candle stream
- In-memory market state
- EMA + RSI strategy
- Risk manager before execution
- Position guard for stop-loss and take-profit exits
- Paper trading executor
- Optional Binance testnet REST executor skeleton
- Structured JSON logging
- PostgreSQL / TimescaleDB Docker service
- SQL migration file
- Tests for indicators, strategy, risk manager, and paper execution
- Live trading guard that blocks accidental real execution

## Safety First

Default mode is always:

```env
TRADE_MODE=paper
ENABLE_LIVE_TRADING=false
```

Live trading must remain disabled until the bot is validated in paper and testnet mode.

Correct execution flow:

```text
Strategy → Signal → Risk Manager → Executor → Exchange
```

Incorrect execution flow:

```text
Strategy → Exchange
```

Strategy code must never place orders directly.

## Project Structure

```text
crypto-trading-bot/
│
├── app/
│   ├── main.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── logging.py
│   │
│   ├── exchange/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── binance_ws.py
│   │   ├── binance_rest.py
│   │   └── models.py
│   │
│   ├── market/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── candle_builder.py
│   │   ├── indicators.py
│   │   └── models.py
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ema_rsi.py
│   │   └── models.py
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── position_guard.py
│   │   ├── rules.py
│   │   └── models.py
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── paper_executor.py
│   │   ├── testnet_executor.py
│   │   ├── live_executor.py
│   │   └── models.py
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── repositories.py
│   │   └── migrations/
│   │       └── 001_init.sql
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── metrics.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── events.py
│   │   └── health.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── time.py
│       └── ids.py
│
├── tests/
│   ├── test_indicators.py
│   ├── test_strategy.py
│   ├── test_risk_manager.py
│   └── test_paper_executor.py
│
├── scripts/
│   ├── init_db.sql
│   └── run_bot.py
│
├── docs/
│   ├── architecture.md
│   ├── execution-flow.md
│   ├── risk-management.md
│   └── strategy-design.md
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Runtime Workers

The main runtime starts independent async workers:

```text
market_data_worker → receives Binance WebSocket candles
strategy_worker    → updates market state and generates signals
risk_worker        → approves/rejects signals
execution_worker   → executes paper/testnet orders
health_worker      → monitors stale data and kill switch behavior
```

Internal queues:

```text
market_event_queue → strategy_worker
signal_queue       → risk_worker
approved_queue     → execution_worker
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

Windows Git Bash / PowerShell:

```bash
.venv/Scripts/activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .[dev]
```

### 3. Create local environment file

```bash
cp .env.example .env
```

For first run, keep:

```env
TRADE_MODE=paper
```

### 4. Run bot

```bash
python -m app.main
```

or:

```bash
python scripts/run_bot.py
```

### 5. Run tests

```bash
pytest
```

## Docker Services

Start TimescaleDB/PostgreSQL:

```bash
docker compose up -d db
```

Stop services:

```bash
docker compose down
```

Run database migration manually:

```bash
psql "postgresql://trader:trader_password@localhost:5432/trading_bot" -f app/storage/migrations/001_init.sql
```

## Trading Modes

### Paper mode

Simulates orders locally. This is the default mode.

```env
TRADE_MODE=paper
```

### Testnet mode

Uses Binance Spot Testnet REST API for orders.

```env
TRADE_MODE=testnet
BINANCE_TESTNET=true
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
```

### Live mode

Live mode is intentionally blocked unless explicitly enabled.

```env
TRADE_MODE=live
ENABLE_LIVE_TRADING=true
```

The current `LiveExecutor` is a guard placeholder and raises an error. This is intentional. Build testnet execution first, then implement live execution only after validation.

## Important Configuration

```env
APP_ENV=local
TRADE_MODE=paper

EXCHANGE=binance
SYMBOL=BTCUSDT
BASE_ASSET=BTC
QUOTE_ASSET=USDT
TIMEFRAME=1m

BINANCE_TESTNET=true
BINANCE_API_KEY=
BINANCE_API_SECRET=

ENABLE_LIVE_TRADING=false

MAX_ORDER_USDT=10
MAX_POSITION_USDT=50
MAX_DAILY_LOSS_USDT=10
MAX_TRADES_PER_HOUR=5
STOP_LOSS_PCT=0.7
TAKE_PROFIT_PCT=1.2
COOLDOWN_SECONDS=60
ALLOW_ONLY_ONE_OPEN_POSITION=true
```

## Strategy

Current strategy:

```text
EMA 9 / EMA 21 crossover + RSI filter
```

BUY signal:

- EMA fast crosses above EMA slow
- RSI is within configured buy range

SELL signal:

- EMA fast crosses below EMA slow
- or RSI is above configured sell threshold

The first version acts only on **closed candles**, not partially formed candles.

## Risk Rules

Current MVP risk manager checks:

- Kill switch
- Signal side validity
- Max daily loss for new BUY entries
- Cooldown after trade for new BUY entries
- Max trades per hour for new BUY entries
- One open position rule
- Max order size
- Max position size
- SELL only if position exists

SELL exits are intentionally not blocked by cooldown, max-trades-per-hour, or daily-loss limits. If a position is already open, the system must be able to close it quickly when strategy exit, stop-loss, or take-profit logic triggers.

If a signal is rejected, the reason is logged.

## Position Guard

The position guard checks every fresh closed candle after the latest price is updated. If an open position exists, it calculates protective exit levels from the average entry price.

Configuration:

```env
STOP_LOSS_PCT=0.7
TAKE_PROFIT_PCT=1.2
```

Behavior:

```text
stop_loss_price   = avg_entry_price * (1 - STOP_LOSS_PCT / 100)
take_profit_price = avg_entry_price * (1 + TAKE_PROFIT_PCT / 100)
```

If the latest price is below or equal to the stop-loss level, the guard emits a protective SELL signal with reason `stop_loss_triggered`.

If the latest price is above or equal to the take-profit level, the guard emits a protective SELL signal with reason `take_profit_triggered`.

The guard runs before the EMA/RSI strategy on each candle. Protective exits have priority over normal strategy signals.

## Development Roadmap

1. Stabilize paper mode
2. Add persistent DB writer worker
3. Add protective stop-loss/take-profit position guard
4. Add Binance testnet order execution validation
5. Add order status verification after unclear responses
6. Add historical data loader
7. Add backtesting reports
8. Add dashboard/API
9. Add more strategies
10. Add multi-symbol support
11. Add live execution only after strict validation

## Disclaimer

This project is personal engineering/research software. Crypto trading involves financial risk. No strategy guarantees profit. Use paper mode and testnet mode before considering any real funds.

## Database Check and Runtime Storage

By default, database writing is disabled so the bot can run safely even when PostgreSQL is not running.

Enable database storage in `.env`:

```env
DATABASE_ENABLED=true
DATABASE_APPLY_MIGRATIONS_ON_START=false
```

Apply migrations manually:

```bash
python -m scripts.apply_migrations
```

Check database connection and required tables:

```bash
python -m scripts.check_db
```

Expected successful logs:

```text
db_connection_ok
db_table_ok table=candles
db_table_ok table=signals
db_table_ok table=risk_decisions
db_table_ok table=orders
db_table_ok table=positions
db_table_ok table=bot_events
```

When `DATABASE_ENABLED=true`, the main bot startup should show:

```text
db_connected database=trading_bot user=trader
```

The bot then writes candles and strategy signals asynchronously through the DB writer worker.


### PostgreSQL vs TimescaleDB

By default the project uses plain PostgreSQL-compatible migrations. TimescaleDB is optional.

For normal local PostgreSQL, keep:

```env
DATABASE_USE_TIMESCALEDB=false
```

For the Docker TimescaleDB service or a PostgreSQL server where TimescaleDB is installed, you may enable:

```env
DATABASE_USE_TIMESCALEDB=true
```

If `DATABASE_USE_TIMESCALEDB=true` is used on a normal PostgreSQL server, migration will fail because the `timescaledb` extension is not available.

## Startup Candle Warm-Up

The bot can load recent candles from PostgreSQL on startup so the strategy does not need to wait 23 fresh 1-minute candles after every restart.

Configuration:

```env
LOAD_RECENT_CANDLES_ON_START=true
STARTUP_CANDLE_LIMIT=100
STARTUP_CANDLE_MAX_AGE_SECONDS=180
STARTUP_CANDLE_GAP_TOLERANCE_SECONDS=2
```

Startup behavior:

1. Connect to PostgreSQL.
2. Load the latest candles for the configured symbol/timeframe.
3. Validate that the latest candle is fresh.
4. Validate that candles are continuous and do not have large gaps.
5. If validation passes, fill `MarketState` before the WebSocket stream starts processing new candles.
6. If validation fails, reject DB warm-up and wait for fresh live candles.

This prevents the bot from trading from yesterday's stale candle history.

Expected logs when warm-up works:

```text
startup_candle_warmup_loaded loaded_count=100 latest_close_time=...
```

Expected logs when DB candles are stale or broken:

```text
startup_candle_warmup_rejected reason='stale_candles: ...'
```

If you already created the database before this feature, run migrations again once to add the extra candle lookup index:

```bash
python -m scripts.apply_migrations
```

## Startup REST Backfill

When DB candle warm-up is rejected because candles are stale or have gaps, the bot can fetch recent closed candles from Binance REST before starting the WebSocket stream. This prevents the strategy from waiting for 23 fresh live candles after every restart.

Recommended settings:

```env
LOAD_RECENT_CANDLES_ON_START=true
STARTUP_CANDLE_LIMIT=100
STARTUP_CANDLE_MAX_AGE_SECONDS=180
STARTUP_CANDLE_GAP_TOLERANCE_SECONDS=2
STARTUP_REST_BACKFILL_ENABLED=true
STARTUP_REST_BACKFILL_LIMIT=100
```

The backfill only uses closed candles and still validates freshness and continuity before loading them into `MarketState`.

## V9: Paper Trade Cycle Verification

V9 adds a controlled paper-only testing path for verifying the full internal trading cycle:

```text
BUY signal → risk approval → paper order fill → position snapshot → protective SELL → position closed → PnL updated
```

### Paper-only forced BUY test

The bot can generate one forced BUY signal on the first live closed candle. This exists only to verify the complete paper execution pipeline and must stay disabled during normal usage.

```env
PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE=false
PAPER_TEST_FORCE_BUY_QUOTE_AMOUNT=10
```

Safety rule:

```text
PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE=true is rejected unless TRADE_MODE=paper.
```

To test the live runtime pipeline in paper mode:

```env
TRADE_MODE=paper
DATABASE_ENABLED=true
PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE=true
PAPER_TEST_FORCE_BUY_QUOTE_AMOUNT=10
```

Then run:

```bash
python -m app.main
```

Expected logs after the first closed candle:

```text
paper_test_signal
risk_approved
paper_order_filled
order_result
portfolio_snapshot
```

After the first BUY is confirmed, set `PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE=false` again.

### Offline paper cycle demo

You can test the full BUY → take-profit SELL cycle without WebSocket or DB:

```bash
python -m scripts.simulate_paper_cycle
```

Expected final log:

```text
paper_cycle_demo_success
```

### Position persistence

After every filled paper order, the bot now saves the latest position snapshot into the `positions` table when DB is enabled. The table is upserted by `symbol`, so the latest state is kept instead of appending endless duplicate position rows.

## V10: Paper Position Recovery on Restart

V10 restores the latest saved paper portfolio snapshot from PostgreSQL on startup.

This prevents the bot from forgetting an already-open paper position after a restart. If `PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE=true` is still enabled, the forced test BUY is skipped when a restored open position exists.

Recommended setting:

```env
LOAD_PAPER_POSITION_ON_START=true
```

Startup flow:

```text
1. Connect to DB
2. Load / backfill recent candles
3. Build paper executor
4. Restore latest paper position snapshot from DB
5. Start WebSocket
6. Skip duplicate forced BUY if position is already open
```

Expected logs when a position is restored:

```text
paper_portfolio_restored
paper_position_restore_loaded
```

The `positions` table now includes `quote_balance`, so restoring state brings back both the open position and remaining paper cash balance. Run migrations once after upgrading:

```bash
python -m scripts.apply_migrations
```

## V11: Paper State Inspection Helpers

V11 adds two local scripts that make paper trading validation easier.

### Show current DB and paper state

```bash
python -m scripts.show_state
```

This prints:

- table row counts
- latest candle
- latest signal
- latest order
- current paper position
- realized PnL
- unrealized PnL
- estimated total paper equity

### Reset local paper position

```bash
python -m scripts.reset_paper_state --reset-realized-pnl
```

This resets the local paper position for the configured symbol back to:

- quantity = 0
- average entry price = 0
- quote balance = `INITIAL_QUOTE_BALANCE`
- realized PnL = 0 when `--reset-realized-pnl` is used

To reset with a custom balance:

```bash
python -m scripts.reset_paper_state --quote-balance 1000 --reset-realized-pnl
```

This does not delete historical candles, signals, risk decisions, or orders. It only resets the current local paper portfolio snapshot.

## Backtesting

V13 includes a local backtesting command for the configured EMA/RSI strategy. It replays historical candles, checks stop-loss/take-profit exits, simulates one long spot position at a time, and prints summary metrics. V13 also includes fee/slippage simulation and report exports, because no-fee backtests are basically bedtime stories for traders.

Run with candles from PostgreSQL first, falling back to Binance REST when needed:

```bash
python -m scripts.backtest_strategy --source auto --limit 500
```

Use only PostgreSQL candles:

```bash
python -m scripts.backtest_strategy --source db --limit 500
```

Use Binance REST candles:

```bash
python -m scripts.backtest_strategy --source rest --limit 500
```

Show each completed round trip:

```bash
python -m scripts.backtest_strategy --source auto --limit 500 --show-trades
```

Run with explicit fee and slippage values:

```bash
python -m scripts.backtest_strategy --source auto --limit 500 --fee-rate-pct 0.1 --slippage-pct 0.02 --show-trades
```

Export a JSON summary and CSV trade list:

```bash
python -m scripts.backtest_strategy --source auto --limit 500 --show-trades --export-json reports/backtest.json --export-csv reports/backtest_trades.csv
```

Useful `.env` values:

```env
BACKTEST_FEE_RATE_PCT=0.1
BACKTEST_SLIPPAGE_PCT=0.02
```

The current backtester is intentionally conservative and still does not include partial fills, funding, order book depth, exchange execution delays, or order queue priority. It is useful for sanity-checking strategy behavior before paper/testnet runtime, not for pretending the future signed a contract.

## Historical candle backfill

### V15 historical data source separation

V15 separates **runtime safety** from **historical research data**. Runtime can stay in `TRADE_MODE=paper` with `BINANCE_TESTNET=true`, while backfill/backtest scripts can use public Binance production candles for realistic historical prices. This matters because Binance Spot Testnet data is synthetic and can contain unrealistic historical jumps. Using that for strategy judgment is how a spreadsheet becomes fan fiction.

Recommended setting for strategy research:

```env
HISTORICAL_MARKET_DATA_SOURCE=production
```

This stores historical research candles under a separate exchange key:

```text
production -> binance_spot
testnet    -> binance_testnet
```

Runtime paper/testnet state still uses the normal runtime configuration. The goal is simple: realistic public history for backtests, safe paper/testnet execution for runtime.

Backfill production candles:

```powershell
python -m scripts.backfill_candles --market-data-source production --limit 5000
```

Backtest production candles from DB:

```powershell
python -m scripts.backtest_strategy --market-data-source production --source db --limit 5000 --show-trades
```

Compare against testnet candles only when debugging plumbing:

```powershell
python -m scripts.backfill_candles --market-data-source testnet --limit 1000
python -m scripts.backtest_strategy --market-data-source testnet --source db --limit 1000 --show-trades
```


V14 adds a dedicated historical backfill command so the database can hold thousands of 1-minute candles instead of a tiny runtime sample. The command pages through Binance REST history, filters out any open candle, saves unique closed candles into PostgreSQL, and leaves existing rows untouched.

Fetch and save the latest 5,000 closed candles:

```bash
python -m scripts.backfill_candles --limit 5000
```

Fetch 10,000 candles and validate continuity:

```bash
python -m scripts.backfill_candles --limit 10000 --validate-continuity
```

Dry run without saving:

```bash
python -m scripts.backfill_candles --limit 5000 --no-save
```

Then run a larger DB-backed backtest:

```bash
python -m scripts.backtest_strategy --source db --limit 5000 --show-trades --export-json reports/backtest_5000.json --export-csv reports/backtest_5000_trades.csv
```

For serious strategy testing, prefer this flow:

```bash
python -m scripts.backfill_candles --limit 10000
python -m scripts.check_db
python -m scripts.backtest_strategy --source db --limit 10000 --show-trades
```

A bigger sample will not magically make the strategy profitable. It just gives us enough data to discover failure honestly, which is apparently the adult version of optimism.

## V16 strategy optimizer

V16 adds a local parameter optimizer for the EMA/RSI strategy. It reuses the same backtest engine, fee/slippage settings, stop-loss/take-profit guard, and production/testnet historical-data separation from V15.

Default optimizer run, using production candles already backfilled into PostgreSQL:

```powershell
python -m scripts.optimize_strategy --market-data-source production --source db --limit 10000 --top 20 --export-json reports/optimization_prod_10000.json --export-csv reports/optimization_prod_10000.csv
```

The default grid is intentionally moderate:

```text
EMA fast:      5, 9, 12
EMA slow:      21, 34
RSI period:    14
RSI buy min:   45
RSI buy max:   65, 70
RSI sell min:  70, 75
Stop-loss %:   0.5, 0.7, 1.0
Take-profit %: 0.8, 1.2, 1.8
```

That is 216 combinations, which is enough to discover whether the current shape is promising without asking the CPU to file a workplace complaint.

You can run a wider search like this:

```powershell
python -m scripts.optimize_strategy --market-data-source production --source db --limit 10000 --ema-fast-values 5,8,9,12 --ema-slow-values 21,34,55 --rsi-buy-min-values 40,45,50 --rsi-buy-max-values 60,65,70 --rsi-sell-min-values 70,75,80 --stop-loss-pct-values 0.5,0.7,1.0,1.5 --take-profit-pct-values 0.8,1.2,1.8,2.5 --min-round-trips 10 --top 30 --export-json reports/optimization_wide.json --export-csv reports/optimization_wide.csv
```

Output logs include:

```text
strategy_optimization_started
strategy_optimization_finished
strategy_optimization_result
strategy_optimization_json_exported
strategy_optimization_csv_exported
```

The CSV is usually the easiest file to inspect first. Sort by `rank`, `final_equity`, `return_pct`, `max_drawdown`, and `round_trips`. A parameter set with very high return but only one or two trades is usually a lucky accident wearing a lab coat, so use `--min-round-trips` to filter those out.

## V17 filtered strategy optimizer

V17 keeps the V16 optimizer and adds optional buy filters for the EMA/RSI strategy:

```text
Trend EMA filter: only buy when price and slow EMA are above a larger trend EMA
Minimum EMA gap: ignore tiny fast/slow EMA crosses that are usually noise
ATR filter: ignore candles where volatility is too low to overcome fees/slippage
Trade-frequency penalty: ranking now slightly penalizes fees and excessive round trips
```

These filters are disabled by default in runtime config, so the bot does not silently change behavior unless the `.env` or optimizer command enables them:

```env
TREND_EMA_PERIOD=0
MIN_EMA_GAP_PCT=0
ATR_PERIOD=0
MIN_ATR_PCT=0
```

Filtered optimizer default run:

```powershell
python -m scripts.optimize_strategy --market-data-source production --source db --limit 10000 --top 20 --export-json reports/optimization_v17_filtered.json --export-csv reports/optimization_v17_filtered.csv
```

Focused filtered run, useful after V16 shows too much overtrading:

```powershell
python -m scripts.optimize_strategy --market-data-source production --source db --limit 10000 --ema-fast-values 12,15,20 --ema-slow-values 34,55,89 --rsi-buy-min-values 40,45,50 --rsi-buy-max-values 55,60,65 --rsi-sell-min-values 65,70,75 --stop-loss-pct-values 0.5,0.7,1.0 --take-profit-pct-values 0.8,1.2,1.8 --trend-ema-values 0,200 --min-ema-gap-pct-values 0,0.03,0.06,0.1 --atr-period-values 0,14 --min-atr-pct-values 0,0.05,0.08,0.12 --min-round-trips 10 --top 30 --export-json reports/optimization_v17_focused.json --export-csv reports/optimization_v17_focused.csv
```

Interpretation rule: a result is not interesting just because it is rank 1. Prefer results with positive final equity, reasonable drawdown, enough trades, and fewer fees. If all results are still negative, the strategy family is not good enough and the next step should be a different signal design, not worshipping the same EMA crossover harder.

## V18 strategy comparison and walk-forward validation

V18 adds a small research harness so you do not keep torturing one EMA/RSI setup forever, which is admirable because CPUs deserve hobbies too.

New pieces:

- `app.strategy.breakout_momentum.BreakoutMomentumStrategy`
- `app.backtesting.benchmarks.no_trade_benchmark`
- `app.backtesting.benchmarks.buy_and_hold_order_sized_benchmark`
- `app.backtesting.benchmarks.split_walk_forward`
- `scripts.compare_strategies`

The comparison script evaluates:

1. `no_trade` benchmark
2. `buy_hold_order_sized` benchmark using `MAX_ORDER_USDT`
3. `ema_rsi_v17_best_region`
4. `breakout_momentum_v1`

It runs each on:

- full dataset
- train segment
- validation segment

Run it against your clean production candle DB:

```powershell
python -m scripts.compare_strategies --market-data-source production --source db --limit 10000 --train-ratio 0.7 --export-json reports/strategy_comparison_v18.json --export-csv reports/strategy_comparison_v18.csv
```

The default EMA/RSI parameters are the best V17 region seen so far:

```text
EMA 12 / EMA 34
RSI buy 45-60
RSI sell 70
SL 0.5
TP 0.8
Trend EMA 200
ATR 14
Min ATR 0.08%
```

The breakout candidate defaults to:

```text
breakout lookback 20
exit lookback 10
trend EMA 200
ATR 14
min ATR 0.08%
```

A candidate is not interesting unless it beats `no_trade` and is not only good on the train segment while failing on validation. That would be overfitting, the financial equivalent of memorizing the exam answers and then forgetting how numbers work.

## V19 multi-timeframe comparison

V19 adds local candle resampling so the same 1m production candle database can be
compared as `1m`, `5m`, and `15m` without downloading separate data first.
This is mainly for strategy research: V18 showed that the implementation works,
but 1m EMA/RSI and simple breakout signals were still losing to the brutally
powerful benchmark known as "doing nothing".

The comparison script now accepts:

```text
--timeframes 1m,5m,15m
--source-timeframe 1m
--min-candles-per-timeframe 200
```

Default V19 comparison:

```powershell
python -m scripts.compare_strategies --market-data-source production --source db --limit 10000 --train-ratio 0.7 --timeframes 1m,5m,15m --export-json reports/strategy_comparison_v19_timeframes.json --export-csv reports/strategy_comparison_v19_timeframes.csv
```

Focused 5m/15m comparison:

```powershell
python -m scripts.compare_strategies --market-data-source production --source db --limit 10000 --train-ratio 0.7 --timeframes 5m,15m --export-json reports/strategy_comparison_v19_5m_15m.json --export-csv reports/strategy_comparison_v19_5m_15m.csv
```

Read validation rows first. A candidate is not interesting unless it beats
`no_trade` and preferably `buy_hold_order_sized` on validation, not just on the
full in-sample window. Backtests love lying; V19 at least makes them lie in
separate columns.


## V20 market-regime filter

V20 adds a market-regime gate around the EMA/RSI strategy. The goal is simple:
if the larger market context is not bullish, the bot should stay in cash instead
of performing tiny long-only rituals while BTC is falling. That is not glamour,
but apparently survival is a feature now.

The comparison script now includes a new strategy candidate:

```text
regime_filtered_ema_rsi_v20
```

It wraps the V17 EMA/RSI region and only allows BUY signals when the current
regime is bullish according to:

```text
regime fast EMA = 50
regime slow EMA = 200
regime slow EMA slope lookback = 20 candles
minimum slow EMA slope = 0.03%
minimum fast-vs-slow EMA gap = 0.05%
```

Run the V20 comparison on the larger production candle set:

```powershell
python -m scripts.compare_strategies --market-data-source production --source db --limit 50000 --train-ratio 0.7 --timeframes 5m,15m --top 10 --export-json reports/strategy_comparison_v20_regime_50000.json --export-csv reports/strategy_comparison_v20_regime_50000.csv
```

If the regime-filtered strategy beats `no_trade` on validation, it becomes a
real candidate for further optimization. If it only reduces losses, that still
teaches us something useful: the bot needs better entry logic, not just a
slightly more nervous doorman.

You can loosen the regime gate if it produces too few trades:

```powershell
python -m scripts.compare_strategies --market-data-source production --source db --limit 50000 --train-ratio 0.7 --timeframes 5m,15m --regime-min-slope-pct 0.01 --regime-min-ema-gap-pct 0.02 --top 10 --export-json reports/strategy_comparison_v20_regime_loose_50000.json --export-csv reports/strategy_comparison_v20_regime_loose_50000.csv
```

You can also make it stricter if it still trades during weak market conditions:

```powershell
python -m scripts.compare_strategies --market-data-source production --source db --limit 50000 --train-ratio 0.7 --timeframes 5m,15m --regime-min-slope-pct 0.06 --regime-min-ema-gap-pct 0.1 --top 10 --export-json reports/strategy_comparison_v20_regime_strict_50000.json --export-csv reports/strategy_comparison_v20_regime_strict_50000.csv
```
