from __future__ import annotations

from decimal import Decimal

import orjson

from app.execution.models import OrderResult, PortfolioSnapshot
from app.market.models import Candle
from app.risk.models import RiskDecision
from app.storage.db import Database
from app.strategy.models import TradeSignal


class TradingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save_candle(self, candle: Candle) -> None:
        pool = self.db.require_pool()
        await pool.execute(
            """
            INSERT INTO candles(exchange, symbol, timeframe, open_time, close_time, open, high, low, close, volume)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT DO NOTHING
            """,
            candle.exchange,
            candle.symbol,
            candle.timeframe,
            candle.open_time,
            candle.close_time,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

    async def save_candles(self, candles: list[Candle]) -> None:
        for candle in candles:
            await self.save_candle(candle)

    async def save_signal(self, signal: TradeSignal) -> None:
        pool = self.db.require_pool()
        await pool.execute(
            """
            INSERT INTO signals(strategy_name, symbol, side, confidence, reason, created_at)
            VALUES($1,$2,$3,$4,$5,$6)
            """,
            signal.strategy_name,
            signal.symbol,
            signal.side.value,
            signal.confidence,
            signal.reason,
            signal.created_at,
        )

    async def save_risk_decision(self, decision: RiskDecision) -> None:
        pool = self.db.require_pool()
        await pool.execute(
            """
            INSERT INTO risk_decisions(status, reason, created_at)
            VALUES($1,$2,$3)
            """,
            decision.status.value,
            decision.reason,
            decision.created_at,
        )

    async def save_order_result(self, order: OrderResult) -> None:
        pool = self.db.require_pool()
        raw = orjson.dumps(order.raw_response).decode("utf-8")
        await pool.execute(
            """
            INSERT INTO orders(client_order_id, exchange_order_id, symbol, side, status,
                               executed_quantity, executed_quote_quantity, raw_response, created_at)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)
            ON CONFLICT (client_order_id) DO NOTHING
            """,
            order.client_order_id,
            order.exchange_order_id,
            order.symbol,
            order.side.value,
            order.status.value,
            order.executed_quantity,
            order.executed_quote_quantity,
            raw,
            order.created_at,
        )



    async def save_position_snapshot(self, symbol: str, snapshot: PortfolioSnapshot) -> None:
        pool = self.db.require_pool()
        await pool.execute(
            """
            INSERT INTO positions(symbol, quantity, avg_entry_price, realized_pnl, quote_balance, updated_at)
            VALUES($1,$2,$3,$4,$5,NOW())
            ON CONFLICT (symbol) DO UPDATE SET
                quantity = EXCLUDED.quantity,
                avg_entry_price = EXCLUDED.avg_entry_price,
                realized_pnl = EXCLUDED.realized_pnl,
                quote_balance = EXCLUDED.quote_balance,
                updated_at = NOW()
            """,
            symbol,
            snapshot.position_quantity,
            snapshot.position_avg_entry_price,
            snapshot.realized_pnl_today,
            snapshot.quote_balance,
        )

    async def load_position_snapshot(
        self,
        *,
        symbol: str,
        fallback_quote_balance: Decimal,
        latest_price: Decimal | None,
    ) -> PortfolioSnapshot | None:
        pool = self.db.require_pool()
        row = await pool.fetchrow(
            """
            SELECT quantity, avg_entry_price, realized_pnl, quote_balance
            FROM positions
            WHERE symbol = $1
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            symbol,
        )

        if row is None:
            return None

        position_quantity = Decimal(str(row["quantity"]))
        position_avg_entry_price = Decimal(str(row["avg_entry_price"]))
        realized_pnl = Decimal(str(row["realized_pnl"]))

        quote_balance = row["quote_balance"]
        if quote_balance is None:
            # Older V9 position rows did not store quote_balance. Estimate the
            # paper cash balance from the initial paper balance, open cost basis,
            # and realized PnL so existing local test positions still recover
            # sensibly after the schema upgrade.
            quote_balance = fallback_quote_balance - (position_quantity * position_avg_entry_price) + realized_pnl

        return PortfolioSnapshot(
            quote_balance=Decimal(str(quote_balance)),
            position_quantity=position_quantity,
            position_avg_entry_price=position_avg_entry_price,
            realized_pnl_today=realized_pnl,
            latest_price=latest_price,
        )

    async def load_recent_candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[Candle]:
        if limit <= 0:
            return []

        pool = self.db.require_pool()
        rows = await pool.fetch(
            """
            SELECT exchange, symbol, timeframe, open_time, close_time, open, high, low, close, volume
            FROM candles
            WHERE exchange = $1
              AND symbol = $2
              AND timeframe = $3
            ORDER BY close_time DESC
            LIMIT $4
            """,
            exchange,
            symbol,
            timeframe,
            limit,
        )

        candles = [
            Candle(
                exchange=row["exchange"],
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                open_time=row["open_time"],
                close_time=row["close_time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                is_closed=True,
            )
            for row in rows
        ]

        candles.reverse()
        return candles

    async def count_candles(self, *, exchange: str, symbol: str, timeframe: str) -> int:
        pool = self.db.require_pool()
        value = await pool.fetchval(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE exchange = $1
              AND symbol = $2
              AND timeframe = $3
            """,
            exchange,
            symbol,
            timeframe,
        )
        return int(value or 0)

    async def count_table(self, table_name: str) -> int:
        allowed = {"candles", "signals", "risk_decisions", "orders", "positions", "bot_events"}
        if table_name not in allowed:
            raise ValueError(f"Unsupported table: {table_name}")
        pool = self.db.require_pool()
        value = await pool.fetchval(f"SELECT COUNT(*) FROM {table_name}")
        return int(value or 0)
