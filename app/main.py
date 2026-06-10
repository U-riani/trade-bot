from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.config.logging import configure_logging, get_logger
from app.config.settings import Settings, TradeMode, get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.exchange.binance_ws import BinanceWebSocketStream
from app.execution.executor import OrderExecutor
from app.execution.live_executor import LiveExecutor
from app.execution.models import OrderResult, OrderStatus
from app.execution.paper_executor import PaperExecutor
from app.market.bootstrap import validate_startup_candles
from app.market.models import Candle, MarketEvent, MarketEventType
from app.market.state import MarketState
from app.monitoring.health import health_worker
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.risk.position_guard import build_position_exit_signal
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.models import SignalSide, TradeSignal

logger = get_logger(__name__)

DbLogItem = tuple[str, Any]


async def market_data_worker(
    stream: BinanceWebSocketStream,
    market_event_queue: asyncio.Queue[MarketEvent],
) -> None:
    async for event in stream.stream():
        await market_event_queue.put(event)


async def strategy_worker(
    market_event_queue: asyncio.Queue[MarketEvent],
    signal_queue: asyncio.Queue[TradeSignal],
    market_state: MarketState,
    strategy: EmaRsiStrategy,
    executor: OrderExecutor,
    stop_loss_pct: Decimal,
    take_profit_pct: Decimal,
    db_log_queue: asyncio.Queue[DbLogItem] | None = None,
    paper_test_force_buy_on_first_candle: bool = False,
    paper_test_force_buy_quote_amount: Decimal | None = None,
) -> None:
    paper_test_entry_sent = False

    while True:
        event = await market_event_queue.get()
        try:
            if event.type == MarketEventType.CANDLE_CLOSED and event.candle is not None:
                candle = event.candle
                candle_added = market_state.add_candle(candle)
                if not candle_added:
                    logger.warning(
                        "candle_ignored_duplicate_or_out_of_order",
                        symbol=candle.symbol,
                        timeframe=candle.timeframe,
                        open_time=candle.open_time.isoformat(),
                        close_time=candle.close_time.isoformat(),
                    )
                    continue

                if hasattr(executor, "set_latest_price"):
                    executor.set_latest_price(Decimal(str(candle.close)))  # type: ignore[attr-defined]

                logger.info(
                    "candle_closed",
                    symbol=candle.symbol,
                    timeframe=candle.timeframe,
                    close=str(candle.close),
                    volume=str(candle.volume),
                    close_time=candle.close_time.isoformat(),
                )

                if db_log_queue is not None:
                    await db_log_queue.put(("candle", candle))

                protective_signal = build_position_exit_signal(
                    portfolio=executor.portfolio_snapshot(),
                    symbol=market_state.symbol,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                )
                if protective_signal is not None:
                    logger.warning(
                        "position_guard_signal",
                        strategy=protective_signal.strategy_name,
                        symbol=protective_signal.symbol,
                        side=protective_signal.side.value,
                        confidence=protective_signal.confidence,
                        reason=protective_signal.reason,
                    )
                    if db_log_queue is not None:
                        await db_log_queue.put(("signal", protective_signal))
                    await signal_queue.put(protective_signal)
                    continue

                if paper_test_force_buy_on_first_candle and not paper_test_entry_sent:
                    paper_test_entry_sent = True
                    snapshot = executor.portfolio_snapshot()
                    if not snapshot.has_open_position:
                        paper_signal = TradeSignal(
                            strategy_name="paper_test",
                            symbol=market_state.symbol,
                            side=SignalSide.BUY,
                            confidence=1.0,
                            reason="paper_test_force_buy_on_first_candle",
                            created_at=candle.close_time,
                            suggested_quote_amount=paper_test_force_buy_quote_amount,
                        )
                        logger.warning(
                            "paper_test_signal",
                            strategy=paper_signal.strategy_name,
                            symbol=paper_signal.symbol,
                            side=paper_signal.side.value,
                            confidence=paper_signal.confidence,
                            reason=paper_signal.reason,
                            suggested_quote_amount=str(paper_signal.suggested_quote_amount),
                        )
                        if db_log_queue is not None:
                            await db_log_queue.put(("signal", paper_signal))
                        await signal_queue.put(paper_signal)
                        continue
                    logger.info("paper_test_force_buy_skipped", reason="position_already_open")

                signal = strategy.on_market_state(market_state)
                logger.info(
                    "strategy_signal",
                    strategy=signal.strategy_name,
                    symbol=signal.symbol,
                    side=signal.side.value,
                    confidence=signal.confidence,
                    reason=signal.reason,
                )

                if db_log_queue is not None:
                    await db_log_queue.put(("signal", signal))

                if signal.side != SignalSide.HOLD:
                    await signal_queue.put(signal)
        finally:
            market_event_queue.task_done()


async def risk_worker(
    signal_queue: asyncio.Queue[TradeSignal],
    approved_queue: asyncio.Queue,
    risk_manager: RiskManager,
    executor: OrderExecutor,
    db_log_queue: asyncio.Queue[DbLogItem] | None = None,
) -> None:
    while True:
        signal = await signal_queue.get()
        try:
            decision = risk_manager.evaluate(signal, executor.portfolio_snapshot())

            if db_log_queue is not None:
                await db_log_queue.put(("risk_decision", decision))

            if decision.approved and decision.order_request is not None:
                logger.info(
                    "risk_approved",
                    symbol=decision.order_request.symbol,
                    side=decision.order_request.side.value,
                    reason=decision.reason,
                )
                await approved_queue.put(decision.order_request)
        finally:
            signal_queue.task_done()


async def execution_worker(
    approved_queue: asyncio.Queue,
    executor: OrderExecutor,
    risk_manager: RiskManager,
    db_log_queue: asyncio.Queue[DbLogItem] | None = None,
) -> None:
    while True:
        order_request = await approved_queue.get()
        try:
            result: OrderResult = await executor.execute(order_request)
            logger.info(
                "order_result",
                client_order_id=result.client_order_id,
                exchange_order_id=result.exchange_order_id,
                symbol=result.symbol,
                side=result.side.value,
                status=result.status.value,
                executed_quantity=str(result.executed_quantity),
                executed_quote_quantity=str(result.executed_quote_quantity),
            )

            if db_log_queue is not None:
                await db_log_queue.put(("order_result", result))

            if result.status == OrderStatus.FILLED:
                risk_manager.register_executed_trade()
                snapshot = executor.portfolio_snapshot()
                logger.info(
                    "portfolio_snapshot",
                    symbol=result.symbol,
                    quote_balance=str(snapshot.quote_balance),
                    position_quantity=str(snapshot.position_quantity),
                    position_avg_entry_price=str(snapshot.position_avg_entry_price),
                    realized_pnl_today=str(snapshot.realized_pnl_today),
                    latest_price=str(snapshot.latest_price),
                )
                if db_log_queue is not None:
                    await db_log_queue.put(("position_snapshot", (result.symbol, snapshot)))
            elif result.status == OrderStatus.UNKNOWN:
                risk_manager.activate_kill_switch()
        finally:
            approved_queue.task_done()


async def db_writer_worker(
    db_log_queue: asyncio.Queue[DbLogItem],
    repository: TradingRepository,
) -> None:
    while True:
        item_type, payload = await db_log_queue.get()
        try:
            if item_type == "candle":
                await repository.save_candle(payload)
            elif item_type == "signal":
                await repository.save_signal(payload)
            elif item_type == "risk_decision":
                await repository.save_risk_decision(payload)
            elif item_type == "order_result":
                await repository.save_order_result(payload)
            elif item_type == "position_snapshot":
                symbol, snapshot = payload
                await repository.save_position_snapshot(symbol, snapshot)
            else:
                logger.warning("db_log_item_ignored", item_type=item_type)
        except Exception as exc:
            logger.exception("db_write_failed", item_type=item_type, error=str(exc))
        finally:
            db_log_queue.task_done()


def build_executor(settings: Settings) -> OrderExecutor:
    paper_executor = PaperExecutor(
        initial_quote_balance=settings.initial_quote_balance,
        symbol=settings.normalized_symbol,
    )

    if settings.trade_mode == TradeMode.PAPER:
        return paper_executor

    if settings.trade_mode == TradeMode.TESTNET:
        rest_client = BinanceRestClient(
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            testnet=True,
        )
        from app.execution.testnet_executor import TestnetExecutor

        return TestnetExecutor(rest_client=rest_client, fallback_snapshot=paper_executor.portfolio_snapshot())

    return LiveExecutor()


async def setup_database(settings: Settings) -> tuple[Database | None, TradingRepository | None]:
    if not settings.database_enabled:
        logger.info("db_disabled")
        return None, None

    db = Database(settings.database_url)
    await db.connect()
    ping = await db.ping()
    logger.info(
        "db_connected",
        database=ping["database_name"],
        user=ping["user_name"],
    )

    if settings.database_apply_migrations_on_start:
        migrations_dir = Path(__file__).parent / "storage" / "migrations"

        base_migration_path = migrations_dir / "001_init.sql"
        await db.apply_migration_file(base_migration_path)
        logger.info("db_migration_applied", migration=str(base_migration_path))

        market_features_migration_path = migrations_dir / "003_market_features.sql"
        await db.apply_migration_file(market_features_migration_path)
        logger.info("db_migration_applied", migration=str(market_features_migration_path))

        if settings.database_use_timescaledb:
            timescale_migration_path = migrations_dir / "002_timescale_optional.sql"
            await db.apply_migration_file(timescale_migration_path)
            logger.info("db_migration_applied", migration=str(timescale_migration_path))

    return db, TradingRepository(db)




async def _load_valid_candles_into_state(
    *,
    candles: list[Candle],
    settings: Settings,
    market_state: MarketState,
    source: str,
) -> bool:
    validation = validate_startup_candles(
        candles,
        timeframe=settings.timeframe,
        max_age_seconds=settings.startup_candle_max_age_seconds,
        gap_tolerance_seconds=settings.startup_candle_gap_tolerance_seconds,
    )

    if not validation.can_use:
        logger.warning(
            f"startup_candle_{source}_rejected",
            reason=validation.reason,
            loaded_count=validation.loaded_count,
            latest_close_time=(
                validation.latest_close_time.isoformat()
                if validation.latest_close_time is not None
                else None
            ),
        )
        return False

    loaded_count = market_state.load_historical_candles(candles)
    logger.info(
        f"startup_candle_{source}_loaded",
        loaded_count=loaded_count,
        latest_close_time=validation.latest_close_time.isoformat()
        if validation.latest_close_time is not None
        else None,
        latest_price=str(market_state.latest_price),
    )
    return True


async def warm_up_market_state_from_db(
    *,
    settings: Settings,
    repository: TradingRepository | None,
    market_state: MarketState,
) -> None:
    if not settings.load_recent_candles_on_start:
        logger.info("startup_candle_warmup_skipped", reason="disabled_by_config")
        return

    if repository is not None:
        db_candles = await repository.load_recent_candles(
            exchange=settings.exchange.value,
            symbol=settings.normalized_symbol,
            timeframe=settings.timeframe,
            limit=settings.startup_candle_limit,
        )

        db_loaded = await _load_valid_candles_into_state(
            candles=db_candles,
            settings=settings,
            market_state=market_state,
            source="warmup",
        )
        if db_loaded:
            return
    else:
        logger.info("startup_candle_warmup_skipped", reason="database_disabled")

    if not settings.startup_rest_backfill_enabled:
        logger.info("startup_candle_backfill_skipped", reason="disabled_by_config")
        return

    rest_client = BinanceRestClient(testnet=settings.binance_testnet)
    try:
        rest_candles = await rest_client.get_closed_candles(
            symbol=settings.normalized_symbol,
            timeframe=settings.timeframe,
            limit=settings.startup_rest_backfill_limit,
            exchange=settings.exchange.value,
        )
        logger.info(
            "startup_candle_backfill_fetched",
            fetched_count=len(rest_candles),
            symbol=settings.normalized_symbol,
            timeframe=settings.timeframe,
        )

        rest_loaded = await _load_valid_candles_into_state(
            candles=rest_candles,
            settings=settings,
            market_state=market_state,
            source="backfill",
        )

        if rest_loaded and repository is not None:
            await repository.save_candles(rest_candles)
            logger.info("startup_candle_backfill_saved", saved_count=len(rest_candles))
    except Exception as exc:
        logger.exception("startup_candle_backfill_failed", error=str(exc))
    finally:
        await rest_client.close()



async def restore_paper_position_from_db(
    *,
    settings: Settings,
    repository: TradingRepository | None,
    executor: OrderExecutor,
    market_state: MarketState,
) -> None:
    if settings.trade_mode != TradeMode.PAPER:
        logger.info("paper_position_restore_skipped", reason="not_paper_mode")
        return

    if not settings.load_paper_position_on_start:
        logger.info("paper_position_restore_skipped", reason="disabled_by_config")
        return

    if repository is None:
        logger.info("paper_position_restore_skipped", reason="database_disabled")
        return

    if not isinstance(executor, PaperExecutor):
        logger.warning("paper_position_restore_skipped", reason="executor_is_not_paper")
        return

    latest_price = Decimal(str(market_state.latest_price)) if market_state.latest_price is not None else None
    snapshot = await repository.load_position_snapshot(
        symbol=settings.normalized_symbol,
        fallback_quote_balance=settings.initial_quote_balance,
        latest_price=latest_price,
    )

    if snapshot is None:
        logger.info("paper_position_restore_skipped", reason="no_saved_position")
        return

    executor.restore_from_snapshot(snapshot)
    logger.info(
        "paper_position_restore_loaded",
        symbol=settings.normalized_symbol,
        quote_balance=str(snapshot.quote_balance),
        position_quantity=str(snapshot.position_quantity),
        position_avg_entry_price=str(snapshot.position_avg_entry_price),
        realized_pnl_today=str(snapshot.realized_pnl_today),
        latest_price=str(snapshot.latest_price),
        has_open_position=snapshot.has_open_position,
    )

async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "bot_starting",
        mode=settings.trade_mode.value,
        exchange=settings.exchange.value,
        symbol=settings.normalized_symbol,
        timeframe=settings.timeframe,
    )

    db, repository = await setup_database(settings)
    db_log_queue: asyncio.Queue[DbLogItem] | None = None
    if repository is not None:
        db_log_queue = asyncio.Queue(maxsize=5000)

    market_event_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(
        maxsize=settings.market_event_queue_size
    )
    signal_queue: asyncio.Queue[TradeSignal] = asyncio.Queue(maxsize=settings.signal_queue_size)
    approved_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.approved_order_queue_size)

    market_state = MarketState(symbol=settings.normalized_symbol)
    await warm_up_market_state_from_db(
        settings=settings,
        repository=repository,
        market_state=market_state,
    )

    strategy = EmaRsiStrategy(
        fast_period=settings.ema_fast_period,
        slow_period=settings.ema_slow_period,
        rsi_period=settings.rsi_period,
        rsi_buy_min=settings.rsi_buy_min,
        rsi_buy_max=settings.rsi_buy_max,
        rsi_sell_min=settings.rsi_sell_min,
        suggested_quote_amount=settings.max_order_usdt,
        trend_ema_period=settings.trend_ema_period,
        min_ema_gap_pct=settings.min_ema_gap_pct,
        atr_period=settings.atr_period,
        min_atr_pct=settings.min_atr_pct,
    )

    risk_manager = RiskManager(
        RiskConfig(
            max_order_usdt=settings.max_order_usdt,
            max_position_usdt=settings.max_position_usdt,
            max_daily_loss_usdt=settings.max_daily_loss_usdt,
            max_trades_per_hour=settings.max_trades_per_hour,
            cooldown_seconds=settings.cooldown_seconds,
            allow_only_one_open_position=settings.allow_only_one_open_position,
        )
    )

    executor = build_executor(settings)
    if hasattr(executor, "set_latest_price") and market_state.latest_price is not None:
        executor.set_latest_price(Decimal(str(market_state.latest_price)))  # type: ignore[attr-defined]

    await restore_paper_position_from_db(
        settings=settings,
        repository=repository,
        executor=executor,
        market_state=market_state,
    )

    stream = BinanceWebSocketStream(
        symbol=settings.normalized_symbol,
        timeframe=settings.timeframe,
        testnet=settings.binance_testnet,
        reconnect_seconds=settings.websocket_reconnect_seconds,
    )

    tasks = [
        asyncio.create_task(market_data_worker(stream, market_event_queue), name="market_data_worker"),
        asyncio.create_task(
            strategy_worker(
                market_event_queue,
                signal_queue,
                market_state,
                strategy,
                executor,
                settings.stop_loss_pct,
                settings.take_profit_pct,
                db_log_queue,
                paper_test_force_buy_on_first_candle=(
                    settings.paper_test_force_buy_on_first_candle
                    and settings.trade_mode == TradeMode.PAPER
                ),
                paper_test_force_buy_quote_amount=settings.paper_test_force_buy_quote_amount,
            ),
            name="strategy_worker",
        ),
        asyncio.create_task(
            risk_worker(signal_queue, approved_queue, risk_manager, executor, db_log_queue),
            name="risk_worker",
        ),
        asyncio.create_task(
            execution_worker(approved_queue, executor, risk_manager, db_log_queue),
            name="execution_worker",
        ),
        asyncio.create_task(
            health_worker(
                market_state,
                risk_manager,
                stale_seconds=settings.market_data_stale_seconds,
            ),
            name="health_worker",
        ),
    ]

    if db_log_queue is not None and repository is not None:
        tasks.append(
            asyncio.create_task(db_writer_worker(db_log_queue, repository), name="db_writer_worker")
        )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.warning("bot_cancelled")
        raise
    except KeyboardInterrupt:
        logger.warning("bot_stopped_by_keyboard")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if db is not None:
            await db.close()
            logger.info("db_closed")
        logger.info("bot_stopped")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("bot_stopped_by_keyboard")


if __name__ == "__main__":
    main()
