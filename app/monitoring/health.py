from __future__ import annotations

import asyncio

from app.config.logging import get_logger
from app.market.state import MarketState
from app.risk.manager import RiskManager
from app.utils.time import utc_now

logger = get_logger(__name__)


async def health_worker(
    market_state: MarketState,
    risk_manager: RiskManager,
    stale_seconds: int,
    interval_seconds: int = 10,
    startup_grace_seconds: int | None = None,
) -> None:
    """Monitor market-data freshness.

    The first kline event may arrive only after the current candle closes.
    Without a startup grace period, the bot can incorrectly activate the
    kill switch before the first candle is received.
    """

    started_at = utc_now()
    grace_seconds = startup_grace_seconds or stale_seconds
    first_event_wait_logged = False

    while True:
        await asyncio.sleep(interval_seconds)

        if market_state.last_market_event_at is None:
            waiting_seconds = (utc_now() - started_at).total_seconds()

            if waiting_seconds <= grace_seconds:
                if not first_event_wait_logged:
                    logger.info(
                        "market_data_waiting_first_event",
                        startup_grace_seconds=grace_seconds,
                    )
                    first_event_wait_logged = True
                continue

            logger.warning(
                "market_data_no_initial_event",
                waiting_seconds=round(waiting_seconds, 2),
                startup_grace_seconds=grace_seconds,
            )
            risk_manager.activate_kill_switch()
            continue

        if market_state.is_stale(stale_seconds):
            logger.warning("market_data_stale", stale_seconds=stale_seconds)
            risk_manager.activate_kill_switch()
