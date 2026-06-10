from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import orjson

try:  # websockets >= 14
    from websockets.asyncio.client import connect
except ImportError:  # websockets 13 compatibility
    from websockets import connect  # type: ignore[no-redef]

from app.config.logging import get_logger
from app.exchange.base import MarketDataStream
from app.market.models import Candle, MarketEvent, MarketEventType
from app.utils.time import ms_to_datetime

logger = get_logger(__name__)


class BinanceWebSocketStream(MarketDataStream):
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1m",
        testnet: bool = True,
        reconnect_seconds: int = 5,
    ) -> None:
        self.symbol = symbol.upper()
        self.stream_symbol = symbol.lower()
        self.timeframe = timeframe
        self.testnet = testnet
        self.reconnect_seconds = reconnect_seconds

    @property
    def base_url(self) -> str:
        if self.testnet:
            return "wss://stream.testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"

    @property
    def stream_url(self) -> str:
        return f"{self.base_url}/{self.stream_symbol}@kline_{self.timeframe}"

    async def stream(self) -> AsyncIterator[MarketEvent]:
        while True:
            try:
                logger.info(
                    "binance_ws_connecting",
                    url=self.stream_url,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    testnet=self.testnet,
                )
                async with connect(
                    self.stream_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=1024,
                ) as websocket:
                    logger.info("binance_ws_connected", symbol=self.symbol)
                    async for raw_message in websocket:
                        event = self._parse_message(raw_message)
                        if event is not None:
                            yield event
            except asyncio.CancelledError:
                logger.warning("binance_ws_cancelled", symbol=self.symbol)
                raise
            except Exception as exc:  # noqa: BLE001 - keep market worker alive
                logger.exception(
                    "binance_ws_error",
                    symbol=self.symbol,
                    error=str(exc),
                    reconnect_seconds=self.reconnect_seconds,
                )
                await asyncio.sleep(self.reconnect_seconds)

    def _parse_message(self, raw_message: str | bytes) -> MarketEvent | None:
        payload: dict[str, Any] = orjson.loads(raw_message)

        if payload.get("e") != "kline":
            return None

        kline = payload.get("k", {})
        is_closed = bool(kline.get("x", False))
        if not is_closed:
            return None

        candle = Candle(
            exchange="binance",
            symbol=str(kline.get("s", self.symbol)).upper(),
            timeframe=str(kline.get("i", self.timeframe)),
            open_time=ms_to_datetime(kline["t"]),
            close_time=ms_to_datetime(kline["T"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            is_closed=True,
        )
        return MarketEvent(type=MarketEventType.CANDLE_CLOSED, candle=candle)
