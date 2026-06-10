from __future__ import annotations

import hashlib
import hmac
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config.logging import get_logger
from app.execution.models import OrderRequest, OrderResult, OrderSide, OrderStatus
from app.market.models import Candle
from app.utils.ids import new_id
from app.utils.time import ms_to_datetime, utc_now

logger = get_logger(__name__)


class BinanceRestClient:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.testnet = testnet
        self.timeout_seconds = timeout_seconds
        self.base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com/api"
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def server_time(self) -> int:
        response = await self.client.get(f"{self.base_url}/v3/time")
        response.raise_for_status()
        return int(response.json()["serverTime"])

    async def get_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[Any]]:
        """Fetch raw Binance kline/candlestick data.

        This endpoint is public and does not require API keys. Binance identifies
        klines by their open time, which also matches our DB uniqueness rule.
        `startTime` and `endTime` let us page through history for larger
        backtests instead of pretending 274 candles is a strategy sample.
        """
        if limit <= 0:
            return []

        safe_limit = min(limit, 1000)
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": safe_limit,
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        response = await self.client.get(
            f"{self.base_url}/v3/klines",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected Binance kline response: {data!r}")
        return data

    async def get_historical_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[list[Any]]:
        """Page raw klines backwards and return up to `limit` rows oldest->newest.

        Unlike get_historical_closed_candles this returns the RAW Binance rows so
        callers can read taker-buy / quote-volume fields (indexes 7, 9, 10) that
        the Candle model does not carry. The currently-forming candle may be
        included; downstream bucket alignment drops incomplete buckets anyway.
        """
        if limit <= 0:
            return []

        remaining = limit
        end_time_ms: int | None = None
        by_open_time: dict[int, list[Any]] = {}

        while remaining > 0:
            request_limit = min(remaining + 1, 1000)
            raw_klines = await self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=request_limit,
                end_time_ms=end_time_ms,
            )
            if not raw_klines:
                break

            for item in raw_klines:
                by_open_time[int(item[0])] = item

            oldest_open_time_ms = int(raw_klines[0][0])
            next_end_time_ms = oldest_open_time_ms - 1
            if end_time_ms is not None and next_end_time_ms >= end_time_ms:
                break

            end_time_ms = next_end_time_ms
            remaining = limit - len(by_open_time)

        ordered = [by_open_time[key] for key in sorted(by_open_time)]
        return ordered[-limit:]

    async def get_order_book(self, *, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Fetch the CURRENT order book depth snapshot.

        FORWARD-ONLY: Binance serves only the live book, never historical books.
        The result must not be attached to past candles.
        """
        response = await self.client.get(
            f"{self.base_url}/v3/depth",
            params={"symbol": symbol.upper(), "limit": limit},
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Binance depth response: {data!r}")
        return data

    async def get_closed_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        exchange: str = "binance",
    ) -> list[Candle]:
        """Fetch recent closed candles from Binance REST.

        Binance may include the currently forming candle. We only return candles
        whose close time is not in the future/current open window, so strategy
        warm-up uses completed market data only.
        """
        raw_klines = await self.get_klines(symbol=symbol, interval=timeframe, limit=limit + 1)
        candles = self._parse_closed_candles(
            raw_klines,
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
        )
        return candles[-limit:]

    async def get_historical_closed_candles(
        self,
        *,
        symbol: str,
        timeframe: str,
        limit: int,
        exchange: str = "binance",
    ) -> list[Candle]:
        """Fetch up to `limit` recent closed candles by paging REST history backwards.

        Binance limits one kline request to 1000 rows. This method repeatedly
        requests older windows using `endTime`, filters the still-open candle,
        de-duplicates by open time, and returns candles sorted oldest -> newest.
        """
        if limit <= 0:
            return []

        remaining = limit
        end_time_ms: int | None = None
        candles_by_open_time: dict[int, Candle] = {}

        while remaining > 0:
            request_limit = min(remaining + 1, 1000)
            raw_klines = await self.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=request_limit,
                end_time_ms=end_time_ms,
            )
            if not raw_klines:
                break

            page_candles = self._parse_closed_candles(
                raw_klines,
                symbol=symbol,
                timeframe=timeframe,
                exchange=exchange,
            )

            for candle in page_candles:
                candles_by_open_time[int(candle.open_time.timestamp() * 1000)] = candle

            oldest_open_time_ms = int(ms_to_datetime(raw_klines[0][0]).timestamp() * 1000)
            next_end_time_ms = oldest_open_time_ms - 1
            if end_time_ms is not None and next_end_time_ms >= end_time_ms:
                logger.warning(
                    "binance_historical_backfill_stopped_no_progress",
                    symbol=symbol,
                    timeframe=timeframe,
                    end_time_ms=end_time_ms,
                    next_end_time_ms=next_end_time_ms,
                )
                break

            end_time_ms = next_end_time_ms
            remaining = limit - len(candles_by_open_time)

        candles = sorted(candles_by_open_time.values(), key=lambda candle: candle.open_time)
        return candles[-limit:]

    def _parse_closed_candles(
        self,
        raw_klines: list[list[Any]],
        *,
        symbol: str,
        timeframe: str,
        exchange: str,
    ) -> list[Candle]:
        now = utc_now()
        candles: list[Candle] = []

        for item in raw_klines:
            if len(item) < 7:
                logger.warning("binance_kline_ignored_invalid_shape", item=item)
                continue

            open_time = ms_to_datetime(item[0])
            close_time = ms_to_datetime(item[6])

            if close_time > now:
                continue

            candles.append(
                Candle(
                    exchange=exchange,
                    symbol=symbol.upper(),
                    timeframe=timeframe,
                    open_time=open_time,
                    close_time=close_time,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    is_closed=True,
                )
            )

        return candles

    async def place_market_order(self, request: OrderRequest) -> OrderResult:
        if request.quote_amount is None and request.quantity is None:
            raise ValueError("Market order requires quote_amount or quantity")

        client_order_id = request.client_order_id or new_id("order")

        params: dict[str, Any] = {
            "symbol": request.symbol,
            "side": request.side.value.upper(),
            "type": "MARKET",
            "newClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000),
        }

        if request.side == OrderSide.BUY:
            if request.quote_amount is None:
                raise ValueError("BUY market order requires quote_amount for this MVP")
            params["quoteOrderQty"] = self._format_decimal(request.quote_amount)
        else:
            if request.quantity is None:
                raise ValueError("SELL market order requires quantity for this MVP")
            params["quantity"] = self._format_decimal(request.quantity)

        signed_params = self._signed_params(params)
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            response = await self.client.post(
                f"{self.base_url}/v3/order",
                params=signed_params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            logger.info("binance_order_placed", client_order_id=client_order_id, response=data)
            return OrderResult(
                client_order_id=client_order_id,
                exchange_order_id=str(data.get("orderId", "")),
                symbol=request.symbol,
                side=request.side,
                status=OrderStatus.FILLED if data.get("status") == "FILLED" else OrderStatus.NEW,
                executed_quantity=Decimal(str(data.get("executedQty", "0"))),
                executed_quote_quantity=Decimal(str(data.get("cummulativeQuoteQty", "0"))),
                raw_response=data,
            )
        except httpx.TimeoutException:
            logger.exception("binance_order_timeout", client_order_id=client_order_id)
            return OrderResult(
                client_order_id=client_order_id,
                exchange_order_id=None,
                symbol=request.symbol,
                side=request.side,
                status=OrderStatus.UNKNOWN,
                executed_quantity=Decimal("0"),
                executed_quote_quantity=Decimal("0"),
                raw_response={"error": "timeout"},
            )

    async def get_order_status(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        params = {
            "symbol": symbol,
            "origClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000),
        }
        headers = {"X-MBX-APIKEY": self.api_key}
        response = await self.client.get(
            f"{self.base_url}/v3/order",
            params=self._signed_params(params),
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def _signed_params(self, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(params)
        signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        return {**params, "signature": signature}

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        return format(value.normalize(), "f")
