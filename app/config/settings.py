from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from functools import lru_cache

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradeMode(StrEnum):
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


class ExchangeName(StrEnum):
    BINANCE = "binance"


class HistoricalMarketDataSource(StrEnum):
    PRODUCTION = "production"
    TESTNET = "testnet"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"
    trade_mode: TradeMode = TradeMode.PAPER

    exchange: ExchangeName = ExchangeName.BINANCE
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    timeframe: str = "1m"

    binance_testnet: bool = True
    binance_api_key: str = ""
    binance_api_secret: str = ""

    enable_live_trading: bool = False

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_bot"
    postgres_user: str = "trader"
    postgres_password: str = "trader_password"
    database_enabled: bool = False
    database_apply_migrations_on_start: bool = False
    database_use_timescaledb: bool = False

    load_recent_candles_on_start: bool = True
    startup_candle_limit: int = Field(default=100, ge=0, le=5000)
    startup_candle_max_age_seconds: int = Field(default=180, ge=10)
    startup_candle_gap_tolerance_seconds: int = Field(default=2, ge=0)
    startup_rest_backfill_enabled: bool = True
    startup_rest_backfill_limit: int = Field(default=100, ge=1, le=1000)

    initial_quote_balance: Decimal = Field(default=Decimal("1000"), ge=Decimal("0"))
    max_order_usdt: Decimal = Field(default=Decimal("10"), gt=Decimal("0"))
    max_position_usdt: Decimal = Field(default=Decimal("50"), gt=Decimal("0"))
    max_daily_loss_usdt: Decimal = Field(default=Decimal("10"), gt=Decimal("0"))
    max_trades_per_hour: int = Field(default=5, gt=0)
    stop_loss_pct: Decimal = Field(default=Decimal("0.7"), gt=Decimal("0"))
    take_profit_pct: Decimal = Field(default=Decimal("1.2"), gt=Decimal("0"))
    cooldown_seconds: int = Field(default=60, ge=0)
    allow_only_one_open_position: bool = True
    load_paper_position_on_start: bool = True

    # Paper-only integration testing switch. This is intentionally blocked
    # outside paper mode by the safety validator.
    paper_test_force_buy_on_first_candle: bool = False
    paper_test_force_buy_quote_amount: Decimal | None = None

    ema_fast_period: int = Field(default=9, gt=1)
    ema_slow_period: int = Field(default=21, gt=2)
    rsi_period: int = Field(default=14, gt=1)
    rsi_buy_min: float = Field(default=45.0, ge=0, le=100)
    rsi_buy_max: float = Field(default=70.0, ge=0, le=100)
    rsi_sell_min: float = Field(default=75.0, ge=0, le=100)

    # V17 optional buy filters. Keep disabled by default so existing behavior
    # remains predictable unless a backtest/optimizer explicitly enables them.
    trend_ema_period: int | None = Field(default=None, ge=0)
    min_ema_gap_pct: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    atr_period: int | None = Field(default=None, ge=0)
    min_atr_pct: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

    # Historical/backtest data can use production public Binance candles while
    # runtime execution remains paper/testnet-safe. Binance Spot Testnet candles
    # are fake and can contain unrealistic historical prices, so production is
    # the default for strategy research.
    historical_market_data_source: HistoricalMarketDataSource = HistoricalMarketDataSource.PRODUCTION

    backtest_fee_rate_pct: Decimal = Field(default=Decimal("0.1"), ge=Decimal("0"))
    backtest_slippage_pct: Decimal = Field(default=Decimal("0.02"), ge=Decimal("0"))

    market_event_queue_size: int = Field(default=2000, gt=0)
    signal_queue_size: int = Field(default=500, gt=0)
    approved_order_queue_size: int = Field(default=500, gt=0)
    websocket_reconnect_seconds: int = Field(default=5, ge=1)
    market_data_stale_seconds: int = Field(default=120, ge=10)

    @model_validator(mode="after")
    def validate_trading_safety(self) -> Settings:
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("EMA_FAST_PERIOD must be smaller than EMA_SLOW_PERIOD")

        if self.rsi_buy_min >= self.rsi_buy_max:
            raise ValueError("RSI_BUY_MIN must be smaller than RSI_BUY_MAX")

        if self.trend_ema_period == 0:
            self.trend_ema_period = None
        if self.atr_period == 0:
            self.atr_period = None

        if self.trend_ema_period is not None and self.trend_ema_period <= self.ema_slow_period:
            raise ValueError("TREND_EMA_PERIOD must be greater than EMA_SLOW_PERIOD when set")

        if self.atr_period is None and self.min_atr_pct > 0:
            raise ValueError("ATR_PERIOD is required when MIN_ATR_PCT is greater than zero")

        if self.paper_test_force_buy_on_first_candle and self.trade_mode != TradeMode.PAPER:
            raise ValueError(
                "PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE can only be enabled in paper mode"
            )

        if (
            self.paper_test_force_buy_quote_amount is not None
            and self.paper_test_force_buy_quote_amount <= 0
        ):
            raise ValueError("PAPER_TEST_FORCE_BUY_QUOTE_AMOUNT must be positive when set")

        if self.trade_mode == TradeMode.LIVE and not self.enable_live_trading:
            raise ValueError(
                "LIVE trading requested, but ENABLE_LIVE_TRADING=false. "
                "This safety guard prevents accidental real orders."
            )

        if (
            self.trade_mode in {TradeMode.TESTNET, TradeMode.LIVE}
            and (not self.binance_api_key or not self.binance_api_secret)
        ):
            raise ValueError("API key and secret are required for testnet/live trading modes")

        return self

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def normalized_symbol(self) -> str:
        return self.symbol.upper().strip()

    @property
    def historical_market_data_uses_testnet(self) -> bool:
        return self.historical_market_data_source == HistoricalMarketDataSource.TESTNET

    @property
    def historical_exchange_id(self) -> str:
        if self.historical_market_data_uses_testnet:
            return "binance_testnet"
        return "binance_spot"

    @property
    def binance_stream_symbol(self) -> str:
        return self.normalized_symbol.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError:
        raise
