from app.config.settings import HistoricalMarketDataSource, Settings
from scripts.backtest_strategy import _resolve_market_data_source


def test_settings_default_historical_market_data_source_is_production():
    settings = Settings()

    assert settings.historical_market_data_source == HistoricalMarketDataSource.PRODUCTION
    assert settings.historical_market_data_uses_testnet is False
    assert settings.historical_exchange_id == "binance_spot"


def test_settings_testnet_historical_exchange_id():
    settings = Settings(historical_market_data_source=HistoricalMarketDataSource.TESTNET)

    assert settings.historical_market_data_uses_testnet is True
    assert settings.historical_exchange_id == "binance_testnet"


def test_backtest_source_override_resolves_exchange_key():
    source, use_testnet, exchange_id = _resolve_market_data_source("testnet")

    assert source == "testnet"
    assert use_testnet is True
    assert exchange_id == "binance_testnet"
