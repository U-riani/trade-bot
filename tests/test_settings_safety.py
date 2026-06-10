from decimal import Decimal

import pytest

from app.config.settings import Settings, TradeMode


def test_paper_force_buy_rejected_outside_paper_mode():
    with pytest.raises(ValueError, match="PAPER_TEST_FORCE_BUY_ON_FIRST_CANDLE"):
        Settings(
            trade_mode=TradeMode.TESTNET,
            binance_api_key="key",
            binance_api_secret="secret",
            paper_test_force_buy_on_first_candle=True,
        )


def test_paper_force_buy_quote_amount_must_be_positive():
    with pytest.raises(ValueError, match="PAPER_TEST_FORCE_BUY_QUOTE_AMOUNT"):
        Settings(
            trade_mode=TradeMode.PAPER,
            paper_test_force_buy_quote_amount=Decimal("0"),
        )


def test_zero_filter_values_disable_v17_filters():
    settings = Settings(trend_ema_period=0, atr_period=0, min_atr_pct=Decimal("0"))

    assert settings.trend_ema_period is None
    assert settings.atr_period is None


def test_min_atr_requires_atr_period():
    with pytest.raises(ValueError, match="ATR_PERIOD"):
        Settings(atr_period=0, min_atr_pct=Decimal("0.08"))
