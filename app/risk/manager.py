from __future__ import annotations

from datetime import datetime, timedelta

from app.config.logging import get_logger
from app.execution.models import OrderRequest, OrderSide, PortfolioSnapshot
from app.risk.models import RiskConfig, RiskDecision, RiskRejectReason
from app.risk.rules import TradeLimiter
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now

logger = get_logger(__name__)


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.kill_switch_active = False
        self.last_trade_at: datetime | None = None
        self.trade_limiter = TradeLimiter(max_trades_per_hour=config.max_trades_per_hour)

    def activate_kill_switch(self) -> None:
        if self.kill_switch_active:
            return
        self.kill_switch_active = True
        logger.critical("risk_kill_switch_activated")

    def deactivate_kill_switch(self) -> None:
        self.kill_switch_active = False
        logger.warning("risk_kill_switch_deactivated")

    def evaluate(self, signal: TradeSignal, portfolio: PortfolioSnapshot) -> RiskDecision:
        if self.kill_switch_active:
            return self._reject(RiskRejectReason.KILL_SWITCH_ACTIVE, signal)

        if signal.side == SignalSide.HOLD:
            return self._reject(RiskRejectReason.HOLD_SIGNAL, signal)

        if signal.side == SignalSide.BUY:
            if portfolio.realized_pnl_today <= -self.config.max_daily_loss_usdt:
                return self._reject(RiskRejectReason.DAILY_LOSS_LIMIT, signal)

            if not self._cooldown_passed():
                return self._reject(RiskRejectReason.COOLDOWN, signal)

            if not self.trade_limiter.can_trade():
                return self._reject(RiskRejectReason.MAX_TRADES_PER_HOUR, signal)

            return self._evaluate_buy(signal, portfolio)

        if signal.side == SignalSide.SELL:
            # Exits must not be blocked by entry-oriented throttles such as cooldown,
            # max trades per hour, or daily-loss limit. If a position is open, SELL is
            # allowed so stop-loss/take-profit can close risk immediately.
            return self._evaluate_sell(signal, portfolio)

        return self._reject(RiskRejectReason.INVALID_SIGNAL_SIDE, signal)

    def register_executed_trade(self) -> None:
        self.last_trade_at = utc_now()
        self.trade_limiter.register_trade()

    def _evaluate_buy(self, signal: TradeSignal, portfolio: PortfolioSnapshot) -> RiskDecision:
        if portfolio.latest_price is None:
            return self._reject(RiskRejectReason.LATEST_PRICE_MISSING, signal)

        if self.config.allow_only_one_open_position and portfolio.has_open_position:
            return self._reject(RiskRejectReason.POSITION_ALREADY_OPEN, signal)

        quote_amount = signal.suggested_quote_amount or self.config.max_order_usdt
        quote_amount = min(quote_amount, self.config.max_order_usdt)

        if quote_amount <= 0:
            return self._reject(RiskRejectReason.MAX_ORDER_SIZE, signal)

        if quote_amount > self.config.max_order_usdt:
            return self._reject(RiskRejectReason.MAX_ORDER_SIZE, signal)

        if quote_amount > portfolio.quote_balance:
            return self._reject(RiskRejectReason.INSUFFICIENT_BALANCE, signal)

        new_position_value = portfolio.position_value + quote_amount
        if new_position_value > self.config.max_position_usdt:
            return self._reject(RiskRejectReason.MAX_POSITION_SIZE, signal)

        order_request = OrderRequest(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            quote_amount=quote_amount,
            reason=signal.reason,
        )
        return RiskDecision.approve("buy_signal_approved", order_request)

    def _evaluate_sell(self, signal: TradeSignal, portfolio: PortfolioSnapshot) -> RiskDecision:
        if not portfolio.has_open_position:
            return self._reject(RiskRejectReason.NO_OPEN_POSITION, signal)

        order_request = OrderRequest(
            symbol=signal.symbol,
            side=OrderSide.SELL,
            quantity=portfolio.position_quantity,
            reason=signal.reason,
        )
        return RiskDecision.approve("sell_signal_approved", order_request)

    def _cooldown_passed(self) -> bool:
        if self.last_trade_at is None:
            return True
        return utc_now() - self.last_trade_at >= timedelta(seconds=self.config.cooldown_seconds)

    def _reject(self, reason: RiskRejectReason, signal: TradeSignal) -> RiskDecision:
        logger.info(
            "risk_rejected",
            reason=reason.value,
            symbol=signal.symbol,
            side=signal.side.value,
            strategy=signal.strategy_name,
            signal_reason=signal.reason,
        )
        return RiskDecision.reject(reason)
