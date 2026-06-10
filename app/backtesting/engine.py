from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.execution.models import PortfolioSnapshot
from app.market.models import Candle
from app.market.state import MarketState
from app.risk.position_guard import build_position_exit_signal
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal


@dataclass(slots=True)
class _OpenBacktestPosition:
    entry_time: object
    entry_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    entry_reason: str


@dataclass(slots=True)
class BacktestEngine:
    strategy: Strategy
    symbol: str
    initial_quote_balance: Decimal
    max_order_usdt: Decimal
    max_position_usdt: Decimal
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    allow_only_one_open_position: bool = True
    fee_rate_pct: Decimal = Decimal("0")
    slippage_pct: Decimal = Decimal("0")

    def run(self, candles: list[Candle]) -> BacktestResult:
        """Replay strategy over candles with a simple paper-trading simulator.

        V13 includes fee and slippage simulation. It is still not a perfect
        exchange simulator, because apparently reality insists on being more
        annoying than unit tests, but it is much closer than the optimistic
        no-fee fantasy backtest.
        """
        sorted_candles = sorted(candles, key=lambda item: item.open_time)
        state = MarketState(symbol=self.symbol)

        quote_balance = self.initial_quote_balance
        position_quantity = Decimal("0")
        avg_entry_price = Decimal("0")
        realized_pnl = Decimal("0")
        total_fees = Decimal("0")
        last_price: Decimal | None = None
        open_position: _OpenBacktestPosition | None = None

        executed_orders = 0
        trades: list[BacktestTrade] = []
        equity_peak = self.initial_quote_balance
        max_drawdown = Decimal("0")

        for candle in sorted_candles:
            if not state.add_candle(candle):
                continue

            last_price = Decimal(str(candle.close))
            portfolio = PortfolioSnapshot(
                quote_balance=quote_balance,
                position_quantity=position_quantity,
                position_avg_entry_price=avg_entry_price,
                realized_pnl_today=realized_pnl,
                latest_price=last_price,
            )

            signal = self._next_signal(state=state, portfolio=portfolio)

            if signal.side == SignalSide.BUY and position_quantity <= 0:
                quote_amount = signal.suggested_quote_amount or self.max_order_usdt
                quote_amount = min(quote_amount, self.max_order_usdt, quote_balance)

                if quote_amount > 0 and quote_amount <= self.max_position_usdt:
                    execution_price = self._apply_buy_slippage(last_price)
                    entry_fee = self._fee_for(quote_amount)
                    total_cost = quote_amount + entry_fee

                    if total_cost <= quote_balance:
                        quantity = quote_amount / execution_price
                        position_quantity = quantity
                        avg_entry_price = execution_price
                        quote_balance -= total_cost
                        realized_pnl -= entry_fee
                        total_fees += entry_fee
                        executed_orders += 1
                        open_position = _OpenBacktestPosition(
                            entry_time=candle.close_time,
                            entry_price=execution_price,
                            quantity=quantity,
                            quote_amount=quote_amount,
                            entry_fee=entry_fee,
                            entry_reason=signal.reason,
                        )

            elif signal.side == SignalSide.SELL and position_quantity > 0:
                quantity = position_quantity
                execution_price = self._apply_sell_slippage(last_price)
                gross_quote_received = quantity * execution_price
                exit_fee = self._fee_for(gross_quote_received)
                net_quote_received = gross_quote_received - exit_fee
                cost_basis = quantity * avg_entry_price
                entry_fee = open_position.entry_fee if open_position is not None else Decimal("0")
                pnl = net_quote_received - cost_basis - entry_fee

                quote_balance += net_quote_received
                realized_pnl += gross_quote_received - cost_basis - exit_fee
                position_quantity = Decimal("0")
                avg_entry_price = Decimal("0")
                total_fees += exit_fee
                executed_orders += 1

                if open_position is not None:
                    trades.append(
                        BacktestTrade(
                            symbol=self.symbol,
                            entry_time=open_position.entry_time,  # type: ignore[arg-type]
                            exit_time=candle.close_time,
                            entry_price=open_position.entry_price,
                            exit_price=execution_price,
                            quantity=quantity,
                            quote_amount=open_position.quote_amount,
                            entry_fee=open_position.entry_fee,
                            exit_fee=exit_fee,
                            pnl=pnl,
                            entry_reason=open_position.entry_reason,
                            exit_reason=signal.reason,
                        )
                    )
                    open_position = None

            current_equity = quote_balance + (position_quantity * last_price)
            if current_equity > equity_peak:
                equity_peak = current_equity
            drawdown = equity_peak - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        unrealized_pnl = Decimal("0")
        if position_quantity > 0 and last_price is not None:
            unrealized_pnl = (last_price - avg_entry_price) * position_quantity

        final_equity = quote_balance + (position_quantity * last_price if last_price is not None else Decimal("0"))
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.pnl < 0)

        metrics = BacktestMetrics(
            candles_processed=len(sorted_candles),
            executed_orders=executed_orders,
            round_trips=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_fees=total_fees,
            max_drawdown=max_drawdown,
            initial_equity=self.initial_quote_balance,
            final_equity=final_equity,
            open_position_quantity=position_quantity,
            open_position_avg_entry_price=avg_entry_price,
            last_price=last_price,
        )
        return BacktestResult(metrics=metrics, trades=trades)

    def _next_signal(self, *, state: MarketState, portfolio: PortfolioSnapshot) -> TradeSignal:
        exit_signal = build_position_exit_signal(
            portfolio=portfolio,
            symbol=self.symbol,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
        )
        if exit_signal is not None:
            return exit_signal

        signal = self.strategy.on_market_state(state)
        if signal.side == SignalSide.BUY and self.allow_only_one_open_position and portfolio.has_open_position:
            return TradeSignal.hold(
                strategy_name=signal.strategy_name,
                symbol=signal.symbol,
                reason="backtest_position_already_open",
            )
        return signal

    def _fee_for(self, notional: Decimal) -> Decimal:
        if self.fee_rate_pct <= 0:
            return Decimal("0")
        return notional * (self.fee_rate_pct / Decimal("100"))

    def _apply_buy_slippage(self, price: Decimal) -> Decimal:
        if self.slippage_pct <= 0:
            return price
        return price * (Decimal("1") + (self.slippage_pct / Decimal("100")))

    def _apply_sell_slippage(self, price: Decimal) -> Decimal:
        if self.slippage_pct <= 0:
            return price
        return price * (Decimal("1") - (self.slippage_pct / Decimal("100")))
