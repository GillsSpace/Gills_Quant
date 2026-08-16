import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from strategies.execution_bridge import (
    TargetPosition,
    StrategyOutput,
    Position,
    PortfolioState,
    OrderType,
    OrderInstruction,
    OrderSession,
    OrderDuration,
    OrderStatus,
    ExecutionMode,
    SchwabOrderGenerator,
    SchwabLiveEngine,
    DataContext,
    ExecutionBridge
)


class TestExecutionBridge(unittest.TestCase):

    def setUp(self):
        self.engine = SchwabLiveEngine(mode=ExecutionMode.PAPER)
        self.context = DataContext()
        self.context.update_quote_tick("AAPL", {"mark": 150.0, "lastPrice": 150.0})
        self.context.update_quote_tick("MSFT", {"mark": 300.0, "lastPrice": 300.0})
        self.bridge = ExecutionBridge(engine=self.engine, context=self.context)

    def tearDown(self):
        self.context.close()

    def test_target_position_validation(self):
        # Valid Market Target
        tp = TargetPosition(symbol="AAPL", target_type="WEIGHT", target_value=0.10)
        tp.validate()

        # Invalid Limit Target (missing price)
        tp_invalid = TargetPosition(symbol="AAPL", target_type="SHARES", target_value=10, order_type=OrderType.LIMIT)
        with self.assertRaises(ValueError):
            tp_invalid.validate()

    def test_order_generator_delta_calculation(self):
        portfolio = PortfolioState(
            account_id="TEST",
            cash_balance=100000.0,
            liquidation_value=100000.0,
            buying_power=200000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            positions={
                "AAPL": Position(symbol="AAPL", quantity=100, average_price=140.0, market_value=15000.0, unrealized_pnl=1000.0)
            }
        )
        latest_prices = {"AAPL": 150.0, "MSFT": 300.0}

        # Target: Increase AAPL to 200 shares
        tp = TargetPosition(symbol="AAPL", target_type="SHARES", target_value=200)
        requests = SchwabOrderGenerator.calculate_order_requests(tp, portfolio, latest_prices)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].instruction, OrderInstruction.BUY)
        self.assertEqual(requests[0].quantity, 100)

        # Target: Reduce AAPL to 50 shares
        tp_reduce = TargetPosition(symbol="AAPL", target_type="SHARES", target_value=50)
        requests_reduce = SchwabOrderGenerator.calculate_order_requests(tp_reduce, portfolio, latest_prices)
        self.assertEqual(len(requests_reduce), 1)
        self.assertEqual(requests_reduce[0].instruction, OrderInstruction.SELL)
        self.assertEqual(requests_reduce[0].quantity, 50)

    def test_position_reversal_two_step(self):
        # Long 100 shares -> Target Short 50 shares
        portfolio = PortfolioState(
            account_id="TEST",
            cash_balance=100000.0,
            liquidation_value=100000.0,
            buying_power=200000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            positions={
                "AAPL": Position(symbol="AAPL", quantity=100, average_price=140.0, market_value=15000.0, unrealized_pnl=1000.0)
            }
        )
        latest_prices = {"AAPL": 150.0}
        tp_flip = TargetPosition(symbol="AAPL", target_type="SHARES", target_value=-50)
        requests = SchwabOrderGenerator.calculate_order_requests(tp_flip, portfolio, latest_prices)
        
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].instruction, OrderInstruction.SELL)
        self.assertEqual(requests[0].quantity, 100)
        self.assertEqual(requests[1].instruction, OrderInstruction.SELL_SHORT)
        self.assertEqual(requests[1].quantity, 50)

    def test_bracket_order_payload_structure(self):
        tp_bracket = TargetPosition(
            symbol="AAPL",
            target_type="SHARES",
            target_value=100,
            order_type=OrderType.BRACKET,
            limit_price=150.0,
            take_profit_price=165.0,
            stop_loss_price=142.0
        )
        portfolio = PortfolioState("TEST", 100000, 100000, 200000, 0, 0)
        latest_prices = {"AAPL": 150.0}
        requests = SchwabOrderGenerator.calculate_order_requests(tp_bracket, portfolio, latest_prices)
        self.assertEqual(len(requests), 1)

        payload = requests[0].schwab_payload
        self.assertEqual(payload["orderStrategyType"], "TRIGGERED")
        self.assertIn("childOrderStrategies", payload)
        self.assertEqual(payload["childOrderStrategies"][0]["orderStrategyType"], "OCO")

    def test_paper_execution_flow(self):
        strategy_output = StrategyOutput(
            strategy_name="TestStrategy",
            timestamp=datetime.now(timezone.utc),
            target_positions=[
                TargetPosition(symbol="MSFT", target_type="SHARES", target_value=50)
            ]
        )
        results = self.bridge.execute_strategy(strategy_output)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, OrderStatus.FILLED)
        self.assertEqual(results[0].filled_qty, 50)

        # Check paper portfolio hydration
        hydrated_portfolio = self.engine.hydrate_portfolio_state()
        self.assertEqual(hydrated_portfolio.get_position_shares("MSFT"), 50)


if __name__ == "__main__":
    unittest.main()
