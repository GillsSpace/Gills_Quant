import unittest
from datetime import datetime
from strategies.base import DataContext, TargetPosition, StrategyOutput, PositionUnit, OrderType, TimeHorizon, BaseStrategy
from strategies.execution_bridge import SchwabExecutionManager, ExecutionMode
from strategies.live_execution_loop import LiveMarketLoopRunner

class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("dummy_strat", symbols=["AAPL", "MSFT"])

    def predict(self, ctx: DataContext, portfolio=None) -> StrategyOutput:
        targets = {
            "AAPL": TargetPosition(
                symbol="AAPL",
                target_value=0.05,
                unit=PositionUnit.WEIGHT,
                order_type=OrderType.MARKET,
                predict_to=TimeHorizon.M5
            )
        }
        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(),
            target_positions=[
                TargetPosition(
                    symbol="AAPL",
                    target_value=0.05,
                    unit=PositionUnit.WEIGHT,
                    order_type=OrderType.MARKET,
                    predict_to=TimeHorizon.FIVE_MIN
                )
            ]
        )

class TestLiveExecutionLoop(unittest.TestCase):
    def test_run_5m_tick(self):
        exec_mgr = SchwabExecutionManager(mode=ExecutionMode.PAPER)
        strategy = DummyStrategy()
        runner = LiveMarketLoopRunner(exec_mgr, [strategy])

        import polars as pl
        from strategies.engines import BacktestDataContext
        dummy_df = pl.DataFrame({
            "timestamp": [datetime(2026, 8, 12, 10, 0)],
            "symbol": ["AAPL"],
            "mark": [150.0],
            "close": [150.0]
        })
        ctx = BacktestDataContext(
            current_timestamp=datetime(2026, 8, 12, 10, 0),
            historical_df=dummy_df,
            active_symbols=["AAPL", "MSFT"]
        )

        result = runner.run_5m_tick("2026-08-12", "10:00", ctx)
        self.assertEqual(result["timestamp"], "2026-08-12 10:00")
        self.assertEqual(result["strategies_run"], 1)
        self.assertGreaterEqual(result["orders_placed"], 0)

if __name__ == "__main__":
    unittest.main()
