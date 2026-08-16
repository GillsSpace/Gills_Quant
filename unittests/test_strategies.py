"""
Unit tests for Gills_Quant Base Strategy Interface and Mean Reversion Strategy.

Verifies:
  1. Enums and dataclasses (PositionUnit, OrderType, TimeHorizon, TargetPosition, StrategyOutput, PortfolioState).
  2. Polars & Pandas DataFrame conversion methods (.to_polars(), .to_pandas()).
  3. MeanReversionStrategy indicator calculation (Z-score, RSI, ATR) and output target positions.
  4. Explicit demonstration of ZERO strategy code changes when running in Offline Backtest Mode (`Backtester`)
     vs Live Schwab Production Mode (`SchwabLiveEngine`).
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from strategies.base import (
    BaseStrategy,
    DataContext,
    OrderType,
    PortfolioState,
    PositionState,
    PositionUnit,
    StrategyOutput,
    TargetPosition,
    TimeHorizon,
)
from strategies.mean_reversion import MeanReversionStrategy
from strategies.engines import (
    BacktestDataContext,
    Backtester,
    SchwabLiveDataContext,
    SchwabLiveEngine,
)


class TestStrategyFramework(unittest.TestCase):

    def test_enums(self):
        """Test PositionUnit, OrderType, and TimeHorizon enum contracts."""
        self.assertEqual(PositionUnit.WEIGHT.value, "WEIGHT")
        self.assertEqual(PositionUnit.DOLLARS.value, "DOLLARS")
        self.assertEqual(PositionUnit.SHARES.value, "SHARES")

        self.assertEqual(OrderType.MARKET.value, "MARKET")
        self.assertEqual(OrderType.LIMIT.value, "LIMIT")
        self.assertEqual(OrderType.STOP.value, "STOP")
        self.assertEqual(OrderType.TWAP.value, "TWAP")

        self.assertEqual(TimeHorizon.from_str("5m"), TimeHorizon.FIVE_MIN)
        self.assertEqual(TimeHorizon.from_str("30m"), TimeHorizon.THIRTY_MIN)
        self.assertEqual(TimeHorizon.from_str("1h"), TimeHorizon.ONE_HOUR)
        self.assertEqual(TimeHorizon.from_str("1d"), TimeHorizon.ONE_DAY)

    def test_dataclasses_and_dataframe_exports(self):
        """Test TargetPosition, StrategyOutput, PortfolioState and DataFrame serialization."""
        now = datetime.now()
        target1 = TargetPosition(
            symbol="AAPL",
            target_value=0.10,
            unit=PositionUnit.WEIGHT,
            order_type=OrderType.MARKET,
            predict_to=TimeHorizon.THIRTY_MIN,
            confidence=0.85,
            stop_loss_price=175.0,
            take_profit_price=190.0,
        )
        target2 = TargetPosition(
            symbol="MSFT",
            target_value=-0.05,
            unit=PositionUnit.WEIGHT,
            order_type=OrderType.LIMIT,
            limit_price=410.0,
            predict_to=TimeHorizon.THIRTY_MIN,
            confidence=0.60,
        )

        output = StrategyOutput(
            timestamp=now,
            strategy_name="TestStrategy",
            positions=[target1, target2],
        )

        # Test Polars Export
        df_pl = output.to_polars()
        self.assertIsInstance(df_pl, pl.DataFrame)
        self.assertEqual(len(df_pl), 2)
        self.assertIn("symbol", df_pl.columns)
        self.assertIn("target_value", df_pl.columns)
        self.assertIn("confidence", df_pl.columns)
        self.assertEqual(df_pl.filter(pl.col("symbol") == "AAPL")["target_value"][0], 0.10)

        # Test Pandas Export
        df_pd = output.to_pandas()
        self.assertIsInstance(df_pd, pd.DataFrame)
        self.assertEqual(len(df_pd), 2)
        self.assertEqual(df_pd[df_pd["symbol"] == "AAPL"]["target_value"].values[0], 0.10)

        # Test PortfolioState Export
        pos_aapl = PositionState(
            symbol="AAPL",
            shares=100.0,
            average_price=170.0,
            current_price=180.0,
            market_value=18000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=0.0588,
            weight=0.18,
        )
        portfolio = PortfolioState(
            timestamp=now,
            total_equity=100000.0,
            cash=82000.0,
            positions={"AAPL": pos_aapl},
        )
        self.assertEqual(portfolio.get_position_weight("AAPL"), 0.18)
        self.assertEqual(portfolio.get_position_shares("MSFT"), 0.0)

        port_pl = portfolio.to_polars()
        self.assertIsInstance(port_pl, pl.DataFrame)
        self.assertEqual(len(port_pl), 1)

    def test_mean_reversion_strategy_signals(self):
        """Test MeanReversionStrategy signal calculation and target positioning."""
        strategy = MeanReversionStrategy(
            name="MR_Test",
            symbols=["AAPL", "MSFT"],
            parameters={
                "lookback_window": 10,
                "entry_z_score": 1.5,
                "exit_z_score": 0.5,
                "max_position_weight": 0.10,
            },
            time_horizon=TimeHorizon.THIRTY_MIN,
        )

        now = datetime(2026, 8, 12, 10, 0, 0)
        records = []
        # Generate 20 historical bars for AAPL (oversold trend -> drop at end)
        for i in range(20):
            ts = now - timedelta(minutes=30 * (20 - i))
            # AAPL drops sharply at last step -> Z-score will be negative (oversold)
            p_aapl = 100.0 - (i * 0.2) if i < 18 else 85.0
            records.append({
                "timestamp": ts,
                "symbol": "AAPL",
                "open": p_aapl + 0.1,
                "high": p_aapl + 0.5,
                "low": p_aapl - 0.5,
                "close": p_aapl,
                "volume": 10000,
            })
            # MSFT stays flat near mean
            p_msft = 300.0 + (i % 2) * 0.1
            records.append({
                "timestamp": ts,
                "symbol": "MSFT",
                "open": p_msft,
                "high": p_msft + 0.2,
                "low": p_msft - 0.2,
                "close": p_msft,
                "volume": 10000,
            })

        df_hist = pl.DataFrame(records)
        ctx = BacktestDataContext(now, df_hist, ["AAPL", "MSFT"])

        output = strategy.predict(ctx)
        self.assertEqual(output.strategy_name, "MR_Test")
        self.assertEqual(len(output.positions), 2)

        aapl_pos = next(p for p in output.positions if p.symbol == "AAPL")
        msft_pos = next(p for p in output.positions if p.symbol == "MSFT")

        # AAPL sharp drop triggers oversold long signal (target_value > 0)
        self.assertGreater(aapl_pos.target_value, 0.0)
        self.assertEqual(aapl_pos.predict_to, TimeHorizon.THIRTY_MIN)
        self.assertIsNotNone(aapl_pos.stop_loss_price)
        self.assertIsNotNone(aapl_pos.take_profit_price)

        # MSFT flat near mean triggers neutral / 0 position
        self.assertEqual(msft_pos.target_value, 0.0)

        # Validate strategy output bounds
        self.assertTrue(strategy.validate_output(output))

    def test_zero_code_change_write_once_run_anywhere(self):
        """
        EXPLICIT DEMONSTRATION OF 'WRITE ONCE, RUN ANYWHERE':
        
        The exact same `strategy` instance is passed to BOTH:
          1. `Backtester` (Offline Backtest Engine)
          2. `SchwabLiveEngine` (Live Schwab Production Engine)
          
        ZERO strategy code changes are made. Both engines evaluate `strategy.predict(ctx, portfolio)`
        identically!
        """
        # 1. Instantiate concrete strategy ONCE
        strategy = MeanReversionStrategy(
            name="Production_MR_Strategy",
            symbols=["AAPL", "SPY"],
            parameters={"lookback_window": 10, "entry_z_score": 1.5},
            time_horizon=TimeHorizon.THIRTY_MIN,
        )

        # Build synthetic market historical data
        now = datetime(2026, 8, 12, 14, 0, 0)
        records = []
        for i in range(15):
            ts = now - timedelta(minutes=30 * (15 - i))
            records.append({"timestamp": ts, "symbol": "AAPL", "open": 180.0, "high": 181.0, "low": 179.0, "close": 180.0 - (i * 0.3), "volume": 5000})
            records.append({"timestamp": ts, "symbol": "SPY", "open": 500.0, "high": 501.0, "low": 499.0, "close": 500.0, "volume": 20000})
        df_hist = pl.DataFrame(records)

        # --------------------------------------------------------------------
        # Mode A: Offline Backtest Execution
        # --------------------------------------------------------------------
        backtester = Backtester(strategy=strategy, historical_df=df_hist, initial_capital=100000.0)
        backtest_outputs = backtester.run([now])

        self.assertEqual(len(backtest_outputs), 1)
        bt_output = backtest_outputs[0]
        self.assertEqual(bt_output.strategy_name, "Production_MR_Strategy")
        self.assertEqual(len(bt_output.positions), 2)

        # --------------------------------------------------------------------
        # Mode B: Live Schwab Production Engine Execution
        # --------------------------------------------------------------------
        schwab_engine = SchwabLiveEngine(strategy=strategy, account_id="SCHWAB_PROD_9999")
        
        live_portfolio = PortfolioState(
            timestamp=now,
            total_equity=100000.0,
            cash=100000.0,
            positions={},
        )
        live_quotes = {"AAPL": 175.5, "SPY": 500.0}

        schwab_engine.start(SchwabLiveDataContext(now, strategy.symbols, live_quotes, df_hist))
        live_result = schwab_engine.execute_tick(
            current_time=now,
            live_quotes=live_quotes,
            portfolio=live_portfolio,
            bar_buffer=df_hist,
        )
        schwab_engine.stop()

        self.assertEqual(live_result["strategy_name"], "Production_MR_Strategy")
        self.assertTrue(live_result["is_valid"])
        self.assertEqual(len(live_result["submitted_orders"]), 2)

        # Confirm both Backtest and Live output structure and parameters are IDENTICAL
        bt_aapl = next(p for p in bt_output.positions if p.symbol == "AAPL")
        live_aapl_order = next(o for o in live_result["submitted_orders"] if o["symbol"] == "AAPL")

        self.assertEqual(bt_aapl.order_type.value, live_aapl_order["order_type"])
        self.assertEqual(bt_aapl.predict_to.value, live_aapl_order["predict_to"])


if __name__ == "__main__":
    unittest.main()
