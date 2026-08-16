import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import polars as pl
from datetime import datetime

# Add the project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.UniverseManager import UniverseManager

class TestUniverseManager(unittest.TestCase):

    def setUp(self):
        # Create a temporary isolated directory for universes and logs
        self.test_dir = Path(tempfile.mkdtemp())
        self.universe_dir = self.test_dir / "universes"
        self.log_dir = self.test_dir / "logs"
        self.universe_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Backup original class attributes
        self.orig_universe_folder_path = UniverseManager.universe_folder_path
        self.orig_log_base_path = UniverseManager.log_base_path
        self.orig_universe_dict = UniverseManager.universe_dict.copy()

        # Override paths to point to temporary directories
        UniverseManager.universe_folder_path = self.universe_dir
        UniverseManager.log_base_path = self.log_dir

        # Add a dummy universe dict entry for testing
        UniverseManager.universe_dict["u_test"] = {
            "in": ["dummy_in_condition"],
            "out": ["dummy_out_condition"]
        }

        self.update_status_patcher = patch("logic.lib_files.update_status")
        self.mock_update_status = self.update_status_patcher.start()

    def tearDown(self):
        self.update_status_patcher.stop()

        # Restore class attributes
        UniverseManager.universe_folder_path = self.orig_universe_folder_path
        UniverseManager.log_base_path = self.orig_log_base_path
        UniverseManager.universe_dict = self.orig_universe_dict

        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # 1. Input Validation & Exception Handling Tests
    # -------------------------------------------------------------------------
    def test_gen_csv_invalid_code_raises_value_error(self):
        """gen_csv raises ValueError when invalid universe_code is provided."""
        with self.assertRaises(ValueError) as ctx:
            UniverseManager.gen_csv("invalid_code_999")
        self.assertIn("not found in universe dictionary", str(ctx.exception))

    def test_return_universe_list_file_not_found(self):
        """return_universe_list raises FileNotFoundError when CSV does not exist."""
        with self.assertRaises(FileNotFoundError) as ctx:
            UniverseManager.return_universe_list("nonexistent_universe")
        self.assertIn("not found", str(ctx.exception))

    def test_return_universe_quotes_raw_empty_or_whitespace_code(self):
        """return_universe_quotes_raw returns (None, errors) for empty/whitespace universe_code."""
        df, errors = UniverseManager.return_universe_quotes_raw("")
        self.assertIsNone(df)
        self.assertIn("Universe code must be a non-empty string.", errors)

        df_space, errors_space = UniverseManager.return_universe_quotes_raw("   ")
        self.assertIsNone(df_space)
        self.assertIn("Universe code must be a non-empty string.", errors_space)

    # -------------------------------------------------------------------------
    # 2. Universe List Retrieval Tests (return_universe_list)
    # -------------------------------------------------------------------------
    def test_return_universe_list_success(self):
        """return_universe_list successfully reads ticker names from CSV."""
        csv_file = self.universe_dir / "u_test.csv"
        df_sample = pd.DataFrame({"name": ["AAPL", "MSFT", "GOOGL"]})
        df_sample.to_csv(csv_file, index=False)

        tickers = UniverseManager.return_universe_list("u_test")
        self.assertEqual(tickers, ["AAPL", "MSFT", "GOOGL"])

    def test_return_universe_list_empty_csv(self):
        """return_universe_list returns [] when CSV has headers but no rows."""
        csv_file = self.universe_dir / "u_test.csv"
        pd.DataFrame(columns=["name"]).to_csv(csv_file, index=False)

        tickers = UniverseManager.return_universe_list("u_test")
        self.assertEqual(tickers, [])

    # -------------------------------------------------------------------------
    # 3. CSV Generation Tests (gen_csv)
    # -------------------------------------------------------------------------
    @patch("logic.UniverseManager.Query")
    def test_gen_csv_success_and_ticker_formatting(self, mock_query_cls):
        """gen_csv creates short & long CSVs, renames tickers, sorts, and logs."""
        # Setup mock scanner data
        raw_data = [
            0,
            pd.DataFrame({
                "name": ["BRK/PA", "TSLA", "AAPL.A"],
                "sector": ["Financials", "Consumer Cyclical", "Technology"],
                "exchange": ["NYSE", "NASDAQ", "NASDAQ"],
                "industry": ["Insurance", "Auto", "Consumer Electronics"],
                "close": [400.0, 200.0, 150.0],
                "average_volume_30d_calc": [1000, 5000, 10000],
                "market_cap_basic": [500e9, 600e9, 2e12]
            })
        ]
        mock_query_inst = MagicMock()
        mock_query_inst.select.return_value = mock_query_inst
        mock_query_inst.where.return_value = mock_query_inst
        mock_query_inst.limit.return_value = mock_query_inst
        mock_query_inst.get_scanner_data.return_value = raw_data
        mock_query_cls.return_value = mock_query_inst

        UniverseManager.gen_csv("u_test")

        long_csv = self.universe_dir / "u_test_long.csv"
        short_csv = self.universe_dir / "u_test.csv"
        self.assertTrue(long_csv.exists())
        self.assertTrue(short_csv.exists())

        # Verify ticker transformations (AAPL.A -> AAPL/A, BRK/PA -> BRK/PRA) and sorting
        df_long = pd.read_csv(long_csv)
        self.assertEqual(df_long["name"].tolist(), ["AAPL/A", "BRK/PRA", "TSLA"])

        df_short = pd.read_csv(short_csv)
        self.assertEqual(df_short["name"].tolist(), ["AAPL/A", "BRK/PRA", "TSLA"])

        # Check log file generation
        current_month = datetime.now().strftime('%m_%Y')
        log_file = self.log_dir / f"universe_change__{current_month}.log"
        self.assertTrue(log_file.exists())
        log_content = log_file.read_text()
        self.assertIn("Freshly Generated universe u_test with 3 symbols", log_content)

    @patch("logic.UniverseManager.Query")
    def test_ticker_transformation_idempotency(self, mock_query_cls):
        """gen_csv ticker transformation does not corrupt already formatted /PR tickers."""
        raw_data = [
            0,
            pd.DataFrame({
                "name": ["BRK/PRA"],
                "sector": ["Financials"],
                "exchange": ["NYSE"],
                "industry": ["Insurance"],
                "close": [400.0],
                "average_volume_30d_calc": [1000],
                "market_cap_basic": [500e9]
            })
        ]
        mock_query_inst = MagicMock()
        mock_query_inst.select.return_value = mock_query_inst
        mock_query_inst.where.return_value = mock_query_inst
        mock_query_inst.limit.return_value = mock_query_inst
        mock_query_inst.get_scanner_data.return_value = raw_data
        mock_query_cls.return_value = mock_query_inst

        UniverseManager.gen_csv("u_test")
        df_short = pd.read_csv(self.universe_dir / "u_test.csv")
        self.assertEqual(df_short["name"].tolist(), ["BRK/PRA"])

    # -------------------------------------------------------------------------
    # 4. Universe Regeneration Tests (regen_csv)
    # -------------------------------------------------------------------------
    @patch("logic.UniverseManager.Query")
    def test_regen_csv_from_scratch(self, mock_query_cls):
        """regen_csv works properly when no existing CSV file exists."""
        in_data = [
            0,
            pd.DataFrame({
                "name": ["NVDA", "AMZN"],
                "sector": ["Tech", "Retail"],
                "exchange": ["NASDAQ", "NASDAQ"],
                "industry": ["Semis", "E-Comm"],
                "close": [120.0, 180.0],
                "average_volume_30d_calc": [20000, 15000],
                "market_cap_basic": [3e12, 1.8e12]
            })
        ]
        mock_query_inst = MagicMock()
        mock_query_inst.select.return_value = mock_query_inst
        mock_query_inst.where.return_value = mock_query_inst
        mock_query_inst.limit.return_value = mock_query_inst
        mock_query_inst.get_scanner_data.return_value = in_data
        mock_query_cls.return_value = mock_query_inst

        UniverseManager.regen_csv("u_test")

        short_csv = self.universe_dir / "u_test.csv"
        self.assertTrue(short_csv.exists())
        df_short = pd.read_csv(short_csv)
        self.assertEqual(df_short["name"].tolist(), ["AMZN", "NVDA"])

    @patch("logic.UniverseManager.Query")
    def test_regen_csv_with_existing_and_out_criteria(self, mock_query_cls):
        """regen_csv retains existing stocks matching out criteria and adds new stocks."""
        # Create existing long CSV
        long_csv = self.universe_dir / "u_test_long.csv"
        existing_df = pd.DataFrame({
            "ticker": ["AAPL", "OLD1"],
            "name": ["AAPL", "OLD1"],
            "sector": ["Tech", "Misc"],
            "exchange": ["NASDAQ", "NYSE"],
            "industry": ["Consumer Elec", "Misc"],
            "close": [150.0, 50.0],
            "average_volume_30d_calc": [10000, 200],
            "market_cap_basic": [2e12, 1e9]
        })
        existing_df.to_csv(long_csv, index=False)

        # In query returns NEW_STOCK; Out query returns OLD1 (so OLD1 is retained)
        in_df = pd.DataFrame({
            "name": ["NEW_STOCK"], "sector": ["Tech"], "exchange": ["NASDAQ"],
            "industry": ["Software"], "close": [80.0], "average_volume_30d_calc": [5000],
            "market_cap_basic": [5e10]
        })
        out_df = pd.DataFrame({
            "name": ["OLD1"], "sector": ["Misc"], "exchange": ["NYSE"],
            "industry": ["Misc"], "close": [50.0], "average_volume_30d_calc": [200],
            "market_cap_basic": [1e9]
        })

        query_in_inst = MagicMock()
        query_in_inst.select.return_value = query_in_inst
        query_in_inst.where.return_value = query_in_inst
        query_in_inst.limit.return_value = query_in_inst
        query_in_inst.get_scanner_data.return_value = (0, in_df)

        query_out_inst = MagicMock()
        query_out_inst.select.return_value = query_out_inst
        query_out_inst.where.return_value = query_out_inst
        query_out_inst.limit.return_value = query_out_inst
        query_out_inst.get_scanner_data.return_value = (0, out_df)

        mock_query_cls.side_effect = [query_in_inst, query_out_inst]

        UniverseManager.regen_csv("u_test")

        df_short = pd.read_csv(self.universe_dir / "u_test.csv")
        # Combined result should contain NEW_STOCK and OLD1, sorted
        self.assertEqual(df_short["name"].tolist(), ["NEW_STOCK", "OLD1"])

        # Check change log for additions/removals
        current_month = datetime.now().strftime('%m_%Y')
        log_file = self.log_dir / f"universe_change__{current_month}.log"
        log_content = log_file.read_text()
        self.assertIn("Added 1 symbols: NEW_STOCK", log_content)
        self.assertIn("Removed 1 symbols: AAPL", log_content)

    @patch("logic.UniverseManager.Query")
    def test_regen_csv_existing_csv_without_ticker_column(self, mock_query_cls):
        """regen_csv does not crash when existing _long.csv lacks a 'ticker' column (Fix A)."""
        long_csv = self.universe_dir / "u_test_long.csv"
        existing_df = pd.DataFrame({
            "name": ["AAPL"],
            "sector": ["Tech"],
            "exchange": ["NASDAQ"],
            "industry": ["Consumer Elec"],
            "close": [150.0],
            "average_volume_30d_calc": [10000],
            "market_cap_basic": [2e12]
        })
        existing_df.to_csv(long_csv, index=False)

        query_inst = MagicMock()
        query_inst.select.return_value = query_inst
        query_inst.where.return_value = query_inst
        query_inst.limit.return_value = query_inst
        query_inst.get_scanner_data.return_value = (0, pd.DataFrame())
        mock_query_cls.return_value = query_inst

        # Should execute cleanly without raising KeyError: "['ticker'] not found in axis"
        UniverseManager.regen_csv("u_test")
        self.assertTrue(long_csv.exists())

    @patch("logic.UniverseManager.Query")
    def test_regen_csv_empty_results(self, mock_query_cls):
        """regen_csv handles empty results from scanner queries cleanly (Fix C schema check)."""
        query_inst = MagicMock()
        query_inst.select.return_value = query_inst
        query_inst.where.return_value = query_inst
        query_inst.limit.return_value = query_inst
        query_inst.get_scanner_data.return_value = (0, pd.DataFrame())
        mock_query_cls.return_value = query_inst

        UniverseManager.regen_csv("u_test")

        short_csv = self.universe_dir / "u_test.csv"
        long_csv = self.universe_dir / "u_test_long.csv"
        self.assertTrue(short_csv.exists())
        self.assertTrue(long_csv.exists())

        df_long = pl.read_csv(long_csv)
        self.assertTrue(df_long.is_empty())
        self.assertEqual(list(df_long.columns), UniverseManager.long_file_vars)

    # -------------------------------------------------------------------------
    # 5. Raw Quotes Retrieval Tests (return_universe_quotes_raw)
    # -------------------------------------------------------------------------
    def test_return_universe_quotes_raw_empty_universe(self):
        """return_universe_quotes_raw returns (None, errors) when universe file is empty."""
        csv_file = self.universe_dir / "u_test.csv"
        pd.DataFrame(columns=["name"]).to_csv(csv_file, index=False)

        df, errors = UniverseManager.return_universe_quotes_raw("u_test")
        self.assertIsNone(df)
        self.assertIn("Universe u_test is empty.", errors)

    @patch("logic.UniverseManager.create_client_schwab")
    def test_return_universe_quotes_raw_success(self, mock_create_client):
        """return_universe_quotes_raw fetches quotes in batches and returns DataFrame."""
        # Setup universe CSV
        csv_file = self.universe_dir / "u_test.csv"
        pd.DataFrame({"name": ["AAPL", "MSFT"]}).to_csv(csv_file, index=False)

        # Mock Schwab client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "AAPL": {"closePrice": 150.0, "volume": 100000},
            "MSFT": {"closePrice": 300.0, "volume": 200000}
        }
        mock_client.quotes.return_value = mock_response
        mock_create_client.return_value = mock_client

        df, errors = UniverseManager.return_universe_quotes_raw("u_test")

        self.assertIsNotNone(df)
        self.assertEqual(errors, [])
        self.assertEqual(len(df), 2)
        self.assertIn("ident", df.columns)
        self.assertListEqual(sorted(df["ident"].to_list()), ["AAPL", "MSFT"])

    @patch("logic.UniverseManager.create_client_schwab")
    def test_return_universe_quotes_raw_batch_failure(self, mock_create_client):
        """return_universe_quotes_raw captures batch exceptions and reports errors."""
        csv_file = self.universe_dir / "u_test.csv"
        pd.DataFrame({"name": ["AAPL"]}).to_csv(csv_file, index=False)

        mock_client = MagicMock()
        mock_client.quotes.side_effect = Exception("API rate limit exceeded")
        mock_create_client.return_value = mock_client

        df, errors = UniverseManager.return_universe_quotes_raw("u_test")

        self.assertIsNone(df)
        self.assertTrue(any("API rate limit exceeded" in err for err in errors))
        self.assertTrue(any("No quotes retrieved" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
