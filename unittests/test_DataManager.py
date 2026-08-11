import sys
import os
import zarr
import shutil
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from datetime import datetime, timedelta

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.DataManager import DataManager
from logic.UniverseManager import UniverseManager as UM

class TestDataManager(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary directory for test execution
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Backup original DataManager class paths and variables
        self.orig_data_path = DataManager.data_path
        self.orig_hot_path = DataManager.hot_path
        self.orig_cold_path = DataManager.cold_path
        self.orig_hot_path_db = DataManager.hot_path_db
        self.orig_log_path = DataManager.log_path
        self.orig_master_universe = DataManager.master_universe
        self.orig_retention_days = DataManager.hot_data_retention_days
        self.orig_um_dict = UM.universe_dict.copy()

        # Override paths to point inside test_dir
        DataManager.data_path = self.test_dir / 'data'
        DataManager.hot_path = DataManager.data_path / 'hot'
        DataManager.cold_path = DataManager.data_path / 'cold'
        DataManager.hot_path_db = DataManager.hot_path / 'master_db.zarr'
        DataManager.log_path = self.test_dir / 'logs'
        DataManager.master_universe = 'u_test'
        DataManager.hot_data_retention_days = 30

        UM.universe_dict['u_test'] = {'in': [], 'out': []}

        # Patch UM.gen_csv globally for test cases so create_new_db doesn't query TradingView
        self.gen_csv_patcher = patch("logic.DataManager.UM.gen_csv")
        self.mock_gen_csv = self.gen_csv_patcher.start()

        self.dm = DataManager()

    def tearDown(self):
        self.gen_csv_patcher.stop()

        # Restore class attributes
        DataManager.data_path = self.orig_data_path
        DataManager.hot_path = self.orig_hot_path
        DataManager.cold_path = self.orig_cold_path
        DataManager.hot_path_db = self.orig_hot_path_db
        DataManager.log_path = self.orig_log_path
        DataManager.master_universe = self.orig_master_universe
        DataManager.hot_data_retention_days = self.orig_retention_days
        UM.universe_dict = self.orig_um_dict

        # Cleanup temporary files
        shutil.rmtree(self.test_dir)

    # -------------------------------------------------------------------------
    # 1. Initialization & Logging Utility Tests
    # -------------------------------------------------------------------------
    def test_init_creates_directories(self):
        """DataManager __init__ creates hot, cold, and log directories."""
        self.assertTrue(DataManager.hot_path.exists())
        self.assertTrue(DataManager.cold_path.exists())
        self.assertTrue(DataManager.log_path.exists())

    def test_log_error_symbols(self):
        """_log_error_symbols logs symbol errors to file."""
        DataManager._log_error_symbols(["INVALID1", "INVALID2"])
        current_month = datetime.now().strftime('%m_%Y')
        log_file = DataManager.log_path / f"symbol_errors__{current_month}.log"
        self.assertTrue(log_file.exists())
        content = log_file.read_text()
        self.assertIn("INVALID1", content)
        self.assertIn("INVALID2", content)

    def test_log_error_categories(self):
        """_log_error_categories logs category anomalies."""
        DataManager._log_error_categories(["UNKNOWN_CAT"], "quote.securityStatus")
        current_month = datetime.now().strftime('%m_%Y')
        log_file = DataManager.log_path / f"category_errors__{current_month}.log"
        self.assertTrue(log_file.exists())
        content = log_file.read_text()
        self.assertIn("UNKNOWN_CAT", content)

    def test_log_error_missed_idents(self):
        """_log_error_missed_idents logs missing tickers."""
        DataManager._log_error_missed_idents(["NEW_TICKER"])
        current_month = datetime.now().strftime('%m_%Y')
        log_file = DataManager.log_path / f"missed_idents__{current_month}.log"
        self.assertTrue(log_file.exists())
        content = log_file.read_text()
        self.assertIn("NEW_TICKER", content)

    # -------------------------------------------------------------------------
    # 2. Database Shell & Structure Tests
    # -------------------------------------------------------------------------
    def test_create_empty_day_shell(self):
        """create_empty_day_shell returns Dataset with exact coordinates and dimensions."""
        day = "2026-08-01"
        idents = ["AAPL", "MSFT"]
        ds = DataManager.create_empty_day_shell(day, idents)

        self.assertIn("5m", ds.data_vars)
        self.assertIn("1d", ds.data_vars)
        self.assertEqual(list(ds.coords["day"].values), [day])
        self.assertEqual(len(ds.coords["time"]), 288)  # 5-min intervals in a day
        self.assertEqual(list(ds.coords["ident"].values), idents)
        self.assertTrue(np.isnan(ds["5m"].values).all())
        self.assertTrue(np.isnan(ds["1d"].values).all())

    @patch("logic.DataManager.UM.return_universe_list")
    def test_create_new_db(self, mock_return_universe):
        """create_new_db generates CSV and initializes a master Zarr database."""
        mock_return_universe.return_value = ["AAPL", "MSFT"]

        DataManager.create_new_db("2026-08-01")

        self.mock_gen_csv.assert_called_once_with("u_test")
        self.assertTrue(DataManager.hot_path_db.exists())
        ds = xr.open_zarr(DataManager.hot_path_db)
        self.assertEqual(list(ds.day.values), ["2026-08-01"])
        self.assertEqual(list(ds.ident.values), ["AAPL", "MSFT"])
        ds.close()

    @patch("logic.DataManager.UM.return_universe_list")
    def test_add_db_day_shell_same_and_expanding_symbols(self, mock_return_universe):
        """add_db_day_shell appends day shell and handles expanding symbol lists."""
        mock_return_universe.return_value = ["AAPL", "MSFT"]
        DataManager.create_new_db("2026-08-01")

        # 1. Append new day with same symbols
        DataManager.add_db_day_shell("2026-08-02", idents_for_day=["AAPL", "MSFT"])
        ds = xr.open_zarr(DataManager.hot_path_db)
        self.assertEqual(list(ds.day.values), ["2026-08-01", "2026-08-02"])
        self.assertEqual(list(ds.ident.values), ["AAPL", "MSFT"])
        ds.close()

        # 2. Append new day with expanded symbols ("NVDA" added)
        DataManager.add_db_day_shell("2026-08-03", idents_for_day=["AAPL", "MSFT", "NVDA"])
        ds_expanded = xr.open_zarr(DataManager.hot_path_db)
        self.assertEqual(list(ds_expanded.day.values), ["2026-08-01", "2026-08-02", "2026-08-03"])
        self.assertEqual(list(ds_expanded.ident.values), ["AAPL", "MSFT", "NVDA"])
        ds_expanded.close()

    # -------------------------------------------------------------------------
    # 3. Data Ingestion Tests (save_qVar_data, save_fVar_data, corporate actions)
    # -------------------------------------------------------------------------
    @patch("logic.DataManager.UM.return_universe_quotes_raw")
    @patch("logic.DataManager.UM.return_universe_list")
    def test_save_qVar_data_success(self, mock_return_univ, mock_raw_quotes):
        """save_qVar_data writes 5m quote fields to Zarr database."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        # Mock raw quote response
        df_quotes = pl.DataFrame([{
            "ident": "AAPL",
            "quote.askPrice": 150.5,
            "quote.bidPrice": 150.4,
            "quote.securityStatus": "Normal"
        }])
        mock_raw_quotes.return_value = (df_quotes, [])

        DataManager.save_qVar_data("2026-08-01", "09:30")

        ds = xr.open_zarr(DataManager.hot_path_db)
        val = ds["5m"].sel(day="2026-08-01", time="09:30", ident="AAPL", qVar="quote.askPrice").values.item()
        self.assertEqual(val, 150.5)
        ds.close()

    @patch("logic.DataManager.UM.return_universe_quotes_raw")
    @patch("logic.DataManager.UM.return_universe_list")
    def test_save_fVar_data_success(self, mock_return_univ, mock_raw_quotes):
        """save_fVar_data writes 1d fundamental fields to Zarr database."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        df_fundamentals = pl.DataFrame([{
            "ident": "AAPL",
            "quote.closePrice": 155.0,
            "fundamental.declarationDate": "2026-07-15",
            "fundamental.divExDate": "2026-07-15",
            "fundamental.divPayDate": "2026-07-15",
            "fundamental.lastEarningsDate": "2026-07-15",
            "fundamental.nextDivExDate": "2026-07-15",
            "fundamental.nextDivPayDate": "2026-07-15",
            "assetSubType": "ADR",
            "reference.exchange": "Q"
        }])
        mock_raw_quotes.return_value = (df_fundamentals, [])

        DataManager.save_fVar_data("2026-08-01")

        ds = xr.open_zarr(DataManager.hot_path_db)
        close_val = ds["1d"].sel(day="2026-08-01", ident="AAPL", fVar="quote.closePrice").values.item()
        self.assertEqual(close_val, 155.0)
        ds.close()

    @patch("logic.DataManager.UM.return_universe_list")
    def test_save_corporate_actions_local_fallback(self, mock_return_univ):
        """save_corporate_actions_for_day uses local fundamental data for dividends when Alpaca is disabled."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        # Manually populate fundamental divExDate (20260801) and divPayAmount (0.50)
        ds_disk = xr.open_zarr(DataManager.hot_path_db)
        ex_idx = DataManager.fundamental_fields.index("fundamental.divExDate")
        amt_idx = DataManager.fundamental_fields.index("fundamental.divPayAmount")
        
        slice_1d = ds_disk["1d"].values.copy()
        slice_1d[0, 0, ex_idx] = 20260801
        slice_1d[0, 0, amt_idx] = 0.50
        
        ds_disk.close()

        # Update disk via xr.Dataset.to_zarr
        ds_write = xr.Dataset({"1d": (["day", "ident", "fVar"], slice_1d)})
        ds_write.to_zarr(DataManager.hot_path_db, region={"day": slice(0, 1)}, mode="r+")

        DataManager.save_corporate_actions_for_day("2026-08-01", use_alpaca=False)

        ds_after = xr.open_zarr(DataManager.hot_path_db)
        div_written = ds_after["1d"].sel(day="2026-08-01", ident="AAPL", fVar="corporate.divAmount").values.item()
        self.assertEqual(div_written, 0.50)
        ds_after.close()

    @patch("logic.DataManager.UM.return_universe_list")
    def test_save_corporate_actions_stock_dividend(self, mock_return_univ):
        """save_corporate_actions_for_day processes Alpaca stock_dividends into corporate.splitRatio."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        with patch("alpaca.data.historical.corporate_actions.CorporateActionsClient.get_corporate_actions") as mock_alpaca:
            mock_alpaca.return_value = {
                'stock_dividends': [{'symbol': 'AAPL', 'ex_date': '2026-08-01', 'rate': 0.05}]
            }
            DataManager.save_corporate_actions_for_day("2026-08-01", use_alpaca=True)

        ds_after = xr.open_zarr(DataManager.hot_path_db)
        split_written = ds_after["1d"].sel(day="2026-08-01", ident="AAPL", fVar="corporate.splitRatio").values.item()
        self.assertEqual(split_written, 1.05)
        ds_after.close()

    def test_safe_replace_zarr_backup_and_recovery(self):
        """_safe_replace_zarr safely swaps target directory and cleans up backup folder (Fix 2.5)."""
        target = self.test_dir / "test_target.zarr"
        temp = self.test_dir / "test_temp.zarr"
        target.mkdir(parents=True, exist_ok=True)
        temp.mkdir(parents=True, exist_ok=True)
        (target / "old.txt").write_text("old")
        (temp / "new.txt").write_text("new")

        DataManager._safe_replace_zarr(temp, target)
        self.assertTrue(target.exists())
        self.assertTrue((target / "new.txt").exists())
        self.assertFalse((self.test_dir / "test_target.zarr.bak").exists())

    def test_calculate_adjustment_factors_large_dividend(self):
        """calculate_adjustment_factors remains positive when dividend exceeds share price (Fix C)."""
        from logic.lib_adjustments import calculate_adjustment_factors
        close_prices = pl.DataFrame({'day': ["2026-08-01", "2026-08-02"], 'close': [2.0, 2.0]})
        dividends = [{'date': '2026-08-02', 'amount': 3.0}]  # Dividend > share price
        factors_df = calculate_adjustment_factors(close_prices, [], dividends)
        self.assertTrue((factors_df['factor'] > 0).all(), "Adjustment factors must remain positive.")

    # -------------------------------------------------------------------------
    # 4. Backup Operations Tests
    # -------------------------------------------------------------------------
    @patch("logic.DataManager.UM.return_universe_list")
    def test_make_month_cold_backup_month_formatting(self, mock_return_univ):
        """make_month_cold_backup and return_cold_store work identically for int 8 and str '08' (Fix B)."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        DataManager.make_month_cold_backup("08", 2026, overwrite_existing=True)

        cold_ds_str = DataManager.return_cold_store("08", 2026)
        cold_ds_int = DataManager.return_cold_store(8, 2026)
        self.assertIsNotNone(cold_ds_str)
        self.assertIsNotNone(cold_ds_int)
        cold_ds_str.close()
        cold_ds_int.close()

    @patch("logic.DataManager.UM.return_universe_list")
    def test_make_month_cold_backup(self, mock_return_univ):
        """make_month_cold_backup extracts and stores specified month's data into cold storage."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")
        DataManager.add_db_day_shell("2026-08-15")

        DataManager.make_month_cold_backup(8, 2026, overwrite_existing=True)

        cold_file = DataManager.cold_path / "master_db_month__2026_08.zarr"
        self.assertTrue(cold_file.exists())
        ds_cold = xr.open_zarr(cold_file)
        self.assertEqual(len(ds_cold.day.values), 31)
        ds_cold.close()

    @patch("logic.DataManager.UM.return_universe_list")
    def test_create_and_insert_backup(self, mock_return_univ):
        """create_backup and insert_backup back up and restore database directories."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        # Create backup into data_backup
        DataManager.create_backup()
        backup_path = Path(__file__).resolve().parent.parent / "data_backup"
        self.assertTrue(backup_path.exists())

        # Modify active DB
        shutil.rmtree(DataManager.hot_path_db)

        # Restore from backup
        DataManager.insert_backup(remove_existing=True)
        self.assertTrue(DataManager.hot_path_db.exists())

        # Clean backup folder
        if backup_path.exists():
            shutil.rmtree(backup_path)

    @patch("logic.DataManager.UM.return_universe_list")
    def test_emergency_hot_restore(self, mock_return_univ):
        """emergency_hot_restore rebuilds hot store from cold backups."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")
        DataManager.make_month_cold_backup(8, 2026, overwrite_existing=True)

        # Delete hot database
        shutil.rmtree(DataManager.hot_path_db)
        self.assertFalse(DataManager.hot_path_db.exists())

        DataManager.emergency_hot_restore()

        self.assertTrue(DataManager.hot_path_db.exists())
        ds_restored = xr.open_zarr(DataManager.hot_path_db)
        self.assertIn("2026-08-01", list(ds_restored.day.values))
        ds_restored.close()

    # -------------------------------------------------------------------------
    # 5. Retention & Trimming Tests
    # -------------------------------------------------------------------------
    @patch("logic.DataManager.UM.return_universe_list")
    def test_retention_trim_db_removes_old_days_and_nan_idents(self, mock_return_univ):
        """retention_trim_db trims days older than retention window and purges NaN inactive idents."""
        mock_return_univ.return_value = ["ACTIVE_SYM"]
        
        # Setup 2-day retention
        DataManager.hot_data_retention_days = 2
        
        old_day = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        recent_day = datetime.now().strftime("%Y-%m-%d")
        
        # Create database with old_day and recent_day, plus an inactive symbol "ALL_NAN_SYM"
        DataManager.add_db_day_shell(old_day, idents_for_day=["ACTIVE_SYM", "ALL_NAN_SYM"], is_initial_creation=True)
        DataManager.add_db_day_shell(recent_day, idents_for_day=["ACTIVE_SYM", "ALL_NAN_SYM"])

        stats = DataManager.retention_trim_db()

        self.assertIsNotNone(stats)
        self.assertIn(old_day, stats["days_removed"])

        ds_trimmed = xr.open_zarr(DataManager.hot_path_db)
        self.assertEqual(list(ds_trimmed.day.values), [recent_day])
        # ALL_NAN_SYM should be purged as inactive NaN symbol, ACTIVE_SYM retained
        self.assertEqual(list(ds_trimmed.ident.values), ["ACTIVE_SYM"])
        ds_trimmed.close()

    # -------------------------------------------------------------------------
    # 6. Database Inspection & Utility Tests
    # -------------------------------------------------------------------------
    @patch("logic.DataManager.UM.return_universe_list")
    def test_return_db_stats(self, mock_return_univ):
        """return_db_stats returns correct dictionary metadata."""
        mock_return_univ.return_value = ["AAPL", "MSFT"]
        
        # No DB exists initially
        self.assertIsNone(DataManager.return_db_stats())

        DataManager.create_new_db("2026-08-01")
        stats = DataManager.return_db_stats()

        self.assertEqual(stats["num_days"], 1)
        self.assertEqual(stats["num_idents"], 2)
        self.assertEqual(stats["num_qVars"], len(DataManager.quote_fields))
        self.assertEqual(stats["num_fVars"], len(DataManager.fundamental_fields))
        self.assertEqual(stats["current_universe_size"], 2)

    @patch("logic.DataManager.UM.return_universe_list")
    def test_return_hot_and_cold_store(self, mock_return_univ):
        """return_hot_store and return_cold_store return valid Datasets."""
        mock_return_univ.return_value = ["AAPL"]
        
        self.assertIsNone(DataManager.return_hot_store())
        self.assertIsNone(DataManager.return_cold_store(8, 2026))

        DataManager.create_new_db("2026-08-01")
        DataManager.make_month_cold_backup(8, 2026, overwrite_existing=True)

        hot_ds = DataManager.return_hot_store()
        self.assertIsNotNone(hot_ds)
        hot_ds.close()

        cold_ds = DataManager.return_cold_store(8, 2026)
        self.assertIsNotNone(cold_ds)
        cold_ds.close()

    def test_gen_test_db(self):
        """gen_test_db creates a mock database with synthetic price walks."""
        start_date = "2026-08-01"
        DataManager.gen_test_db(num_days=5, num_idents=4, start_date=start_date, num_full_nan_idents=1, random_day_skips=False)

        self.assertTrue(DataManager.hot_path_db.exists())
        ds = xr.open_zarr(DataManager.hot_path_db)
        self.assertEqual(len(ds.day), 5)
        self.assertEqual(len(ds.ident), 4)
        self.assertIn("FULLNAN00000", list(ds.ident.values))
        ds.close()

    @patch("logic.DataManager.DataManager.save_corporate_actions_for_day")
    @patch("logic.DataManager.UM.return_universe_list")
    def test_backfill_missing_days_and_corporate_actions(self, mock_return_univ, mock_save_ca):
        """backfill_missing_days_and_corporate_actions scans and backfills missing days in current & prev month."""
        mock_return_univ.return_value = ["AAPL"]
        current_date = datetime.now().date()
        first_curr = current_date.replace(day=1)
        last_prev = first_curr - timedelta(days=1)
        start_date_str = last_prev.replace(day=1).strftime("%Y-%m-%d")

        # Create DB with only start_date_str
        DataManager.create_new_db(start_date_str)

        DataManager.backfill_missing_days_and_corporate_actions()

        ds = xr.open_zarr(DataManager.hot_path_db)
        # Should have added missing days up to today
        self.assertGreater(len(ds.day), 1)
        ds.close()
        self.assertTrue(mock_save_ca.called)

    @patch("logic.DataManager.UM.return_universe_list")
    def test_has_fundamental_data(self, mock_return_univ):
        """has_fundamental_data returns False for empty NaN shells and True once populated."""
        mock_return_univ.return_value = ["AAPL"]
        DataManager.create_new_db("2026-08-01")

        # Initially empty NaN shell
        self.assertFalse(DataManager.has_fundamental_data("2026-08-01"))

        # Populate with mock fundamentals
        mock_df = pl.DataFrame([{
            "ident": "AAPL",
            "fundamental.declarationDate": "2026-07-15",
            "fundamental.divExDate": "2026-07-15",
            "fundamental.divPayDate": "2026-07-15",
            "fundamental.lastEarningsDate": "2026-07-15",
            "fundamental.nextDivExDate": "2026-07-15",
            "fundamental.nextDivPayDate": "2026-07-15",
            "assetSubType": "ADR",
            "reference.exchange": "Q"
        }])
        with patch("logic.DataManager.UM.return_universe_quotes_raw", return_value=(mock_df, [])):
            DataManager.save_fVar_data("2026-08-01")

        self.assertTrue(DataManager.has_fundamental_data("2026-08-01"))


if __name__ == "__main__":
    unittest.main()
