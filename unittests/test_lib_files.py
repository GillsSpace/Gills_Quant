import unittest
import json
import tempfile
import shutil
from pathlib import Path

from logic.lib_files import (
    update_status,
    update_cron_status,
    reset_daily_status,
    DEFAULT_CRON_STATUS,
    setup_dir_structure
)

class TestLibFiles(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.status_file = self.test_path / 'status.json'
        self.universes_dir = self.test_path / 'universes'
        self.universes_dir.mkdir(parents=True, exist_ok=True)
        self.filings_file = self.universes_dir / 'todays_filing_symbols.json'

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_update_status_and_nested_merge(self):
        """Test update_status creates and merges nested dictionaries like cron_status and breakdown."""
        # Initial write
        update_status({
            "hot_db_symbols_count": 100,
            "cron_status": {"daily_status_reset": "Completed", "sec_ticker_cik_map": "Scheduled"}
        }, base_path=self.test_path)

        with open(self.status_file, 'r') as f:
            s1 = json.load(f)
        self.assertEqual(s1["hot_db_symbols_count"], 100)
        self.assertEqual(s1["cron_status"]["daily_status_reset"], "Completed")
        self.assertEqual(s1["cron_status"]["sec_ticker_cik_map"], "Scheduled")

        # Nested update to cron_status should merge, not overwrite other entries
        update_status({
            "cron_status": {"sec_ticker_cik_map": "Completed", "sec_rss_filings": "Completed"}
        }, base_path=self.test_path)

        with open(self.status_file, 'r') as f:
            s2 = json.load(f)
        self.assertEqual(s2["hot_db_symbols_count"], 100)
        self.assertEqual(s2["cron_status"]["daily_status_reset"], "Completed")
        self.assertEqual(s2["cron_status"]["sec_ticker_cik_map"], "Completed")
        self.assertEqual(s2["cron_status"]["sec_rss_filings"], "Completed")

    def test_reset_daily_status_logic(self):
        """Test reset_daily_status shifts filings, resets flags, and resets cron matrix."""
        initial_data = {
            "hot_db_symbols_count": 9800,
            "edgar_filings_symbols_today": 42,
            "edgar_filings_symbols_yesterday": 10,
            "fundamental_data_pulled_today": True,
            "cron_status": {
                "daily_status_reset": "Scheduled",
                "sec_ticker_cik_map": "Completed",
                "sec_rss_filings": "Completed",
                "sec_xbrl_facts": "Completed",
                "daily_fundamentals": "Completed"
            }
        }
        with open(self.status_file, 'w') as f:
            json.dump(initial_data, f)

        # Setup test filings file
        with open(self.filings_file, 'w') as f:
            json.dump(["AAPL", "MSFT"], f)

        reset_daily_status(base_path=self.test_path)

        self.assertTrue(self.status_file.exists())
        with open(self.status_file, 'r') as f:
            res = json.load(f)

        # Check that yesterday's filings took today's count (42) and today is 0
        self.assertEqual(res['edgar_filings_symbols_yesterday'], 42)
        self.assertEqual(res['edgar_filings_symbols_today'], 0)

        # Check fundamental flag reset
        self.assertFalse(res['fundamental_data_pulled_today'])

        # Check cron status reset with all jobs set to Scheduled / Standby
        self.assertEqual(res['cron_status']['daily_status_reset'], 'Scheduled')
        self.assertEqual(res['cron_status']['sec_ticker_cik_map'], 'Scheduled')
        self.assertEqual(res['cron_status']['sec_rss_filings'], 'Scheduled')
        self.assertEqual(res['cron_status']['sec_xbrl_facts'], 'Scheduled')
        self.assertEqual(res['cron_status']['daily_fundamentals'], 'Scheduled')
        self.assertEqual(res['cron_status']['fundamental_retry'], 'Standby')
        self.assertEqual(res['cron_status']['universe_regeneration'], 'Scheduled')
        self.assertEqual(res['cron_status']['retention_trim'], 'Scheduled')
        self.assertEqual(res['cron_status']['monthly_cold_backup'], 'Scheduled')
        self.assertEqual(res['cron_status']['daily_notification'], 'Scheduled')

        # Check persistent fields preserved
        self.assertEqual(res['hot_db_symbols_count'], 9800)

        # Check filing cache cleared
        with open(self.filings_file, 'r') as f:
            cleared_filings = json.load(f)
        self.assertEqual(cleared_filings, [])

    def test_update_cron_status_individual_job(self):
        """Test that updating an individual cron status preserves other cron entries."""
        initial_data = {
            "cron_status": DEFAULT_CRON_STATUS.copy()
        }
        with open(self.status_file, 'w') as f:
            json.dump(initial_data, f)

        update_cron_status("sec_ticker_cik_map", "Completed", base_path=self.test_path)

        with open(self.status_file, 'r') as f:
            res = json.load(f)

        self.assertEqual(res['cron_status']['sec_ticker_cik_map'], 'Completed')
        self.assertEqual(res['cron_status']['daily_status_reset'], 'Scheduled')
        self.assertEqual(res['cron_status']['sec_rss_filings'], 'Scheduled')
        self.assertEqual(res['cron_status']['daily_fundamentals'], 'Scheduled')

    def test_corrupted_status_json_recovery(self):
        """Test that reset_daily_status recovers gracefully from malformed status.json."""
        with open(self.status_file, 'w') as f:
            f.write("{invalid json content---")

        reset_daily_status(base_path=self.test_path)

        self.assertTrue(self.status_file.exists())
        with open(self.status_file, 'r') as f:
            res = json.load(f)

        self.assertEqual(res['cron_status']['daily_status_reset'], 'Scheduled')
        self.assertEqual(res['edgar_filings_symbols_today'], 0)
        self.assertFalse(res['fundamental_data_pulled_today'])

if __name__ == '__main__':
    unittest.main()
