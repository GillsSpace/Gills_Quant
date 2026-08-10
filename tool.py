# General Imports
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import local modules
from logic.lib_files import *
from logic.lib_clients import *
from logic.DataManager import DataManager as DM
from logic.UniverseManager import UniverseManager as UM

if __name__ == "__main__":
    print("Gills Quant Tool")
    print("Type 'help' or 'h' for a list of commands.")
    command = ""
    while command not in ["exit", "e", "quit", "q"]:
        print("\n----------------------------------------")
        command = input("Enter command: ").strip().lower()

        # Help Test:
        if command in ["help", "h"]:
            print("Available commands:")
            print("  ----------------------- Setup Commands --------------------------")
            print("  [s-d]  setup-dirs            Set up the directory structure (non-destructive).")
            print("  [s-c]  setup-collection      Create a new universe 'u00' (stocks & funds) and generate its CSV file.")
            print("  ----------------------- Test Commands ---------------------------")
            print("  [t-s] test-schwab            Test the Schwab client connection.")
            print("  ----------------------- Database Commands -----------------------")
            print("  [db-g] db-gen                Generate a new database for the current date.")
            print("  [db-b] db-backup             Backup the current database.")
            print("  [db-r] db-restore            Restore the database from the latest backup.")
            print("  [db-s] db-status             Show the status and statistics of the current database.")
            print("  [db-a] db-archive            Moves data in hot storage to cold storage.")
            print("  [db-t] db-trim               Manual Hot DB trim.")
            print("  [db-e] db-emergency-restore  Emergency restore from backup (use with caution, will overwrite hot db).")
            print(" ------------------------------------------------------------------")
            print("  [e] exit                     Exit the tool.")

        # Setup Commands:
        if command in ["s-d", "setup-dirs"]:
            setup_dir_structure()
        if command in ["s-c", "setup-collection"]:
            print("This will create/overwrite a new universe 'u00' (stocks & funds) and generate its CSV file.")
            confirmation = input("This will also create a new database for the current date. This will delete any existing database. Continue? (y/n): ").strip().lower()
            if confirmation not in ['y', 'yes']:
                print("Operation cancelled.")
                continue
            dm = DM()
            UM.gen_csv('u00')
            current_date = datetime.now().strftime("%Y-%m-%d")
            dm.create_new_db(current_date)
            

        # Test Commands:
        if command in ['t-s', 'test-schwab']:
            test_client_schwab()


        # Database Commands:
        if command in ['db-g', 'db-gen']:
            confirmation = input("This will create a new database for the current date. This will delete any existing database. Continue? (y/n): ").strip().lower()
            if confirmation not in ['y', 'yes']:
                print("Operation cancelled.")
                continue
            dm = DM()
            current_date = datetime.now().strftime("%Y-%m-%d")
            dm.create_new_db(current_date)
        if command in ['db-b', 'db-backup']:
            dm = DM()
            dm.create_backup()
        if command in ['db-r', 'db-restore']:
            clean = input("Delete current database before restoring backup? (y/n): ").strip().lower()
            clean = clean in ['y', 'yes']
            overwrite_hot = input("Overwrite hot database? (y/n): ").strip().lower()
            overwrite_hot = overwrite_hot in ['y', 'yes']
            overwrite_cold = input("Overwrite cold database files? (y/n): ").strip().lower()
            overwrite_cold = overwrite_cold in ['y', 'yes']
            dm = DM()
            # Map CLI flags to DataManager.insert_backup parameters:
            # clean -> remove_existing, overwrite_hot -> overwrite_existing_hot, overwrite_cold -> overwrite_existing_cold
            dm.insert_backup(overwrite_existing_cold=overwrite_cold, overwrite_existing_hot=overwrite_hot, remove_existing=clean)
        if command in ['db-s','db-status']:
            dm = DM()
            stats = dm.return_db_stats()
            print("Database Status:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        if command in ['db-a','db-archive']:
            single = input("Archive a single month or all months in current hot db? (s/A):").strip().lower()
            single = True if single in ['s','single'] else False
            if single:
                month_str = input("Enter month to archive (YYYY-MM): ").strip()
                try:
                    year, month = month_str.split("-")
                    year, month = int(year), int(month)
                    overwrite = input("Overwrite existing cold store for this month? Generally should not be done as hot db will contian incomplete months. (y/N) ").strip().lower()
                    overwrite = True if overwrite in ['y','yes'] else False
                    dm = DM()
                    dm.make_month_cold_backup(month=month, year=year, overwrite_existing=overwrite)
                except Exception as e:
                    print(f"Invalid month format or error during backup: {e}")
            else:
                overwrite = input("Overwrite existing cold stores? Generally should not be done as hot db will contian incomplete months. (y/N) ").strip().lower()
                overwrite = True if overwrite in ['y','yes'] else False
                dm = DM()
                zarr_store = DM.return_hot_store()
                months = pd.to_datetime(zarr_store.day.values).strftime("%Y-%m").unique().to_list()
                for month in months:
                    print(f"    Archiving month: {month}")
                    year, month = month.split("-")
                    year, month = int(year), int(month)
                    dm.make_month_cold_backup(month=month, year=year, overwrite_existing=overwrite)
        if command in ['db-t','db-trim']:
            dm = DM()
            result = dm.retention_trim_db()
            print("Retention Trim Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")

        if command in ['db-e','db-emergency-restore']:
            confirmation = input("This will restore the database from the latest backup and overwrite the current hot database. This can lead to data loss if the backup is not recent. Are you sure you want to proceed? (y/n): ").strip().lower()
            if confirmation not in ['y', 'yes']:
                print("Operation cancelled.")
                continue
            dm = DM()
            dm.emergency_hot_restore()   
            store = dm.return_hot_store()
            print(f"Current hot store date range after restore: {store.day.values[0]} to {store.day.values[-1]}") 

