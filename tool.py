# General Imports
import os
import sys
from pathlib import Path
from datetime import datetime

# Import local modules
from logic.lib_files import *
from logic.lib_clients import *
from logic.DataManager import DataManager as DM

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
            print("  [s-d] setup-dirs       Set up the directory structure (non-destructive).")
            print("  ----------------------- Test Commands ---------------------------")
            print("  [t-s] test-schwab      Test the Schwab client connection.")
            print("  ----------------------- Database Commands -----------------------")
            print("  [db-g] db-gen          Generate a new database for the current date.")
            print("  [db-b] db-backup       Backup the current database.")
            print("  [db-r] db-restore      Restore the database from the latest backup.")
            print(" ------------------------------------------------------------------")
            print("  [e] exit               Exit the tool.")

        # Setup Commands:
        if command in ["s-d", "setup-dirs"]:
            setup_dir_structure()

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
            dm.backup_db()
        if command in ['db-r', 'db-restore']:
            clean = input("Delete current database before restoring backup? (y/n): ").strip().lower()
            clean = clean in ['y', 'yes']
            overwrite_hot = input("Overwrite hot database? (y/n): ").strip().lower()
            overwrite_hot = overwrite_hot in ['y', 'yes']
            overwrite_cold = input("Overwrite cold database files? (y/n): ").strip().lower()
            overwrite_cold = overwrite_cold in ['y', 'yes']
            dm = DM()
            dm.restore_db_backup(clean=clean, overwrite_hot=overwrite_hot, overwrite_cold=overwrite_cold)