# General Imports
import os
import sys
from pathlib import Path

# Import local modules
from logic.lib_file_management import *

if __name__ == "__main__":
    print("Gills Quant Tool")
    print("Type 'help' or 'h' for a list of commands.")
    command = ""
    while command not in ["exit", "e", "quit", "q"]:
        print("\n----------------------------------------")
        command = input("Enter command: ").strip().lower()

        if command in ["help", "h"]:
            print("Available commands:")
            print("  s-d, setup-dirs - Set up the directory structure.")
            print("  exit, e, quit, q - Exit the tool.")
        if command in ["s-d", "setup-dirs"]:
            setup_dir_structure()