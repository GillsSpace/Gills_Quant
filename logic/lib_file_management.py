# Imports
import os
import sys
from pathlib import Path

def setup_dir_structure():
    """
    Sets up the directory structure for this project not found in Github (e.g. Data Folder).
    """
    base_path = Path(__file__).resolve().parent.parent
    dirs = ['data', 'logs', 'secrets', 'universes', 'tests']
    for dir_name in dirs:
        dir_path = Path(base_path) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"    Created directory: {dir_path}")

    file_path = Path(base_path) / 'status.json'
    if not file_path.exists():
        file_path.touch()

    with open(file_path, 'w') as f:
        f.write('{}')

    print(f"    Updated file: {file_path}")