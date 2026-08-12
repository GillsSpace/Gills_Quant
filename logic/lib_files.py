import json
from pathlib import Path
from datetime import datetime

def update_status(updates: dict):
    """
    Atomically updates status.json with the provided dictionary of key-value pairs.
    Attaches a 'last_updated_at' timestamp.
    """
    base_path = Path(__file__).resolve().parent.parent
    status_file = Path(base_path) / 'status.json'
    try:
        if status_file.exists() and status_file.stat().st_size > 0:
            with open(status_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data.update(updates)
        data['last_updated_at'] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error updating status.json: {e}")

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