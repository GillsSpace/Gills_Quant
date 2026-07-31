import json
from pathlib import Path

import schwabdev as sd


def create_client_schwab():
    """
    Creates and returns a Schwab client using schwabdev library.
    """
    creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
    tokens_db = Path(__file__).resolve().parent.parent / 'secrets' / 'tokens.db'
    with open(creds_file, 'r') as f:
        keys = json.load(f)

    return sd.Client(
        keys['schwab']['app_key'],
        keys['schwab']['app_secret'],
        tokens_db=str(tokens_db),
    )

def test_client_schwab():
    """
    Tests the Schwab client by fetching and printing account information to standard output.
    """
    client: sd.Client = create_client_schwab()
    accounts = client.account_details_all().json()
    print("Schwab Account Info:")
    for account in accounts:
        print(json.dumps(account, indent=4))