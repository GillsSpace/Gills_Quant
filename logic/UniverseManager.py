import os
import json
import time as tm
import pandas as pd
import schwabdev as sd

from pathlib import Path
from datetime import datetime
from tradingview_screener import Column, Query

from logic.lib_time import *
from logic.lib_clients import *

class UniverseManager:
    
    universe_folder_path = Path(__file__).resolve().parent.parent / 'universes'
    log_base_path = Path(__file__).resolve().parent.parent / 'logs'

    long_file_vars = ['name','sector','exchange','industry','close','volume|1W','market_cap_basic']

    universe_dict = {
        "u00": {
            "in": [
                Column('volume|1W') > 2_000,
                Column('type') == 'stock',
                Column('exchange').isin(['AMEX', 'NASDAQ', 'NYSE']),
            ],
            "out": [
                Column('volume|1W') > 500,
                Column('type') == 'stock',
                Column('exchange').isin(['AMEX', 'NASDAQ', 'NYSE']),
            ]
        },
    }

    @staticmethod
    def gen_csv(universe_code: str):
        """
        Generates a CSV file for the given universe code based on its inclusion criteria. Logs any changes.
        """
        if universe_code not in UniverseManager.universe_dict:
            raise ValueError(f"Universe code {universe_code} not found in universe dictionary.")

        universe_criteria = UniverseManager.universe_dict[universe_code]["in"]
        query = (
            Query()
            .select(*UniverseManager.long_file_vars)
            .where(*universe_criteria)
            .limit(10_000)
        )
        df: pd.DataFrame = query.get_scanner_data()[1]

        # Transform Names:
        df['name'] = df['name'].str.replace(r'/P([^/]*)', r'/PR\1', regex=True)  # */P* -> */PR*
        df['name'] = df['name'].str.replace(r'\.', '/', regex=True)  # *.* -> */*

        # Save CSV:
        df.to_csv(UniverseManager.universe_folder_path / f"{universe_code}_long.csv", index=False)
        df['name'].to_csv(UniverseManager.universe_folder_path / f"{universe_code}.csv", index=False)

        current_month = datetime.now().strftime('%m_%Y')
        log_dir = UniverseManager.log_base_path / f"universe_change__{current_month}.log"
        with log_dir.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Freshly Generated universe {universe_code} with {len(df)} stocks.\n")

    @staticmethod
    def regen_csv(universe_code: str):
        """
        Regenerates the CSV files for the given universe code.
        """
        long_csv_path = UniverseManager.universe_folder_path / f"{universe_code}_long.csv"
        short_csv_path = UniverseManager.universe_folder_path / f"{universe_code}.csv"

        in_conditions = UniverseManager.universe_dict[universe_code]['in']
        out_conditions = UniverseManager.universe_dict[universe_code]['out']

        in_query = (
            Query()
            .select(*UniverseManager.long_file_vars)
            .where(*in_conditions)
            .limit(10_000)
        )
        in_result = in_query.get_scanner_data()
        new_stocks_df = pd.DataFrame(in_result[1])

        print("new_stocks_df:", new_stocks_df)
        
        # Transform names in new_stocks_df
        if not new_stocks_df.empty:
            new_stocks_df['name'] = new_stocks_df['name'].str.replace(r'/P([^/]*)', r'/PR\1', regex=True)  # */P* -> */PR*
            new_stocks_df['name'] = new_stocks_df['name'].str.replace(r'\.', '/', regex=True)  # *.* -> */*
        
        existing_df = pd.DataFrame()

        if os.path.exists(long_csv_path):
            existing_df = pd.read_csv(long_csv_path)
            
            out_query = (
                Query()
                .select(*UniverseManager.long_file_vars)
                .where(*out_conditions)
                .limit(10_000)
            )
            out_result = out_query.get_scanner_data()
            out_stocks_df = pd.DataFrame(out_result[1])
            
            # Transform names in out_stocks_df
            if not out_stocks_df.empty:
                out_stocks_df['name'] = out_stocks_df['name'].str.replace(r'/P([^/]*)', r'/PR\1', regex=True)  # */P* -> */PR*
                out_stocks_df['name'] = out_stocks_df['name'].str.replace(r'\.', '/', regex=True)  # *.* -> */*
            
            if not out_stocks_df.empty and not existing_df.empty:
                existing_out_stocks = existing_df[existing_df['name'].isin(out_stocks_df['name'])]
            else:
                existing_out_stocks = pd.DataFrame()
        else:
            existing_out_stocks = pd.DataFrame()
        
        if not existing_out_stocks.empty and not new_stocks_df.empty:
            combined_df = pd.concat([new_stocks_df, existing_out_stocks], ignore_index=True)
        elif not new_stocks_df.empty:
            combined_df = new_stocks_df
        elif not existing_out_stocks.empty:
            combined_df = existing_out_stocks
        else:
            combined_df = pd.DataFrame()
        
        if not combined_df.empty:
            combined_df = combined_df.drop_duplicates(subset=['name'], keep='first')
        
        if not combined_df.empty:
            combined_df.to_csv(long_csv_path, index=False)
            combined_df['name'].to_csv(short_csv_path, index=False)
        else:
            pd.DataFrame(columns=["name", "sector", "exchange", "industry"]).to_csv(long_csv_path, index=False)
            pd.DataFrame(columns=["name"]).to_csv(short_csv_path, index=False)

        before_stocks_list = existing_df['name'].tolist() if not existing_df.empty else []
        after_stocks_list = combined_df['name'].tolist() if not combined_df.empty else []
        added_stocks = list(set(after_stocks_list) - set(before_stocks_list))
        removed_stocks = list(set(before_stocks_list) - set(after_stocks_list))

        current_month = datetime.now().strftime('%m_%Y')
        log_dir = UniverseManager.log_base_path / f"universe_change__{current_month}.log"
        UniverseManager.log_base_path.mkdir(parents=True, exist_ok=True)
        with log_dir.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if added_stocks:
                f.write(f"{timestamp} - Added stocks: {', '.join(added_stocks)}\n")
            if removed_stocks:
                f.write(f"{timestamp} - Removed stocks: {', '.join(removed_stocks)}\n")
            if not added_stocks and not removed_stocks:
                f.write(f"{timestamp} - No changes in the universe.\n")

    @staticmethod
    def return_universe_list(universe_code: str) -> list:
        """
        Returns the list of stock names in the given universe code.
        """
        csv_path = UniverseManager.universe_folder_path / f"{universe_code}.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Universe CSV file for code {universe_code} not found.")

        df = pd.read_csv(csv_path, keep_default_na=False)
        if df.empty:
            return []
        return df['name'].tolist()
    
    @staticmethod
    def return_universe_quotes_raw(universe_code: str) -> tuple[pd.DataFrame|None, list]:
        """
        Return the raw DataFrame of stock quotes for a universe.

        Parameters
        ----------
        universe_code : str
            Universe identifier. It will be coerced to a stripped string.

        Returns
        -------
        tuple[Optional[pd.DataFrame], list[str]]
            a tuple containing a DataFrame of stock quotes if successful, otherwise None,
            and a list of error messages encountered during the process.
        """
        # Normalize and validate input early
        universe_code = str(universe_code).strip()
        if not universe_code:
            return (None, ["Universe code must be a non-empty string."])
        error_messages = []

        tickers = UniverseManager.return_universe_list(universe_code)
        if not tickers:
            error_messages.append(f"Universe {universe_code} is empty.")
            return (None,error_messages)

        client = create_client_schwab()
        list_of_quotes = []
        batch_size = 500

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                quotes = client.quotes(batch)
                quotes_dict = quotes.json()

                list_of_quotes.extend([
                    {"ident":key, **value}
                    for key, value in quotes_dict.items()
                ])

                if i + batch_size < len(tickers):
                    tm.sleep(0.2)  # Rate limiting

            except Exception as e:
                error_messages.append(f"Error fetching quotes for batch {i//batch_size+1}: {str(e)}")
                continue

        if not list_of_quotes:
            error_messages.append(f"No quotes retrieved for universe {universe_code}.")
            return (None,error_messages)

        quotes_df = pd.json_normalize(list_of_quotes)
        return (quotes_df,error_messages)