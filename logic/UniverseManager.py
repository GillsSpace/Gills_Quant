import os
import re
import json
import time as tm
import pandas as pd
import polars as pl
import schwabdev as sd

from pathlib import Path
from datetime import datetime
from tradingview_screener import Column, Query

from logic.lib_time import *
from logic.lib_clients import *

class UniverseManager:
    
    universe_folder_path = Path(__file__).resolve().parent.parent / 'universes'
    log_base_path = Path(__file__).resolve().parent.parent / 'logs'

    long_file_vars = ['name','type','sector','exchange','industry','close','average_volume_30d_calc','market_cap_basic']

    universe_dict = {
        "u00": {
            "in": [
                Column('average_volume_30d_calc') > 400,
                Column('type').isin(['stock', 'fund']),
                Column('exchange').isin(['AMEX', 'NASDAQ', 'NYSE']),
            ],
            "out": [
                Column('average_volume_30d_calc') > 100,
                Column('type').isin(['stock', 'fund']),
                Column('exchange').isin(['AMEX', 'NASDAQ', 'NYSE']),
            ]
        },
    }

    @staticmethod
    def _clean_symbol(s: str) -> str:
        if not isinstance(s, str):
            return s
        s = re.sub(r'/P(?!R)([^/]*)', r'/PR\1', s)
        return s.replace('.', '/')

    @staticmethod
    def _clean_ticker_df(df: pl.DataFrame) -> pl.DataFrame:
        if not df.is_empty() and 'name' in df.columns:
            df = df.with_columns(
                pl.col('name').map_elements(UniverseManager._clean_symbol, return_dtype=pl.String).alias('name')
            )
        return df

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
        raw_df = query.get_scanner_data()[1]
        df: pl.DataFrame = pl.from_pandas(raw_df)

        # Transform Names:
        df = UniverseManager._clean_ticker_df(df)

        # Sort alphabetically by name:
        df = df.sort('name')

        # Save CSV:
        UniverseManager.universe_folder_path.mkdir(parents=True, exist_ok=True)
        UniverseManager.log_base_path.mkdir(parents=True, exist_ok=True)
        df.write_csv(UniverseManager.universe_folder_path / f"{universe_code}_long.csv")
        df.select('name').write_csv(UniverseManager.universe_folder_path / f"{universe_code}.csv")

        current_month = datetime.now().strftime('%m_%Y')
        log_dir = UniverseManager.log_base_path / f"universe_change__{current_month}.log"
        with log_dir.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Freshly Generated universe {universe_code} with {len(df)} symbols.\n")

        UniverseManager._update_universe_status(universe_code, df)

    @staticmethod
    def _update_universe_status(universe_code: str, df: pl.DataFrame):
        try:
            from logic.lib_files import update_status
            symbol_count = len(df) if not df.is_empty() else 0
            stocks_count = len(df.filter(pl.col('type') == 'stock')) if not df.is_empty() and 'type' in df.columns else 0
            funds_count = len(df.filter(pl.col('type') == 'fund')) if not df.is_empty() and 'type' in df.columns else 0
            
            update_status({
                f"{universe_code}_symbols_count": symbol_count,
                f"{universe_code}_asset_breakdown": {
                    "stocks": stocks_count,
                    "funds": funds_count
                }
            })
        except Exception as e:
            print(f"Warning: Failed to update status.json for universe {universe_code}: {e}")

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
        new_stocks_df = UniverseManager._clean_ticker_df(pl.from_pandas(pd.DataFrame(in_result[1])))
        
        existing_df = pl.DataFrame()

        if os.path.exists(long_csv_path):
            existing_df = pl.read_csv(long_csv_path, null_values=[])
            
            # Align existing_df to the current schema (long_file_vars)
            for col in UniverseManager.long_file_vars:
                if col not in existing_df.columns:
                    existing_df = existing_df.with_columns(pl.lit(None).alias(col))
            existing_df = existing_df.select(UniverseManager.long_file_vars)
            
            out_query = (
                Query()
                .select(*UniverseManager.long_file_vars)
                .where(*out_conditions)
                .limit(10_000)
            )
            out_result = out_query.get_scanner_data()
            out_stocks_df = UniverseManager._clean_ticker_df(pl.from_pandas(pd.DataFrame(out_result[1])))
            
            if not out_stocks_df.is_empty() and not existing_df.is_empty():
                existing_out_stocks = existing_df.filter(pl.col('name').is_in(out_stocks_df['name'].to_list()))
            else:
                existing_out_stocks = pl.DataFrame(schema=existing_df.schema if not existing_df.is_empty() else None)
        else:
            existing_out_stocks = pl.DataFrame()
        
        dfs_to_concat = [d for d in [new_stocks_df, existing_out_stocks] if not d.is_empty()]
        if dfs_to_concat:
            combined_df = pl.concat(dfs_to_concat, how='diagonal')
        else:
            combined_df = pl.DataFrame()
        
        if not combined_df.is_empty():
            combined_df = combined_df.unique(subset=['name'], keep='first').sort('name')
        
        UniverseManager.universe_folder_path.mkdir(parents=True, exist_ok=True)
        if not combined_df.is_empty():
            combined_df.write_csv(long_csv_path)
            combined_df.select('name').write_csv(short_csv_path)
        else:
            pl.DataFrame(schema={col: pl.String for col in UniverseManager.long_file_vars}).write_csv(long_csv_path)
            pl.DataFrame(schema={'name': pl.String}).write_csv(short_csv_path)

        before_stocks_list = existing_df['name'].to_list() if not existing_df.is_empty() and 'name' in existing_df.columns else []
        after_stocks_list = combined_df['name'].to_list() if not combined_df.is_empty() and 'name' in combined_df.columns else []
        added_stocks = list(set(after_stocks_list) - set(before_stocks_list))
        removed_stocks = list(set(before_stocks_list) - set(after_stocks_list))

        current_month = datetime.now().strftime('%m_%Y')
        log_dir = UniverseManager.log_base_path / f"universe_change__{current_month}.log"
        UniverseManager.log_base_path.mkdir(parents=True, exist_ok=True)
        with log_dir.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if added_stocks:
                f.write(f"{timestamp} - Added {len(added_stocks)} symbols: {', '.join(str(s) for s in added_stocks)}\n")
            if removed_stocks:
                f.write(f"{timestamp} - Removed {len(removed_stocks)} symbols: {', '.join(str(s) for s in removed_stocks)}\n")
            if not added_stocks and not removed_stocks:
                f.write(f"{timestamp} - No changes in the universe.\n")

        print(f"\tNew Universe Size: {len(after_stocks_list)} symbols")
        if added_stocks:
            print(f"\tAdded {len(added_stocks)} symbols: {', '.join(str(s) for s in sorted(added_stocks))}")
        else:
            print(f"\tAdded 0 symbols")
        if removed_stocks:
            print(f"\tRemoved {len(removed_stocks)} symbols: {', '.join(str(s) for s in sorted(removed_stocks))}")
        else:
            print(f"\tRemoved 0 symbols")

        UniverseManager._update_universe_status(universe_code, combined_df)

        return {
            'new_size': len(after_stocks_list),
            'added': sorted(added_stocks),
            'removed': sorted(removed_stocks)
        }

    @staticmethod
    def return_universe_list(universe_code: str) -> list:
        """
        Returns the list of stock names in the given universe code.
        """
        csv_path = UniverseManager.universe_folder_path / f"{universe_code}.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Universe CSV file for code {universe_code} not found.")

        df = pl.read_csv(csv_path, null_values=[])
        if df.is_empty():
            return []
        return df['name'].to_list()
    
    @staticmethod
    def return_universe_quotes_raw(universe_code: str) -> tuple[pl.DataFrame|None, list]:
        """
        Return the raw DataFrame of stock quotes for a universe.

        Parameters
        ----------
        universe_code : str
            Universe identifier. It will be coerced to a stripped string.

        Returns
        -------
        tuple[Optional[pl.DataFrame], list[str]]
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

        quotes_df = pl.from_pandas(pd.json_normalize(list_of_quotes))
        return (quotes_df,error_messages)