import os
import zarr
import shutil
import warnings
import time as tm
import numpy as np
import xarray as xr
import pandas as pd

from pathlib import Path
from pandas.api.types import CategoricalDtype
from datetime import datetime, timedelta, time, date

from logic.lib_time import *
from logic.UniverseManager import UniverseManager as UM

class DataManager:

    data_path = Path(__file__).resolve().parent.parent / 'data'
    hot_path = data_path / 'hot'
    cold_path = data_path / 'cold'
    hot_path_db = hot_path / 'master_db.zarr'
    log_path = Path(__file__).resolve().parent.parent / 'logs'
    

    master_universe = 'u00'
    hot_data_retention_days = 120

    quote_fields = [
        'reference.htbRate',
        'reference.htbQuantity',
        'extended.askPrice',
        'extended.askSize',
        'extended.bidPrice',
        'extended.bidSize',
        'extended.lastPrice',
        'extended.lastSize',
        'extended.tradeTime',
        'extended.totalVolume',
        'extended.quoteTime',
        'extended.mark',
        'quote.askPrice',
        'quote.askSize',
        'quote.askTime',
        'quote.bidPrice',
        'quote.bidSize',
        'quote.bidTime',
        'quote.lastPrice',
        'quote.lastSize',
        'quote.tradeTime',
        'quote.totalVolume',
        'quote.quoteTime',
        'quote.mark',
        'quote.52WeekHigh',
        'quote.52WeekLow',
        'quote.highPrice',
        'quote.lowPrice',
        'quote.markChange',
        'quote.markPercentChange',
        'quote.openPrice',
        'quote.netChange',
        'quote.netPercentChange',
        'quote.securityStatus',
        'quote.postMarketChange',
        'quote.postMarketPercentChange',
    ]

    fundamental_fields = [
        'assetSubType',
        'ssid',
        'reference.exchange',
        'fundamental.avg10DaysVolume',
        'fundamental.avg1YearVolume',
        'fundamental.declarationDate',
        'fundamental.divAmount',
        'fundamental.divYield',
        'fundamental.divExDate',
        'fundamental.divFreq',
        'fundamental.divPayDate',
        'fundamental.divPayAmount',
        'fundamental.eps',
        'fundamental.lastEarningsDate',
        'fundamental.nextDivExDate',
        'fundamental.nextDivPayDate',
        'fundamental.peRatio',
        'quote.closePrice',
    ]

    quote_securityStatus_dtype = CategoricalDtype(categories=[
        'Normal',
        'Halted',
        'Closed',
        'Unknown',
    ], ordered=True)

    fundamental_assetSubType_dtype = CategoricalDtype(categories=[
        'ADR',
        'COE',
        'PRF',
    ], ordered=True)

    fundamental_exchange_dtype = CategoricalDtype(categories=[
        'N',
        'A',
        '9',
        'P',
        'Q',
    ], ordered=True)

    def __init__(self):
        for path in [self.hot_path, self.cold_path, self.log_path]:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _log_error_symbols(error_symbols):
        if not error_symbols:
            return

        current_month = datetime.now().strftime('%m_%Y')
        log_path = DataManager.log_path / f"symbol_errors__{current_month}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Errors for symbols: {error_symbols}\n")

    @staticmethod
    def _log_error_categories(missed_categories, category_type:str):
        if not missed_categories:
            return

        current_month = datetime.now().strftime('%m_%Y')
        log_path = DataManager.log_path / f"category_errors__{current_month}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Missed category for {category_type}: {missed_categories}\n")

    @staticmethod
    def create_empty_day_shell(day,idents):
        time_cords = pd.date_range(start='00:00', end='23:55', freq='5min').strftime('%H:%M').tolist()

        qVar_length = len(DataManager.quote_fields)
        fVar_length = len(DataManager.fundamental_fields)

        nan_qVar_array = np.full((1, len(time_cords), len(idents), qVar_length), np.nan)
        nan_fVar_array = np.full((1, len(idents), fVar_length), np.nan)

        coords = {
            'day': [day],
            'time': time_cords,
            'ident': idents,
            'qVar': DataManager.quote_fields,
            'fVar': DataManager.fundamental_fields,
        }

        data = {
            '5m': (['day', 'time', 'ident', 'qVar'], nan_qVar_array),
            '1d': (['day', 'ident', 'fVar'], nan_fVar_array)
        }

        return xr.Dataset(data, coords=coords)

    @staticmethod
    def add_db_day_shell(day,idents_for_day=None,is_initial_creation=False):
        """
        Adds a new day shell. If the symbols have changed, it rebuilds the entire
        database with a combined list of symbols.
        
        Args:
            day: Date string (YYYY-MM-DD format)
            new_idents: List of symbol identifiers. If None, fetches from UniverseManager
            is_initial_creation: Set True when creating database from scratch
        """
         
        # Suppress Zarr V3 specification warnings
        warnings.filterwarnings('ignore', message='.*Zarr V3 specification.*')
    
        temp_db_path = DataManager.hot_path / f'temp_master_db.zarr'
        db_path = DataManager.hot_path / 'master_db.zarr'

        if not idents_for_day:
            idents_for_day = UM.return_universe_list(DataManager.master_universe)

        if is_initial_creation:
            existing_idents = []
        else:
            ds_disk = xr.open_zarr(db_path, consolidated=True)
            existing_idents = ds_disk.ident.values.tolist()
            if day in ds_disk.day.values:
                return
        
        old_set = set(existing_idents)
        new_set = set(idents_for_day)

        if old_set == new_set and not is_initial_creation:
            ds_shell = DataManager.create_empty_day_shell(day,existing_idents)
            # Clear encoding to avoid chunk conflicts with existing Zarr store
            for var in ds_shell.variables:
                ds_shell[var].encoding.clear()
            ds_shell.to_zarr(db_path, mode='a-', append_dim='day')
            ds_disk.close()
            return
        
        combined_idents = sorted(list(old_set.union(new_set)))

        if os.path.exists(temp_db_path):
            shutil.rmtree(temp_db_path)

        # Process existing days one at a time and write to temp
        # This keeps memory usage constant regardless of database size
        if not is_initial_creation:
            for i, existing_day in enumerate(ds_disk.day.values):
                # Create empty shell with new symbol list
                reindexed_shell = DataManager.create_empty_day_shell(existing_day, combined_idents)
                # Reindex existing data to align with new symbol list (fills missing with NaN)
                reindexed_data = ds_disk.sel(day=[existing_day]).reindex({'ident': combined_idents}, fill_value=np.nan)
                # Merge reindexed data into the shell
                reindexed_shell.update(reindexed_data)

                # CRITICAL: Rechunk to uniform sizes to prevent Zarr chunk conflicts
                # This replaces irregular chunks created by reindex with regular chunks
                # Chunk dims: day=full, time=full for 5m var, ident=1000 to balance memory/performance
                reindexed_shell = reindexed_shell.chunk({
                    'day': -1,      # Don't chunk along day dimension (always 1)
                    'time': -1,     # Don't chunk along time dimension (always 288 for 5m)
                    'ident': 1000,  # Chunk identifiers in groups of 1000
                })

                # Clear encoding after chunking to ensure Zarr uses our chunk specification
                for var in reindexed_shell.variables:
                    reindexed_shell[var].encoding.clear()

                # First write creates the file, subsequent writes append
                mode = 'w' if i == 0 else 'a-'
                append_dim = None if i == 0 else 'day'

                # Write without consolidated=True to avoid metadata conflicts during rebuild
                # We'll consolidate once at the end for performance
                reindexed_shell.to_zarr(temp_db_path, mode=mode, append_dim=append_dim, 
                                    consolidated=False)

            ds_disk.close()

        # Add the new day to the temp database
        new_day_shell = DataManager.create_empty_day_shell(day, combined_idents)

        # Clear encoding to avoid chunk conflicts with existing Zarr store
        for var in new_day_shell.variables:
            new_day_shell[var].encoding.clear()

        # Determine mode and append dimension for new day
        # If this is initial creation, start fresh (mode='w')
        # Otherwise, append to existing (mode='a-')
        mode = 'w' if is_initial_creation else 'a-'
        append_dim = None if is_initial_creation else 'day'

        new_day_shell.to_zarr(temp_db_path, mode=mode, append_dim=append_dim, consolidated=False)

        # Consolidate metadata once after all writes for optimal read performance
        # This is much faster than consolidating after each day
        zarr.consolidate_metadata(str(temp_db_path))

        # Atomically replace old database with new one
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        shutil.move(temp_db_path, db_path)

    @staticmethod
    def create_new_db(initial_day):
        """
        Creates a new database starting from the initial_day with the master universe.
        """
        UM.gen_csv(DataManager.master_universe)
        idents = UM.return_universe_list(DataManager.master_universe)
        if os.path.exists(DataManager.hot_path_db):
            shutil.rmtree(DataManager.hot_path_db)
        DataManager.add_db_day_shell(initial_day, idents, is_initial_creation=True)

    @staticmethod
    def save_qVar_data(day,time):
        """
        Saves quote variable data for a specific day and time into master database.
        """
        (raw_quotes_df,_) = UM.return_universe_quotes_raw(DataManager.master_universe)

        error_mask = raw_quotes_df['ident'] == 'errors'

        if error_mask.any() and 'invalid_symbols' in raw_quotes_df.columns:
            error_symbols = raw_quotes_df.loc[error_mask, 'invalid_symbols'].dropna().unique().tolist()
            DataManager._log_error_symbols(error_symbols)

        quotes_df = raw_quotes_df[~error_mask].copy()

        missing_cols = [col for col in DataManager.quote_fields if col not in quotes_df.columns]
        for col in missing_cols:
            quotes_df[col] = np.nan

        #Custom Data Cleaning:
        quotes_df['quote.securityStatus'] = quotes_df['quote.securityStatus'].astype(DataManager.quote_securityStatus_dtype).cat.codes.replace(-1, np.nan)

        missed_securityStatus = quotes_df['quote.securityStatus'][quotes_df['quote.securityStatus'].isna()].unique()
        if len(missed_securityStatus) > 0:
            DataManager._log_error_categories(missed_securityStatus,'quote.securityStatus')

        quotes_df = quotes_df[['ident'] + DataManager.quote_fields].set_index('ident')
        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        # If Day not in DB, add day shell
        if day not in ds_disk.day.values:
            ds_disk.close()
            DataManager.add_db_day_shell(day)
            ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        day_idx = int(np.where(ds_disk.day.values == day)[0][0])
        time_idx = int(np.where(ds_disk.time.values == time)[0][0])

        existing_idents = ds_disk.ident.values.tolist()
        empty_time_shell = np.full((1,1,len(existing_idents),len(DataManager.quote_fields)),np.nan)

        target_idxs = [
            existing_idents.index(ident) for ident in quotes_df.index if ident in existing_idents
        ]

        empty_time_shell[0,0,target_idxs,:] = quotes_df.to_numpy()

        region_to_update = {
            "day": slice(day_idx, day_idx + 1),
            "time": slice(time_idx, time_idx + 1),
        }

        ds_to_write = xr.Dataset({
            '5m': (['day', 'time', 'ident', 'qVar'], empty_time_shell)
        })

        ds_to_write.to_zarr(DataManager.hot_path_db, region=region_to_update, mode='r+')
        ds_disk.close()

    @staticmethod
    def save_fVar_data(day):
        """
        Saves fundamental variable data for a specific day into master database.
        """
        (raw_fundamentals_df,_) = UM.return_universe_quotes_raw(DataManager.master_universe)

        error_mask = raw_fundamentals_df['ident'] == 'errors'

        if error_mask.any() and 'invalid_symbols' in raw_fundamentals_df.columns:
            error_symbols = raw_fundamentals_df.loc[error_mask, 'invalid_symbols'].dropna().unique().tolist()
            DataManager._log_error_symbols(error_symbols)

        fundamentals_df = raw_fundamentals_df[~error_mask].copy()

        missing_cols = [col for col in DataManager.fundamental_fields if col not in fundamentals_df.columns]
        for col in missing_cols:
            fundamentals_df[col] = np.nan

        #Custom Data Cleaning:
        fundamentals_df['fundamental.declarationDate'] = pd.to_numeric(fundamentals_df['fundamental.declarationDate'].str[:10].str.replace('-', ''), errors='coerce')
        fundamentals_df['fundamental.divExDate'] = pd.to_numeric(fundamentals_df['fundamental.divExDate'].str[:10].str.replace('-', ''), errors='coerce')
        fundamentals_df['fundamental.divPayDate'] = pd.to_numeric(fundamentals_df['fundamental.divPayDate'].str[:10].str.replace('-', ''), errors='coerce')
        fundamentals_df['fundamental.lastEarningsDate'] = pd.to_numeric(fundamentals_df['fundamental.lastEarningsDate'].str[:10].str.replace('-', ''), errors='coerce')
        fundamentals_df['fundamental.nextDivExDate'] = pd.to_numeric(fundamentals_df['fundamental.nextDivExDate'].str[:10].str.replace('-', ''), errors='coerce')
        fundamentals_df['fundamental.nextDivPayDate'] = pd.to_numeric(fundamentals_df['fundamental.nextDivPayDate'].str[:10].str.replace('-', ''), errors='coerce')

        fundamentals_df['assetSubType'] = fundamentals_df['assetSubType'].astype(DataManager.fundamental_assetSubType_dtype).cat.codes.replace(-1,np.nan)
        fundamentals_df['reference.exchange'] = fundamentals_df['reference.exchange'].astype(DataManager.fundamental_exchange_dtype).cat.codes.replace(-1,np.nan)
        missed_asset_subtypes = fundamentals_df['assetSubType'][fundamentals_df['assetSubType'].isna()].unique()
        missed_exchanges = fundamentals_df['reference.exchange'][fundamentals_df['reference.exchange'].isna()].unique()
        if len(missed_asset_subtypes) > 0:
            DataManager._log_error_categories(missed_asset_subtypes, 'assetSubType')
        if len(missed_exchanges) > 0:
            real_time = datetime.now()
            DataManager._log_error_categories(missed_exchanges, 'reference.exchange')

        fundamentals_df = fundamentals_df[['ident']+DataManager.fundamental_fields].set_index('ident')

        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        # If Day not in DB, add day shell
        if day not in ds_disk.day.values:
            ds_disk.close()
            DataManager.add_db_day_shell(day)
            ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        day_idx = int(np.where(ds_disk.day.values == day)[0][0])

        existing_idents = ds_disk.ident.values.tolist()
        empty_fVar_shell = np.full((1,len(existing_idents),len(DataManager.fundamental_fields)),np.nan)

        target_idxs = [
            existing_idents.index(ident) for ident in fundamentals_df.index if ident in existing_idents
        ]

        empty_fVar_shell[0,target_idxs,:] = fundamentals_df.to_numpy()

        region_to_update = {
            "day": slice(day_idx, day_idx + 1),
        }

        ds_to_write = xr.Dataset({
            '1d': (['day', 'ident', 'fVar'], empty_fVar_shell)
        })

        ds_to_write.to_zarr(DataManager.hot_path_db, region=region_to_update, mode='r+')
        ds_disk.close()
    
    @staticmethod
    def make_month_cold_backup(month, year, overwrite_existing=False):
        """
        Creates a cold backup of data in master that corresponds to the specified month and year.
        Args: 
            month: Integer month (1-12)
            year: Integer year (e.g., 2024)
            overwrite_existing: If True, overwrites existing backup for the month.
        """
        backup_path = DataManager.cold_path / f"master_db_month__{month}_{year}.zarr"

        if os.path.exists(backup_path) and not overwrite_existing:
            return
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        if not os.path.exists(DataManager.hot_path_db):
            return

        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        try:
            # Normalize day coordinate values to strings
            day_vals = [str(d) for d in ds_disk.day.values]

            # Parse into datetimes using pandas which handles multiple formats
            parsed = pd.to_datetime(day_vals, errors='coerce')

            # Build mask for requested month/year
            mask = (parsed.month == int(month)) & (parsed.year == int(year))
            days_to_keep = [d for d, m in zip(day_vals, mask) if m]

            if not days_to_keep:
                ds_disk.close()
                return

            # Select only the days for the requested month/year
            ds_subset = ds_disk.sel(day=days_to_keep)

            # Ensure clean encodings (prevents chunk/encoding issues on write)
            for var in ds_subset.variables:
                ds_subset[var].encoding.clear()

            ds_subset.to_zarr(backup_path, mode='w', consolidated=False)
            zarr.consolidate_metadata(str(backup_path))
        finally:
            ds_disk.close()

    @staticmethod
    def retention_trim_db():
        """
        Removes data from hot database that is older than the retention period. Also removes any idents that have all NaN data across all days(have not been in the universe for full retention period).
        """
        if not os.path.exists(DataManager.hot_path_db):
            return
        
        # Suppress Zarr V3 specification warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='zarr.*')

        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        try:
            current_date = datetime.now().date()
            retention_delta = timedelta(days=DataManager.hot_data_retention_days)

            # Normalize day coordinate values to strings
            day_vals = [str(d) for d in ds_disk.day.values]

            # Parse into datetimes using pandas which handles multiple formats
            parsed = pd.to_datetime(day_vals, errors='coerce').date

            # Build mask for days to keep
            days_to_keep = [
                d for d, p in zip(day_vals, parsed) 
                if (current_date - p) <= retention_delta
            ]

            if len(days_to_keep) == len(day_vals):
                return  # No old data to remove

            # Select only the days to keep
            ds_subset = ds_disk.sel(day=days_to_keep)

            # Identify idents with all NaN data across all days
            has_5m_data = ~ds_subset['5m'].isnull().all(dim=['day', 'time', 'qVar'])
            has_1d_data = ~ds_subset['1d'].isnull().all(dim=['day', 'fVar'])
            valid_idents_mask = has_5m_data | has_1d_data
            idents_to_keep = ds_subset.ident.values[valid_idents_mask.values].tolist()

            # Re-select dataset with valid idents only
            ds_subset = ds_subset.sel(ident=idents_to_keep)

            # 1. Clear encodings to prevent old chunk metadata from interfering
            for var in ds_subset.variables:
                ds_subset[var].encoding.clear()

            # 2. UNIFY CHUNKS (The Fix)
            # You can use 'auto' or specify a logical shape. 
            # Given your 5m data is (day, time, ident, qVar):
            ds_subset = ds_subset.chunk({
                'day': 1,        # One day per chunk is usually best for time-series access
                'time': -1,      # All times in one chunk (288 is small)
                'ident': 10,     # Small groups of symbols
                'qVar': -1       # All variables in one chunk
            })

            temp_db_path = DataManager.hot_path / 'temp_master_db.zarr'

            if os.path.exists(temp_db_path):
                shutil.rmtree(temp_db_path)

            # Use consolidated=True for better read performance later
            ds_subset.to_zarr(temp_db_path, mode='w', consolidated=True)
            zarr.consolidate_metadata(str(temp_db_path))

            ds_disk.close()

            # Atomically replace old database with new one
            shutil.rmtree(DataManager.hot_path_db)
            shutil.move(temp_db_path, DataManager.hot_path_db)
        finally:
            if 'ds_disk' in locals():
                ds_disk.close()

    @staticmethod
    def insert_backup(overwrite_existing_cold=False, overwrite_existing_hot=False, remove_existing=False):
        backup_path = Path(__file__).resolve().parent.parent / 'data_backup'
        hot_backup_path = backup_path / 'hot'
        cold_backup_path = backup_path / 'cold'

        if not backup_path.exists():
            return

        # 1. Total Replacement
        if remove_existing:
            if DataManager.data_path.exists():
                shutil.rmtree(DataManager.data_path)
            # Use copytree to keep the backup source intact for future use
            shutil.copytree(hot_backup_path, DataManager.hot_path)
            shutil.copytree(cold_backup_path, DataManager.cold_path)
            return # Exit early

        # 2. Selective Hot Overwrite
        if overwrite_existing_hot:
            if DataManager.hot_path.exists():
                shutil.rmtree(DataManager.hot_path)
            shutil.copytree(hot_backup_path, DataManager.hot_path)

        # 3. Cold Merging logic
        if cold_backup_path.exists():
            for src_root, _, files in os.walk(cold_backup_path):
                rel_root = Path(src_root).relative_to(cold_backup_path)
                dest_root = DataManager.cold_path / rel_root
                dest_root.mkdir(parents=True, exist_ok=True)

                for fname in files:
                    src_file = Path(src_root) / fname
                    dest_file = dest_root / fname

                    exists = dest_file.exists()
                    
                    if overwrite_existing_cold or not exists:
                        if exists:
                            dest_file.unlink()
                        shutil.copy2(src_file, dest_file)

    @staticmethod
    def create_backup():
        """
        Creates a backup of the current data (hot and cold) into data_backup directory.
        """
        backup_path = Path(__file__).resolve().parent.parent / 'data_backup'
        hot_backup_path = backup_path / 'hot'
        cold_backup_path = backup_path / 'cold'

        if backup_path.exists():
            shutil.rmtree(backup_path)

        shutil.copytree(DataManager.hot_path, hot_backup_path)
        shutil.copytree(DataManager.cold_path, cold_backup_path)
            
    @staticmethod
    def return_db_stats() -> dict:
        """
        Returns statistics about the master database.
        """
        if not os.path.exists(DataManager.hot_path_db):
            return None

        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

        try:
            num_days = len(ds_disk.day)
            num_idents = len(ds_disk.ident)
            num_qVars = len(ds_disk.qVar)
            num_fVars = len(ds_disk.fVar)
            all_days = ds_disk.day.values.tolist()
            current_universe_size = len(UM.return_universe_list(DataManager.master_universe))

            stats = {
                'num_days': num_days,
                'num_idents': num_idents,
                'num_qVars': num_qVars,
                'num_fVars': num_fVars,
                'current_universe_size': current_universe_size,
            }

            return stats
        finally:
            ds_disk.close()
        
    @staticmethod
    def gen_test_db(num_days:int, num_idents:int, start_date:str, num_full_nan_idents:int, random_day_skips:bool=True):
        """
        Gnerate a test database with specified parameters. Will overwrite existing test database if present.
        
        :param num_days: length of days dimension
        :type num_days: int
        :param num_idents: length of idents dimension
        :type num_idents: int
        :param start_date: Day to start the database from (YYYY-MM-DD format)
        :type start_date: str
        :param num_full_nan_idents: Include this many idents that have all NaN data across all days (to simulate removed symbols).
        :type num_full_nan_idents: int
        :param random_day_skips: If True, randomly skip some days to simulate missing data.
        :type random_day_skips: bool
        """
        from random import sample

        # Suppress Zarr V3 specification warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='zarr.*')

        db_path = DataManager.hot_path_db
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        date_range = pd.date_range(start=start_date, periods=num_days, freq='D')
        day_list = date_range.strftime('%Y-%m-%d').tolist()

        if random_day_skips:
            num_skips = max(1, num_days // 10)  # Skip ~10% of days
            skip_days = set(sample(day_list, num_skips))
            day_list = [d for d in day_list if d not in skip_days]

        idents = [f'SYM{i:05d}' for i in range(num_idents - num_full_nan_idents)]
        idents += [f'FULLNAN{i:05d}' for i in range(num_full_nan_idents)]

        num_valid = num_idents - num_full_nan_idents
        qVar_length = len(DataManager.quote_fields)
        fVar_length = len(DataManager.fundamental_fields)

        # Initialize with NaNs
        nan_qVar_array = np.full((len(day_list), 288, len(idents), qVar_length), np.nan)
        nan_fVar_array = np.full((len(day_list), len(idents), fVar_length), np.nan)

        # Indices for specific field types to make data look "plausible"
        # We'll target prices and volume to ensure the symbol is considered 'active'
        price_indices = [i for i, f in enumerate(DataManager.quote_fields) if 'Price' in f or 'mark' in f]
        vol_indices = [i for i, f in enumerate(DataManager.quote_fields) if 'Volume' in f or 'Size' in f]

        for s_idx in range(num_valid):
            # 1. Simulate a random walk for prices to fill 5m data
            # Start at a random price between 10 and 500
            start_px = np.random.uniform(10, 500)
            
            # Generate returns: Mean 0, 0.1% volatility per 5m bar
            returns = np.random.normal(loc=0, scale=0.001, size=(len(day_list), 288))
            price_path = start_px * np.exp(np.cumsum(returns))
            price_path = price_path.reshape(len(day_list), 288)

            for p_idx in price_indices:
                nan_qVar_array[:, :, s_idx, p_idx] = price_path
            
            for v_idx in vol_indices:
                # Random volumes between 100 and 10000
                nan_qVar_array[:, :, s_idx, v_idx] = np.random.randint(100, 10000, size=(len(day_list), 288))

            # 2. Add Fundamental data (Close Price)
            # Use the last price of the day for quote.closePrice in fVar
            if 'quote.closePrice' in DataManager.fundamental_fields:
                f_idx = DataManager.fundamental_fields.index('quote.closePrice')
                nan_fVar_array[:, s_idx, f_idx] = price_path[:, -1]

            # 3. Simulate missing data (randomly re-insert NaNs)
            # This masks ~15% of the "valid" data points to simulate dropped ticks
            mask = np.random.choice([True, False], size=nan_qVar_array[:, :, s_idx, :].shape, p=[0.15, 0.85])
            nan_qVar_array[:, :, s_idx, :][mask] = np.nan

        coords = {
            'day': day_list,
            'time': pd.date_range(start='00:00', end='23:55', freq='5min').strftime('%H:%M').tolist(),
            'ident': idents,
            'qVar': DataManager.quote_fields,
            'fVar': DataManager.fundamental_fields,
        }

        data = {
            '5m': (['day', 'time', 'ident', 'qVar'], nan_qVar_array),
            '1d': (['day', 'ident', 'fVar'], nan_fVar_array)
        }

        ds_test = xr.Dataset(data, coords=coords)

        ds_test.to_zarr(db_path, mode='w', consolidated=True)

    @staticmethod
    def return_hot_store() -> xr.Dataset:
        """
        Returns the hot master database as an xarray Dataset.
        """
        if not os.path.exists(DataManager.hot_path_db):
            return None

        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)
        return ds_disk
    
    @staticmethod
    def return_cold_store(month:int, year:int) -> xr.Dataset:
        """
        Returns the cold backup database for the specified month and year as an xarray Dataset.
        """
        backup_path = DataManager.cold_path / f"master_db_month__{month}_{year}.zarr"

        if not os.path.exists(backup_path):
            return None

        ds_disk = xr.open_zarr(backup_path, consolidated=True)
        return ds_disk