import os
import zarr
import shutil
import warnings
import time as tm
import numpy as np
import xarray as xr
import pandas as pd

import json
from pathlib import Path
from pandas.api.types import CategoricalDtype
from datetime import datetime, timedelta, time, date

from logic.lib_time import *
from logic.UniverseManager import UniverseManager as UM

warnings.filterwarnings("ignore", message=".*Zarr format 3.*")
warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")

class DataManager:

    data_path = Path(__file__).resolve().parent.parent / 'data'
    hot_path = data_path / 'hot'
    cold_path = data_path / 'cold'
    hot_path_db = hot_path / 'master_db.zarr'
    log_path = Path(__file__).resolve().parent.parent / 'logs'

    master_universe = 'u00'
    hot_data_retention_days = 180

    Q_HTB_RATE = 'reference.htbRate'
    Q_HTB_QUANTITY = 'reference.htbQuantity'
    Q_ASK_PRICE_EXT = 'extended.askPrice'
    Q_ASK_SIZE_EXT = 'extended.askSize'
    Q_BID_PRICE_EXT = 'extended.bidPrice'
    Q_BID_SIZE_EXT = 'extended.bidSize'
    Q_LAST_PRICE_EXT = 'extended.lastPrice'
    Q_LAST_SIZE_EXT = 'extended.lastSize'
    Q_TRADE_TIME_EXT = 'extended.tradeTime'
    Q_TOTAL_VOLUME_EXT = 'extended.totalVolume'
    Q_QUOTE_TIME_EXT = 'extended.quoteTime'
    Q_MARK_EXT = 'extended.mark'
    Q_ASK_PRICE = 'quote.askPrice'
    Q_ASK_SIZE = 'quote.askSize'
    Q_ASK_TIME = 'quote.askTime'
    Q_BID_PRICE = 'quote.bidPrice'
    Q_BID_SIZE = 'quote.bidSize'
    Q_BID_TIME = 'quote.bidTime'
    Q_LAST_PRICE = 'quote.lastPrice'
    Q_LAST_SIZE = 'quote.lastSize'
    Q_TRADE_TIME = 'quote.tradeTime'
    Q_TOTAL_VOLUME = 'quote.totalVolume'
    Q_QUOTE_TIME = 'quote.quoteTime'
    Q_MARK = 'quote.mark'
    Q_52WEEK_HIGH = 'quote.52WeekHigh'
    Q_52WEEK_LOW = 'quote.52WeekLow'
    Q_HIGH_PRICE = 'quote.highPrice'
    Q_LOW_PRICE = 'quote.lowPrice'
    Q_MARK_CHANGE = 'quote.markChange'
    Q_MARK_PERCENT_CHANGE = 'quote.markPercentChange'
    Q_OPEN_PRICE = 'quote.openPrice'
    Q_NET_CHANGE = 'quote.netChange'
    Q_NET_PERCENT_CHANGE = 'quote.netPercentChange'
    Q_SECURITY_STATUS = 'quote.securityStatus'
    Q_POST_MARKET_CHANGE = 'quote.postMarketChange'
    Q_POST_MARKET_PERCENT_CHANGE = 'quote.postMarketPercentChange'

    F_ASSET_SUBTYPE = 'assetSubType'
    F_SSID = 'ssid'
    F_EXCHANGE = 'reference.exchange'
    F_AVG_10DAYS_VOLUME = 'fundamental.avg10DaysVolume'
    F_AVG_1YEAR_VOLUME = 'fundamental.avg1YearVolume'
    F_DECLARATION_DATE = 'fundamental.declarationDate'
    F_DIV_AMOUNT = 'fundamental.divAmount'
    F_DIV_YIELD = 'fundamental.divYield'
    F_DIV_EX_DATE = 'fundamental.divExDate'
    F_DIV_FREQ = 'fundamental.divFreq'
    F_DIV_PAY_DATE = 'fundamental.divPayDate'
    F_DIV_PAY_AMOUNT = 'fundamental.divPayAmount'
    F_EPS = 'fundamental.eps'
    F_LAST_EARNINGS_DATE = 'fundamental.lastEarningsDate'
    F_NEXT_DIV_EX_DATE = 'fundamental.nextDivExDate'
    F_NEXT_DIV_PAY_DATE = 'fundamental.nextDivPayDate'
    F_PE_RATIO = 'fundamental.peRatio'
    F_CLOSE_PRICE = 'quote.closePrice'

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
        'corporate.splitRatio',
        'corporate.divAmount',
    ]

    quote_securityStatus_dtype = CategoricalDtype(categories=[
        'Normal',
        'Halted',
        'Closed',
        'Unknown',
        'None',
    ], ordered=True)

    fundamental_assetSubType_dtype = CategoricalDtype(categories=[
        'ADR',
        'COE',
        'PRF',
        'UIT',
        'CEF',
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
    def _safe_replace_zarr(temp_path: Path, target_path: Path):
        """
        Safely replaces target_path with temp_path using a backup folder to prevent data loss.
        """
        bak_path = target_path.with_name(target_path.name + '.bak')
        if os.path.exists(bak_path):
            shutil.rmtree(bak_path)

        if os.path.exists(target_path):
            shutil.move(str(target_path), str(bak_path))

        try:
            shutil.move(str(temp_path), str(target_path))
            if os.path.exists(bak_path):
                shutil.rmtree(bak_path)
        except Exception as e:
            if os.path.exists(bak_path) and not os.path.exists(target_path):
                shutil.move(str(bak_path), str(target_path))
            raise e

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
    def _log_error_missed_idents(missed_idents):
        if not missed_idents:
            return

        current_month = datetime.now().strftime('%m_%Y')
        log_path = DataManager.log_path / f"missed_idents__{current_month}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open('a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Missed idents: {missed_idents}\n")

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

        if new_set.issubset(old_set) and not is_initial_creation:
            ds_shell = DataManager.create_empty_day_shell(day,existing_idents)
            # Clear encoding to avoid chunk conflicts with existing Zarr store
            for var in ds_shell.variables:
                ds_shell[var].encoding.clear()
            ds_shell.to_zarr(db_path, mode='a-', append_dim='day')
            zarr.consolidate_metadata(str(db_path))
            ds_disk.close()
            return
        
        # Keep existing order of idents in database, append new ones at the end (sorted alphabetically)
        new_idents = sorted(list(new_set - old_set))
        combined_idents = existing_idents + new_idents

        if os.path.exists(temp_db_path):
            shutil.rmtree(temp_db_path)

        # Lazy concatenation and reindexing using xarray/dask (extremely fast & constant memory)
        new_day_shell = DataManager.create_empty_day_shell(day, combined_idents)

        if not is_initial_creation:
            # Reindex the entire dataset to combined symbols lazily
            reindexed_ds = ds_disk.reindex({'ident': combined_idents}, fill_value=np.nan)
            combined_ds = xr.concat([reindexed_ds, new_day_shell], dim='day')
        else:
            combined_ds = new_day_shell

        # Rechunk uniformly to prevent Zarr chunk conflicts
        combined_ds = combined_ds.chunk({
            'day': 1,
            'time': -1,
            'ident': 1000,
            'qVar': -1,
            'fVar': -1,
        })

        # Clear encoding after chunking to ensure Zarr uses our chunk specification
        for var in combined_ds.variables:
            combined_ds[var].encoding.clear()

        # Write the entire concatenated dataset in one operation
        combined_ds.to_zarr(temp_db_path, mode='w', consolidated=False)
        zarr.consolidate_metadata(str(temp_db_path))

        if not is_initial_creation:
            ds_disk.close()

        # Safely replace old database with new one
        DataManager._safe_replace_zarr(temp_db_path, db_path)

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
        (raw_quotes_df, error_messages) = UM.return_universe_quotes_raw(DataManager.master_universe)
        if raw_quotes_df is None:
            print(f"Error: Failed to retrieve quotes for universe {DataManager.master_universe}. Messages: {error_messages}")
            return

        error_mask = raw_quotes_df['ident'] == 'errors'

        if error_mask.any() and 'invalid_symbols' in raw_quotes_df.columns:
            error_symbols = raw_quotes_df.loc[error_mask, 'invalid_symbols'].dropna().unique().tolist()
            DataManager._log_error_symbols(error_symbols)

        quotes_df = raw_quotes_df[~error_mask].copy()

        missing_cols = [col for col in DataManager.quote_fields if col not in quotes_df.columns]
        for col in missing_cols:
            quotes_df[col] = np.nan

        #Custom Data Cleaning:
        valid_cats = DataManager.quote_securityStatus_dtype.categories
        raw_status = quotes_df['quote.securityStatus'].dropna().unique()
        missed_securityStatus = [s for s in raw_status if s not in valid_cats]
        
        if len(missed_securityStatus) > 0:
            DataManager._log_error_categories(missed_securityStatus, 'quote.securityStatus')

        quotes_df['quote.securityStatus'] = quotes_df['quote.securityStatus'].astype(DataManager.quote_securityStatus_dtype).cat.codes.replace(-1, np.nan)
        quotes_df = quotes_df[['ident'] + DataManager.quote_fields].set_index('ident')

        ds_disk = None
        try:
            ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

            # If Day not in DB, add day shell
            if day not in ds_disk.day.values:
                ds_disk.close()
                ds_disk = None
                DataManager.add_db_day_shell(day)
                ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

            day_idx = int(np.where(ds_disk.day.values == day)[0][0])
            time_idx = int(np.where(ds_disk.time.values == time)[0][0])

            existing_idents = ds_disk.ident.values.tolist()
            empty_time_shell = np.full((1,1,len(existing_idents),len(DataManager.quote_fields)),np.nan)

            # 2. Log any Schwab API anomalies or new tickers awaiting tomorrow's shell
            missed_idxs = [ident for ident in quotes_df.index if ident not in existing_idents]
            if len(missed_idxs) > 0:
                DataManager._log_error_missed_idents(missed_idxs)

            # 3. Intersect and filter to prevent shape mismatch
            valid_idents = quotes_df.index.intersection(existing_idents)
            quotes_df = quotes_df.loc[valid_idents]
            
            ident_to_idx = {ident: idx for idx, ident in enumerate(existing_idents)}
            target_idxs = [ident_to_idx[ident] for ident in valid_idents]

            # 4. Insert safely
            empty_time_shell[0,0,target_idxs,:] = quotes_df.to_numpy()
            
            region_to_update = {
                "day": slice(day_idx, day_idx + 1),
                "time": slice(time_idx, time_idx + 1),
            }

            ds_to_write = xr.Dataset({
                '5m': (['day', 'time', 'ident', 'qVar'], empty_time_shell)
            })

            ds_to_write.to_zarr(DataManager.hot_path_db, region=region_to_update, mode='r+')
        finally:
            if ds_disk is not None:
                ds_disk.close()

    @staticmethod
    def save_fVar_data(day):
        """
        Saves fundamental variable data for a specific day into master database.
        """
        (raw_fundamentals_df, error_messages) = UM.return_universe_quotes_raw(DataManager.master_universe)
        if raw_fundamentals_df is None:
            print(f"Error: Failed to retrieve fundamentals for universe {DataManager.master_universe}. Messages: {error_messages}")
            return

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

        # 1. Find unmapped categories BEFORE transforming (ignoring normal NaNs)
        valid_subtypes = DataManager.fundamental_assetSubType_dtype.categories
        raw_subtypes = fundamentals_df['assetSubType'].dropna().unique()
        missed_asset_subtypes = [s for s in raw_subtypes if s not in valid_subtypes]
        
        valid_exchanges = DataManager.fundamental_exchange_dtype.categories
        raw_exchanges = fundamentals_df['reference.exchange'].dropna().unique()
        missed_exchanges = [s for s in raw_exchanges if s not in valid_exchanges]

        # 2. Log any missing strings
        if len(missed_asset_subtypes) > 0:
            DataManager._log_error_categories(missed_asset_subtypes, 'assetSubType')
        if len(missed_exchanges) > 0:
            DataManager._log_error_categories(missed_exchanges, 'reference.exchange')

        # 3. Safely transform to codes
        fundamentals_df['assetSubType'] = fundamentals_df['assetSubType'].astype(DataManager.fundamental_assetSubType_dtype).cat.codes.replace(-1,np.nan)
        fundamentals_df['reference.exchange'] = fundamentals_df['reference.exchange'].astype(DataManager.fundamental_exchange_dtype).cat.codes.replace(-1,np.nan)

        fundamentals_df = fundamentals_df[['ident']+DataManager.fundamental_fields].set_index('ident')

        ds_disk = None
        try:
            ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

            # If Day not in DB, add day shell
            if day not in ds_disk.day.values:
                ds_disk.close()
                ds_disk = None
                DataManager.add_db_day_shell(day)
                ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)

            day_idx = int(np.where(ds_disk.day.values == day)[0][0])

            existing_idents = ds_disk.ident.values.tolist()
            empty_fVar_shell = np.full((1,len(existing_idents),len(DataManager.fundamental_fields)),np.nan)

            missed_idxs = [ident for ident in fundamentals_df.index if ident not in existing_idents]
            if len(missed_idxs) > 0:
                DataManager._log_error_missed_idents(missed_idxs)

            valid_idents = fundamentals_df.index.intersection(existing_idents)
            fundamentals_df = fundamentals_df.loc[valid_idents]

            ident_to_idx = {ident: idx for idx, ident in enumerate(existing_idents)}
            target_idxs = [ident_to_idx[ident] for ident in valid_idents]

            empty_fVar_shell[0,target_idxs,:] = fundamentals_df.to_numpy()

            region_to_update = {
                "day": slice(day_idx, day_idx + 1),
            }

            ds_to_write = xr.Dataset({
                '1d': (['day', 'ident', 'fVar'], empty_fVar_shell)
            })

            ds_to_write.to_zarr(DataManager.hot_path_db, region=region_to_update, mode='r+')
        finally:
            if ds_disk is not None:
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
        month_str = f"{int(month):02d}"
        backup_path = DataManager.cold_path / f"master_db_month__{year}_{month_str}.zarr"
        temp_backup_path = DataManager.cold_path / f"temp_master_db_month__{year}_{month_str}.zarr"
        print(f"\tCreating cold backup for {year}-{month_str} at {backup_path} (overwrite_existing={overwrite_existing})")

        if os.path.exists(backup_path) and not overwrite_existing:
            return
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
                return

            # Select only the days for the requested month/year
            ds_subset = ds_disk.sel(day=days_to_keep)

            # Ensure clean encodings (prevents chunk/encoding issues on write)
            for var in ds_subset.variables:
                ds_subset[var].encoding.clear()

            # Clean up any stale temp folder if it exists
            if os.path.exists(temp_backup_path):
                shutil.rmtree(temp_backup_path)

            # Write to temp path first
            ds_subset.to_zarr(temp_backup_path, mode='w', consolidated=False)
            zarr.consolidate_metadata(str(temp_backup_path))

            # Swap safely
            DataManager._safe_replace_zarr(temp_backup_path, backup_path)

        except Exception as e:
            print(f"Error creating cold backup for {year}-{month:02d}: {e}")
            if os.path.exists(temp_backup_path):
                shutil.rmtree(temp_backup_path)
        finally:
            ds_disk.close()


    @staticmethod
    def retention_trim_db():
        """
        Removes data from hot database that is older than the retention period. Also removes any idents that have all NaN data across all days(have not been in the universe for full retention period).
        """
        if not os.path.exists(DataManager.hot_path_db):
            return None
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

            # Build list of days to keep and removed days
            days_to_keep = [
                d for d, p in zip(day_vals, parsed)
                if (current_date - p) <= retention_delta
            ]
            days_removed = [d for d in day_vals if d not in days_to_keep]

            stats = {
                'num_days_before': len(day_vals),
                'num_days_after': len(days_to_keep),
                'days_removed': days_removed,
            }

            if len(days_to_keep) == len(day_vals):
                return stats  # No old data to remove

            # Select only the days to keep
            ds_subset = ds_disk.sel(day=days_to_keep)

            idents_before = ds_disk.ident.values.tolist()
            
            # CRITICAL: Always keep idents that are in the current master universe
            current_universe = set(UM.return_universe_list(DataManager.master_universe))
            existing_idents = set(idents_before)
            
            # 1.3 fix: ensure we only keep/query active stocks that actually exist in the database coordinates
            current_universe_existing = current_universe.intersection(existing_idents)
            
            # 3.3 fix: exclude active stocks from the heavy NaN check
            inactive_idents = sorted(list(existing_idents - current_universe_existing))
            
            if inactive_idents:
                ds_inactive = ds_subset.sel(ident=inactive_idents)
                has_5m_data = ~ds_inactive['5m'].isnull().all(dim=['day', 'time', 'qVar'])
                has_1d_data = ~ds_inactive['1d'].isnull().all(dim=['day', 'fVar'])
                valid_idents_mask = has_5m_data | has_1d_data
                inactive_to_keep = ds_inactive.ident.values[valid_idents_mask.values].tolist()
            else:
                inactive_to_keep = []
            
            # Preserve original order of idents in the database to prevent chunk fragmentation
            set_to_keep = current_universe_existing.union(inactive_to_keep)
            idents_to_keep = [ident for ident in idents_before if ident in set_to_keep]

            idents_removed = [ident for ident in idents_before if ident not in idents_to_keep]

            # Re-select dataset with valid idents only
            ds_subset = ds_subset.sel(ident=idents_to_keep)

            # 1. Clear encodings to prevent old chunk metadata from interfering
            for var in ds_subset.variables:
                ds_subset[var].encoding.clear()

            # 2. UNIFY CHUNKS (The Fix)
            ds_subset = ds_subset.chunk({
                'day': 1,        # One day per chunk is usually best for time-series access
                'time': -1,      # All times in one chunk (288 is small)
                'ident': 1000,   # Keep group size at 1000 to avoid excessive files & slow I/O
                'qVar': -1       # All variables in one chunk
            })

            temp_db_path = DataManager.hot_path / 'temp_master_db.zarr'

            if os.path.exists(temp_db_path):
                shutil.rmtree(temp_db_path)

            # Use consolidated=True for better read performance later
            ds_subset.to_zarr(temp_db_path, mode='w', consolidated=True)
            zarr.consolidate_metadata(str(temp_db_path))

            ds_disk.close()

            # Safely replace old database with new one
            DataManager._safe_replace_zarr(temp_db_path, DataManager.hot_path_db)

            # Complete statistics
            stats.update({
                'num_idents_before': len(idents_before),
                'num_idents_after': len(idents_to_keep),
                'idents_removed': idents_removed,
            })

            return stats
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
        month_str = f"{int(month):02d}"
        backup_path = DataManager.cold_path / f"master_db_month__{year}_{month_str}.zarr"

        if not os.path.exists(backup_path):
            return None

        ds_disk = xr.open_zarr(backup_path, consolidated=True)
        return ds_disk
    
    @staticmethod
    def emergency_hot_restore():
        """
        Restores the hot database from the most recent cold backups within the hot window. Use with caution as this will overwrite the current hot database.
        """
        backup_files = list(DataManager.cold_path.glob("master_db_month__*.zarr"))
        if not backup_files:
            print("No cold backups found for restore.")
            return

        # Extract month/year from filenames and sort by date
        def extract_date(f):
            parts = f.stem.split('__')[-1].split('_')
            return int(parts[0]), int(parts[1])  # year, month

        backup_files.sort(key=extract_date, reverse=True)

        # Only restore last 7 monts then trims old data with retention trim to prevent restoring very old data
        max_files = min(7, len(backup_files))
        backup_files = backup_files[:max_files]

        # Open backup files and combine to restore as much recent data as possible
        opened_datasets = []
        try:
            for backup_file in backup_files:
                try:
                    ds_backup = xr.open_zarr(backup_file, consolidated=True)
                    opened_datasets.append(ds_backup)
                except Exception as e:
                    print(f"Error reading backup {backup_file}: {e}")

            if not opened_datasets:
                return

            if len(opened_datasets) == 1:
                combined_ds = opened_datasets[0]
            else:
                combined_ds = xr.combine_by_coords(opened_datasets, combine_attrs='override')

            temp_db_path = DataManager.hot_path / 'temp_master_db.zarr'

            if os.path.exists(temp_db_path):
                shutil.rmtree(temp_db_path)

            combined_ds.to_zarr(temp_db_path, mode='w', consolidated=True)
            zarr.consolidate_metadata(str(temp_db_path))

            # Safely replace old database with restored one
            DataManager._safe_replace_zarr(temp_db_path, DataManager.hot_path_db)
        finally:
            for ds in opened_datasets:
                ds.close()

        DataManager.retention_trim_db()  # Trim any old data if restore failed to prevent issues with old chunks

    @staticmethod
    def save_corporate_actions_for_day(day, use_alpaca=True):
        """
        Fetches corporate actions (splits and dividends) from Alpaca or local fallback
        and writes them into the Zarr database for the given day.
        """
        warnings.filterwarnings('ignore', category=UserWarning, module='zarr.*')

        ds_disk = None
        try:
            ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)
            if day not in ds_disk.day.values:
                print(f"Day {day} not found in database. Cannot write corporate actions.")
                return

            day_idx = int(np.where(ds_disk.day.values == day)[0][0])
            existing_idents = ds_disk.ident.values.tolist()
            
            # Read current values (shape: 1 x len(idents) x len(fVar))
            current_fVar_slice = ds_disk['1d'].sel(day=day).values.copy()
            
            # Indices of the corporate action columns
            split_col_idx = DataManager.fundamental_fields.index('corporate.splitRatio')
            div_col_idx = DataManager.fundamental_fields.index('corporate.divAmount')
            
            splits_map = {}
            divs_map = {}
            
            from logic.lib_adjustments import HAS_ALPACA
            if use_alpaca and HAS_ALPACA:
                try:
                    from logic.lib_adjustments import get_alpaca_corporate_actions
                    from alpaca.data.historical.corporate_actions import CorporateActionsClient
                    from alpaca.data.requests import CorporateActionsRequest
                    
                    creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
                    if creds_file.exists():
                        with open(creds_file, 'r') as f:
                            keys = json.load(f)
                        alpaca_key = keys['alpaca']['key']
                        alpaca_secret = keys['alpaca']['secret']
                        
                        client = CorporateActionsClient(alpaca_key, alpaca_secret, raw_data=True)
                        batch_size = 100
                        for i in range(0, len(existing_idents), batch_size):
                            batch = existing_idents[i:i+batch_size]
                            request = CorporateActionsRequest(
                                symbols=batch,
                                start=day,
                                end=day
                            )
                            raw_data = client.get_corporate_actions(request)
                            if isinstance(raw_data, dict):
                                target_day = str(day)[:10]
                                # Process dividends
                                for div in raw_data.get('cash_dividends', []):
                                    ex_date = str(div.get('ex_date') or '')[:10]
                                    if ex_date == target_day:
                                        val = float(div.get('rate') or div.get('amount') or 0)
                                        divs_map[div.get('symbol')] = val
                                # Process forward splits
                                for spl in raw_data.get('forward_splits', []):
                                    ex_date = str(spl.get('ex_date') or '')[:10]
                                    if ex_date == target_day:
                                        new_rate = float(spl.get('new_rate') or 1)
                                        old_rate = float(spl.get('old_rate') or 1)
                                        splits_map[spl.get('symbol')] = new_rate / old_rate if old_rate != 0 else 1.0
                                # Process reverse splits
                                for spl in raw_data.get('reverse_splits', []):
                                    ex_date = str(spl.get('ex_date') or '')[:10]
                                    if ex_date == target_day:
                                        new_rate = float(spl.get('new_rate') or 1)
                                        old_rate = float(spl.get('old_rate') or 1)
                                        splits_map[spl.get('symbol')] = new_rate / old_rate if old_rate != 0 else 1.0
                                # Process stock dividends (dilutes price like forward split)
                                for div in raw_data.get('stock_dividends', []):
                                    ex_date = str(div.get('ex_date') or '')[:10]
                                    if ex_date == target_day:
                                        rate = float(div.get('rate') or 0)
                                        if rate > 0:
                                            splits_map[div.get('symbol')] = 1.0 + rate
                except Exception as e:
                    print(f"Alpaca query failed for {day}: {e}")

            div_ex_col_idx = DataManager.fundamental_fields.index('fundamental.divExDate')
            div_amt_col_idx = DataManager.fundamental_fields.index('fundamental.divPayAmount')
            
            for idx, symbol in enumerate(existing_idents):
                has_action = False
                if symbol in splits_map:
                    current_fVar_slice[idx, split_col_idx] = splits_map[symbol]
                    has_action = True
                else:
                    current_fVar_slice[idx, split_col_idx] = np.nan
                    
                if symbol in divs_map:
                    current_fVar_slice[idx, div_col_idx] = divs_map[symbol]
                    has_action = True
                else:
                    # Fallback to local Schwab data:
                    try:
                        ex_date_num = current_fVar_slice[idx, div_ex_col_idx]
                        if not np.isnan(ex_date_num):
                            ex_date_int = int(ex_date_num)
                            day_clean = str(day)[:10].replace('-', '')
                            if day_clean.isdigit() and ex_date_int == int(day_clean):
                                amount = current_fVar_slice[idx, div_amt_col_idx]
                                if not np.isnan(amount) and amount > 0:
                                    current_fVar_slice[idx, div_col_idx] = amount
                                    has_action = True
                    except:
                        pass
                    
                    if not has_action:
                        current_fVar_slice[idx, div_col_idx] = np.nan

            region_to_update = {
                "day": slice(day_idx, day_idx + 1),
            }

            # Reshape slice to match (1, len(idents), len(fVar))
            empty_fVar_shell = np.expand_dims(current_fVar_slice, axis=0)

            ds_to_write = xr.Dataset({
                '1d': (['day', 'ident', 'fVar'], empty_fVar_shell)
            })

            ds_to_write.to_zarr(DataManager.hot_path_db, region=region_to_update, mode='r+')
        finally:
            if ds_disk is not None:
                ds_disk.close()

    @staticmethod
    def backfill_missing_days_and_corporate_actions():
        """
        Scans for missing day shells in current and previous month, inserts empty day shells,
        and fetches corporate actions for backfilled days.
        """
        if not os.path.exists(DataManager.hot_path_db):
            return

        ds_disk = xr.open_zarr(DataManager.hot_path_db, consolidated=True)
        try:
            day_vals = [str(d) for d in ds_disk.day.values]
            if not day_vals:
                return

            current_date = datetime.now().date()
            first_curr = current_date.replace(day=1)
            last_prev = first_curr - timedelta(days=1)
            start_date_str = last_prev.replace(day=1).strftime("%Y-%m-%d")
            end_date_str = current_date.strftime("%Y-%m-%d")

            expected_days = return_day_str_range(start_date_str, end_date_str)
            existing_days = set(day_vals)
            missing_days = [d for d in expected_days if d not in existing_days]
        finally:
            ds_disk.close()

        for day in missing_days:
            DataManager.add_db_day_shell(day)
            DataManager.save_corporate_actions_for_day(day, use_alpaca=True)
