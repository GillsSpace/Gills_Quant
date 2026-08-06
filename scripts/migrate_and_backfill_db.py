import os
import sys
import json
import shutil
import calendar
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xarray as xr
import zarr

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from logic.DataManager import DataManager
from logic.lib_time import return_day_str_range, return_time_str_range

warnings.filterwarnings('ignore', category=UserWarning, module='zarr.*')

def fetch_bulk_alpaca_corporate_actions(symbols, start_day, end_day):
    """
    Fetches all corporate actions for a list of symbols across a DATE RANGE in bulk batches.
    Returns (splits_map, divs_map) keyed by (symbol, ex_date_str).
    """
    try:
        from alpaca.data.historical.corporate_actions import CorporateActionsClient
        from alpaca.data.requests import CorporateActionsRequest
        creds_file = Path(__file__).resolve().parent.parent / 'secrets' / 'keys.json'
        if not creds_file.exists():
            return {}, {}

        with open(creds_file, 'r') as f:
            keys = json.load(f)
        alpaca_key = keys['alpaca']['key']
        alpaca_secret = keys['alpaca']['secret']
        client = CorporateActionsClient(alpaca_key, alpaca_secret, raw_data=True)

        splits_map = {}  # (symbol, ex_date) -> ratio
        divs_map = {}    # (symbol, ex_date) -> amount

        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            request = CorporateActionsRequest(
                symbols=batch,
                start=start_day,
                end=end_day
            )
            raw_data = client.get_corporate_actions(request)
            if isinstance(raw_data, dict):
                # Cash dividends
                for div in raw_data.get('cash_dividends', []):
                    sym = div.get('symbol')
                    ex_date = str(div.get('ex_date') or '')[:10]
                    val = float(div.get('rate') or div.get('amount') or 0)
                    if sym and ex_date:
                        divs_map[(sym, ex_date)] = val
                # Forward splits
                for spl in raw_data.get('forward_splits', []):
                    sym = spl.get('symbol')
                    ex_date = str(spl.get('ex_date') or '')[:10]
                    new_rate = float(spl.get('new_rate') or 1)
                    old_rate = float(spl.get('old_rate') or 1)
                    ratio = new_rate / old_rate if old_rate != 0 else 1.0
                    if sym and ex_date:
                        splits_map[(sym, ex_date)] = ratio
                # Reverse splits
                for spl in raw_data.get('reverse_splits', []):
                    sym = spl.get('symbol')
                    ex_date = str(spl.get('ex_date') or '')[:10]
                    new_rate = float(spl.get('new_rate') or 1)
                    old_rate = float(spl.get('old_rate') or 1)
                    ratio = new_rate / old_rate if old_rate != 0 else 1.0
                    if sym and ex_date:
                        splits_map[(sym, ex_date)] = ratio
                # Stock dividends
                for div in raw_data.get('stock_dividends', []):
                    sym = div.get('symbol')
                    ex_date = str(div.get('ex_date') or '')[:10]
                    rate = float(div.get('rate') or 0)
                    if sym and ex_date and rate > 0:
                        splits_map[(sym, ex_date)] = 1.0 + rate

        return splits_map, divs_map
    except Exception as e:
        print(f"\tBulk Alpaca fetch failed: {e}")
        return {}, {}

def apply_corporate_actions_to_store(store_path, days, bulk_splits_map, bulk_divs_map):
    """
    Writes corporate actions into a Zarr store using bulk maps with local Schwab fallback.
    """
    ds_disk = xr.open_zarr(store_path, consolidated=True)
    existing_idents = ds_disk.ident.values.tolist()
    split_col_idx = DataManager.fundamental_fields.index('corporate.splitRatio')
    div_col_idx = DataManager.fundamental_fields.index('corporate.divAmount')
    div_ex_col_idx = DataManager.fundamental_fields.index('fundamental.divExDate')
    div_amt_col_idx = DataManager.fundamental_fields.index('fundamental.divPayAmount')
    ds_disk.close()  # Close read handle before region updates

    ds_disk = xr.open_zarr(store_path, consolidated=True)
    try:
        for day in days:
            if day not in ds_disk.day.values:
                continue

            day_idx = int(np.where(ds_disk.day.values == day)[0][0])
            current_fVar_slice = ds_disk['1d'].sel(day=day).values.copy()

            for idx, symbol in enumerate(existing_idents):
                # 1. Split check
                if (symbol, day) in bulk_splits_map:
                    current_fVar_slice[idx, split_col_idx] = bulk_splits_map[(symbol, day)]
                else:
                    current_fVar_slice[idx, split_col_idx] = np.nan

                # 2. Dividend check
                has_div = False
                if (symbol, day) in bulk_divs_map:
                    current_fVar_slice[idx, div_col_idx] = bulk_divs_map[(symbol, day)]
                    has_div = True
                else:
                    # Local Schwab fallback
                    try:
                        ex_date_num = current_fVar_slice[idx, div_ex_col_idx]
                        if not np.isnan(ex_date_num):
                            ex_date_int = int(ex_date_num)
                            day_clean = str(day)[:10].replace('-', '')
                            if day_clean.isdigit() and ex_date_int == int(day_clean):
                                amount = current_fVar_slice[idx, div_amt_col_idx]
                                if not np.isnan(amount) and amount > 0:
                                    current_fVar_slice[idx, div_col_idx] = amount
                                    has_div = True
                    except Exception:
                        pass

                    if not has_div:
                        current_fVar_slice[idx, div_col_idx] = np.nan

            region_to_update = {"day": slice(day_idx, day_idx + 1)}
            empty_fVar_shell = np.expand_dims(current_fVar_slice, axis=0)
            ds_to_write = xr.Dataset({'1d': (['day', 'ident', 'fVar'], empty_fVar_shell)})
            ds_to_write.to_zarr(store_path, region=region_to_update, mode='r+')
    finally:
        ds_disk.close()
    zarr.consolidate_metadata(str(store_path))

def main():
    data_dir = DataManager.data_path  # /home/willse/Gills_Quant/data
    backup_dir = data_dir.parent / 'data_backup_migration'

    print("==================================================")
    print("Starting Optimized Hot & Cold Zarr Database Migration")
    print("==================================================")

    # Clean stale temp backup or restore if needed
    if backup_dir.exists():
        if data_dir.exists():
            shutil.rmtree(backup_dir)

    print(f"Creating temporary safety backup at {backup_dir}...")
    shutil.copytree(data_dir, backup_dir, ignore=shutil.ignore_patterns('temp_*'))
    print("Safety backup created successfully.\n")

    try:
        # 2. Normalize and Deduplicate Cold Backup filenames
        print("--- Step 1: Normalizing Cold Store Filenames & Merging Duplicates ---")
        cold_files = list(DataManager.cold_path.glob("master_db_month__*.zarr"))
        for cf in cold_files:
            name = cf.name
            parts = name.replace(".zarr", "").split("__")
            if len(parts) == 2:
                ym = parts[1].split("_")
                if len(ym) == 2:
                    year_str, month_str = ym[0], ym[1]
                    if len(month_str) == 1:
                        correct_month_str = f"{int(month_str):02d}"
                        correct_path = DataManager.cold_path / f"master_db_month__{year_str}_{correct_month_str}.zarr"
                        if correct_path.exists() and cf.exists() and correct_path != cf:
                            print(f"\tMerging duplicate cold store {cf.name} into {correct_path.name}...")
                            ds_unpadded = xr.open_zarr(cf)
                            ds_padded = xr.open_zarr(correct_path)
                            ds_merged = ds_padded.combine_first(ds_unpadded)
                            
                            for v in ds_merged.variables:
                                ds_merged[v].encoding.clear()
                            ds_merged = ds_merged.chunk({'day': 1, 'time': -1, 'ident': 1000, 'qVar': -1, 'fVar': -1})

                            temp_merged = DataManager.cold_path / f"temp_merge_{correct_month_str}.zarr"
                            ds_merged.to_zarr(temp_merged, mode='w', consolidated=True)
                            zarr.consolidate_metadata(str(temp_merged))
                            ds_unpadded.close()
                            ds_padded.close()
                            DataManager._safe_replace_zarr(temp_merged, correct_path)
                            shutil.rmtree(cf)
                        elif cf.exists() and not correct_path.exists():
                            print(f"\tRenaming {cf.name} -> {correct_path.name}")
                            shutil.move(cf, correct_path)

        # 3. Hot Database Migration (180 Days)
        print("\n--- Step 2: Migrating Hot Database (180-Day Coverage & Corporate Actions) ---")
        if DataManager.hot_path_db.exists():
            today = datetime.now().date()
            start_date_str = (today - timedelta(days=179)).strftime("%Y-%m-%d")
            end_date_str = today.strftime("%Y-%m-%d")
            target_180_days = return_day_str_range(start_date_str, end_date_str)

            ds_hot = xr.open_zarr(DataManager.hot_path_db)
            existing_days = set(str(d) for d in ds_hot.day.values)
            hot_idents = ds_hot.ident.values.tolist()
            ds_hot.close()

            missing_days = [d for d in target_180_days if d not in existing_days]
            print(f"\tHot DB target days: 180. Currently missing: {len(missing_days)} days.")
            for d in missing_days:
                DataManager.add_db_day_shell(d)

            # Ensure dataset is chronologically sorted and uniformly chunked
            ds_hot = xr.open_zarr(DataManager.hot_path_db)
            if list(ds_hot.day.values) != target_180_days:
                print("\tSorting Hot DB day dimension chronologically and rechunking...")
                ds_sorted = ds_hot.sortby('day')
                for v in ds_sorted.variables:
                    ds_sorted[v].encoding.clear()
                ds_sorted = ds_sorted.chunk({'day': 1, 'time': -1, 'ident': 1000, 'qVar': -1, 'fVar': -1})
                temp_hot = DataManager.hot_path / "temp_master_db.zarr"
                ds_sorted.to_zarr(temp_hot, mode='w', consolidated=True)
                zarr.consolidate_metadata(str(temp_hot))
                ds_hot.close()
                DataManager._safe_replace_zarr(temp_hot, DataManager.hot_path_db)
            else:
                ds_hot.close()

            print("\tFetching bulk Alpaca corporate actions for 180-day window...")
            bulk_splits_map, bulk_divs_map = fetch_bulk_alpaca_corporate_actions(hot_idents, start_date_str, end_date_str)
            print(f"\tFound {len(bulk_splits_map)} splits and {len(bulk_divs_map)} dividends in Alpaca.")

            print("\tWriting corporate actions to Hot DB...")
            apply_corporate_actions_to_store(DataManager.hot_path_db, target_180_days, bulk_splits_map, bulk_divs_map)
            print("\tHot DB migration completed.")

        # 4. Cold Database Migration (Full Month Length + Corporate Actions)
        print("\n--- Step 3: Migrating Cold Storage Databases (Full Month Coverage) ---")
        cold_files = list(DataManager.cold_path.glob("master_db_month__*.zarr"))
        for cold_path in cold_files:
            name = cold_path.name
            parts = name.replace(".zarr", "").split("__")
            if len(parts) == 2:
                ym = parts[1].split("_")
                year, month = int(ym[0]), int(ym[1])
                _, num_days = calendar.monthrange(year, month)
                month_str = f"{month:02d}"

                print(f"\tProcessing cold store {name} (Year: {year}, Month: {month_str}, Target Days: {num_days})...")

                month_start = f"{year}-{month_str}-01"
                month_end = f"{year}-{month_str}-{num_days:02d}"
                target_month_days = return_day_str_range(month_start, month_end)

                ds_cold = xr.open_zarr(cold_path)
                existing_cold_days = set(str(d) for d in ds_cold.day.values)
                cold_idents = ds_cold.ident.values.tolist()
                cold_time_coords = ds_cold.time.values
                missing_cold_days = sorted([d for d in target_month_days if d not in existing_cold_days])
                ds_cold.close()

                if missing_cold_days:
                    print(f"\t\tBackfilling {len(missing_cold_days)} missing days into {name}...")
                    empty_5m = np.full((len(missing_cold_days), len(cold_time_coords), len(cold_idents), len(DataManager.quote_fields)), np.nan)
                    empty_1d = np.full((len(missing_cold_days), len(cold_idents), len(DataManager.fundamental_fields)), np.nan)

                    batch_day_shells = xr.Dataset(
                        data_vars={
                            '5m': (['day', 'time', 'ident', 'qVar'], empty_5m),
                            '1d': (['day', 'ident', 'fVar'], empty_1d),
                        },
                        coords={
                            'day': missing_cold_days,
                            'time': cold_time_coords,
                            'ident': cold_idents,
                            'qVar': DataManager.quote_fields,
                            'fVar': DataManager.fundamental_fields,
                        }
                    )

                    ds_cold_curr = xr.open_zarr(cold_path)
                    combined_cold = xr.concat([ds_cold_curr, batch_day_shells], dim='day')
                    combined_cold = combined_cold.sortby('day')
                    for v in combined_cold.variables:
                        combined_cold[v].encoding.clear()
                    combined_cold = combined_cold.chunk({'day': 1, 'time': -1, 'ident': 1000, 'qVar': -1, 'fVar': -1})

                    temp_cold = DataManager.cold_path / f"temp_{name}"
                    combined_cold.to_zarr(temp_cold, mode='w', consolidated=True)
                    zarr.consolidate_metadata(str(temp_cold))
                    ds_cold_curr.close()
                    DataManager._safe_replace_zarr(temp_cold, cold_path)

                # Fetch bulk corporate actions for this month
                print(f"\t\tFetching bulk Alpaca corporate actions for {name}...")
                cold_splits_map, cold_divs_map = fetch_bulk_alpaca_corporate_actions(cold_idents, month_start, month_end)

                # Write corporate actions to cold store
                print(f"\t\tWriting corporate actions to {name}...")
                apply_corporate_actions_to_store(cold_path, target_month_days, cold_splits_map, cold_divs_map)

        # 5. Integrity Verification
        print("\n--- Step 4: Database Integrity Validation ---")
        if DataManager.hot_path_db.exists():
            ds_hot = xr.open_zarr(DataManager.hot_path_db, consolidated=True)
            print(f"\tHot DB Validated: {len(ds_hot.day)} days, {len(ds_hot.ident)} idents.")
            ds_hot.close()

        for cold_path in DataManager.cold_path.glob("master_db_month__*.zarr"):
            ds_cold = xr.open_zarr(cold_path, consolidated=True)
            print(f"\tCold Store {cold_path.name} Validated: {len(ds_cold.day)} days, {len(ds_cold.ident)} idents.")
            ds_cold.close()

        # 6. Safety Backup Cleanup
        print(f"\nMigration successful. Removing temporary safety backup at {backup_dir}...")
        shutil.rmtree(backup_dir)
        print("==================================================")
        print("Migration Completed Successfully!")
        print("==================================================")

    except Exception as e:
        print(f"\nERROR: Migration failed with exception: {e}")
        print("Rolling back databases from safety backup...")
        if backup_dir.exists():
            if data_dir.exists():
                shutil.rmtree(data_dir)
            shutil.copytree(backup_dir, data_dir)
            shutil.rmtree(backup_dir)
            print("Rollback completed. Original database restored.")
        raise e

if __name__ == "__main__":
    main()
