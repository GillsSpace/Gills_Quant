import os
import numpy as np
from pathlib import Path
from datetime import datetime
from logic.DataManager import DataManager

def get_dashboard_stats():
    """
    Calculates operational dashboard statistics for the homepage.
    """
    try:
        ds = DataManager.return_hot_store()
        if ds is None:
            return None
            
        stats = {}
        
        # 1. Database metrics
        stats['num_days'] = len(ds.day)
        stats['num_idents'] = len(ds.ident)
        stats['num_qVars'] = len(ds.qVar)
        stats['num_fVars'] = len(ds.fVar)
        stats['retention_days'] = DataManager.hot_data_retention_days
        
        # 2. Database size on disk
        total_size = 0
        db_path = DataManager.hot_path_db
        if os.path.exists(db_path):
            for dirpath, dirnames, filenames in os.walk(db_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
        stats['db_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        # 3. Find the latest day and time that actually has data
        found = False
        latest_day = "N/A"
        latest_time = "N/A"
        valid_tickers_count = 0
        
        for day in reversed(ds.day.values):
            # Load the day block (shape: time x ident x qVar)
            day_data = ds['5m'].sel(day=day).values
            if not np.all(np.isnan(day_data)):
                # Search backwards through times
                for t_idx in range(day_data.shape[0] - 1, -1, -1):
                    if not np.all(np.isnan(day_data[t_idx])):
                        latest_day = str(day)
                        latest_time = str(ds.time.values[t_idx])
                        
                        # Count symbols with valid data
                        valid_mask = ~np.all(np.isnan(day_data[t_idx]), axis=1)
                        valid_tickers_count = int(np.sum(valid_mask))
                        
                        # Count variables with valid data
                        valid_vars_mask = ~np.all(np.isnan(day_data[t_idx]), axis=0)
                        valid_vars_count = int(np.sum(valid_vars_mask))
                        
                        # Find Top Gainer, Top Loser, and Most Active Ticker in this slice
                        top_gainer_symbol = "N/A"
                        top_gainer_val = 0.0
                        top_loser_symbol = "N/A"
                        top_loser_val = 0.0
                        most_active_symbol = "N/A"
                        most_active_vol = 0.0
                        
                        qvars_list = list(ds.qVar.values)
                        
                        # 1. Top Gainer & Top Loser (based on mark percent change or net percent change)
                        change_var = next((v for v in ['quote.markPercentChange', 'quote.netPercentChange'] if v in qvars_list), None)
                        if change_var:
                            var_idx = qvars_list.index(change_var)
                            change_values = day_data[t_idx, :, var_idx]
                            valid_indices = np.where(~np.isnan(change_values))[0]
                            if len(valid_indices) > 0:
                                max_idx = valid_indices[np.argmax(change_values[valid_indices])]
                                top_gainer_symbol = str(ds.ident.values[max_idx])
                                top_gainer_val = round(float(change_values[max_idx]), 2)

                                min_idx = valid_indices[np.argmin(change_values[valid_indices])]
                                top_loser_symbol = str(ds.ident.values[min_idx])
                                top_loser_val = round(float(change_values[min_idx]), 2)
                                
                        # 2. Most Active (based on total volume)
                        if 'quote.totalVolume' in qvars_list:
                            vol_idx = qvars_list.index('quote.totalVolume')
                            vol_values = day_data[t_idx, :, vol_idx]
                            valid_vol_indices = np.where(~np.isnan(vol_values))[0]
                            if len(valid_vol_indices) > 0:
                                max_vol_idx = valid_vol_indices[np.argmax(vol_values[valid_vol_indices])]
                                most_active_symbol = str(ds.ident.values[max_vol_idx])
                                raw_vol = float(vol_values[max_vol_idx])
                                if raw_vol >= 1_000_000:
                                    most_active_vol = f"{raw_vol / 1_000_000:.1f}M"
                                elif raw_vol >= 1_000:
                                    most_active_vol = f"{raw_vol / 1_000:.0f}K"
                                else:
                                    most_active_vol = str(int(raw_vol))
                        
                        found = True
                        break
            if found:
                break
                
        stats['last_pull_day'] = latest_day
        stats['last_pull_time'] = latest_time
        stats['last_pull_tickers'] = valid_tickers_count
        stats['last_pull_vars'] = valid_vars_count
        stats['top_gainer_symbol'] = top_gainer_symbol
        stats['top_gainer_val'] = top_gainer_val
        stats['top_loser_symbol'] = top_loser_symbol
        stats['top_loser_val'] = top_loser_val
        stats['most_active_symbol'] = most_active_symbol
        stats['most_active_vol'] = most_active_vol
        
        # 4. Status Indicator (Health status)
        if latest_day != "N/A":
            last_pull_date = datetime.strptime(latest_day, "%Y-%m-%d")
            diff = datetime.now() - last_pull_date
            if diff.days <= 1:
                stats['db_health'] = "Healthy"
                stats['db_health_color'] = "#10b981"  # Emerald green
            else:
                stats['db_health'] = "Out of Date"
                stats['db_health_color'] = "#f59e0b"  # Amber
        else:
            stats['db_health'] = "Empty Database"
            stats['db_health_color'] = "#ef4444"  # Red
            
        # 5. Read status.json for status metrics
        status_file = Path(__file__).resolve().parent.parent.parent / 'status.json'
        status_data = {}
        if status_file.exists() and status_file.stat().st_size > 0:
            try:
                import json
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
            except Exception:
                status_data = {}
        stats['status_json'] = status_data

        # 6. Check secondary API statuses
        alpaca_status = "Disconnected"
        keys_file = Path(__file__).resolve().parent.parent.parent / 'secrets' / 'keys.json'
        if keys_file.exists():
            try:
                import json
                with open(keys_file, 'r') as f:
                    keys = json.load(f)
                if 'alpaca' in keys and keys['alpaca'].get('key'):
                    alpaca_status = "Active (Paper)"
            except Exception:
                pass

        edgar_filings = status_data.get('edgar_filings_symbols_yesterday', 0)
        edgar_status = f"Active ({edgar_filings} Filings Yday)"

        stats['api_alpaca_status'] = alpaca_status
        stats['api_edgar_status'] = edgar_status

        return stats
    except Exception as e:
        print(f"Error gathering stats: {e}")
        return None
