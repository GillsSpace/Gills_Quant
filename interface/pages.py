from flask import Flask, Blueprint, render_template
from datetime import datetime, timedelta, timezone
import os

from logic.lib_clients import *
from logic.DataManager import DataManager
from interface.libs.lib_dash import get_dashboard_stats

bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    sd_client = create_client_schwab()
    rt_delta = timedelta(seconds=sd_client.tokens._refresh_token_timeout) - (datetime.now(timezone.utc) - sd_client.tokens._refresh_token_issued)
    
    # Format the refresh token expiration countdown nicely with days, hours, and minutes
    days = rt_delta.days
    hours, remainder = divmod(rt_delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    if days > 0:
        token_str = f"Refresh token expires in: {days}d {hours}h {minutes}m"
    else:
        token_str = f"Refresh token expires in: {hours}h {minutes}m"
    
    stats = get_dashboard_stats()
    
    return render_template('pages/home.html', tokens=token_str, stats=stats)


import json
import time

_daily_stats_cache = {
    'days': None,
    'counts': None,
    'timestamp': 0
}

def get_daily_stats(ds):
    current_days = [str(d) for d in ds.day.values]
    now = time.time()
    if _daily_stats_cache['days'] == current_days and (now - _daily_stats_cache['timestamp']) < 300:
        return _daily_stats_cache['counts']
        
    # Calculate count of active tickers (any non-NaN fundamental variable) for each day
    counts = (~ds['1d'].isnull()).any(dim='fVar').sum(dim='ident').values
    counts_list = [int(c) for c in counts]
    
    _daily_stats_cache['days'] = current_days
    _daily_stats_cache['counts'] = counts_list
    _daily_stats_cache['timestamp'] = now
    
    return counts_list


@bp.route('/database')
def database():
    ds = DataManager.return_hot_store()
    
    if ds is not None:
        days = sorted([str(d) for d in ds.day.values], reverse=True)
        counts = get_daily_stats(ds)
        
        # Build mapping from day to count (ds.day.values is chronological)
        day_to_count = dict(zip([str(d) for d in ds.day.values], counts))
        
        # Chronological order for chart
        chronological_days = sorted(days)
        chart_counts = [day_to_count.get(d, 0) for d in chronological_days]
        
        # Reverse chronological for table
        table_data = [{'day': d, 'count': day_to_count.get(d, 0)} for d in days]
        
        db_info = {
            'days': days,
            'days_json': json.dumps(chronological_days),
            'counts_json': json.dumps(chart_counts),
            'table_data': table_data,
            'idents': sorted([str(i) for i in ds.ident.values]),
            'qvars': sorted([str(q) for q in ds.qVar.values]),
            'fvars': sorted([str(f) for f in ds.fVar.values]),
            'num_days': len(ds.day),
            'num_idents': len(ds.ident),
            'num_qvars': len(ds.qVar),
            'num_fvars': len(ds.fVar),
            'path': str(DataManager.hot_path_db),
            'retention': DataManager.hot_data_retention_days
        }
        
        # Calculate database size in GB
        total_size = 0
        db_path = DataManager.hot_path_db
        if os.path.exists(db_path):
            for dirpath, dirnames, filenames in os.walk(db_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
        db_info['db_size_gb'] = round(total_size / (1024 * 1024 * 1024), 3)
    else:
        db_info = None
        
    return render_template('pages/database.html', db_info=db_info)



