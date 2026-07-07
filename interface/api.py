from flask import Blueprint, request
import numpy as np
import pandas as pd

from logic.DataManager import DataManager

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.route('/get-last-pull', methods=['GET'])
def get_last_pull():
    try:
        ds = DataManager.return_hot_store()
        if ds is None:
            return {"status": "error", "message": "Database not found"}, 404

        # Highly optimized backward search:
        # Instead of scanning the entire database across all history (slow I/O),
        # we check the latest days one by one, load that single day's data block,
        # and search backwards in memory for the most recent valid timestamp.
        found = False
        df = None
        
        for day in reversed(ds.day.values):
            # Load the day block (shape: time x ident x qVar)
            day_data = ds['5m'].sel(day=day).values
            if not np.all(np.isnan(day_data)):
                # Search backwards in memory through the time dimension of this day
                for t_idx in range(day_data.shape[0] - 1, -1, -1):
                    if not np.all(np.isnan(day_data[t_idx])):
                        # Build DataFrame directly from memory
                        df = pd.DataFrame(
                            day_data[t_idx],
                            index=ds.ident.values,
                            columns=ds.qVar.values
                        ).dropna(how='all')
                        found = True
                        break
            if found:
                break

        if not found or df is None:
            return {"status": "error", "message": "No data found in database"}, 404
            
        # Return as JSON
        return df.to_json(orient='split')
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@bp.route('/get-filtered-data', methods=['GET', 'POST'])
def get_filtered_data():
    try:
        ds = DataManager.return_hot_store()
        if ds is None:
            return {"status": "error", "message": "Database not found"}, 404

        # Parse request body (POST) or query params (GET)
        if request.method == 'POST':
            params = request.get_json(silent=True) or {}
        else:
            params = request.args

        # Helper to extract comma-separated string or list as list of strings
        def parse_param(keys):
            for k in keys:
                val = params.get(k)
                if val is not None:
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str):
                        return [v.strip() for v in val.split(',')]
                    return [str(val)]
            return None

        days = parse_param(['day', 'days'])
        times = parse_param(['time', 'times'])
        idents = parse_param(['ident', 'idents', 'symbol', 'symbols', 'ticker', 'tickers'])
        qvars = parse_param(['var', 'vars', 'variable', 'variables', 'qVar', 'qVars'])

        # Build selection dict
        selectors = {}
        
        if days:
            valid_days = [d for d in days if d in ds.day.values]
            if not valid_days:
                return {"status": "error", "message": f"Requested days {days} not found in database"}, 400
            selectors['day'] = valid_days
            
        if times:
            valid_times = [t for t in times if t in ds.time.values]
            if not valid_times:
                return {"status": "error", "message": f"Requested times {times} not found in database"}, 400
            selectors['time'] = valid_times
            
        if idents:
            valid_idents = [i for i in idents if i in ds.ident.values]
            if not valid_idents:
                return {"status": "error", "message": f"Requested symbols {idents} not found in database"}, 400
            selectors['ident'] = valid_idents
            
        if qvars:
            valid_qvars = [v for v in qvars if v in ds.qVar.values]
            if not valid_qvars:
                return {"status": "error", "message": f"Requested variables {qvars} not found in database"}, 400
            selectors['qVar'] = valid_qvars

        # Optimize selectors to use scalar selection when lists have size 1
        # This drops the dimension to keep the output clean (e.g. 2D instead of 4D)
        for dim in list(selectors.keys()):
            if len(selectors[dim]) == 1:
                selectors[dim] = selectors[dim][0]

        # Slice the dataset
        da_slice = ds['5m'].sel(**selectors)

        # Convert to DataFrame
        # If flat 2D (ident x qVar), keep index/columns flat.
        # Otherwise, return a tidy long-format DataFrame with reset index.
        if len(da_slice.dims) == 2 and 'ident' in da_slice.dims and 'qVar' in da_slice.dims:
            df = da_slice.to_pandas().dropna(how='all')
        else:
            df = da_slice.to_dataframe().dropna(subset=['5m']).reset_index()
            df = df.rename(columns={'5m': 'value'})

        return df.to_json(orient='split')

    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
