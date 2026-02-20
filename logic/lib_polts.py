#Imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_zarr_day(zarr_store, day, qVar='quote.mark', ident='AAPL'):
    """Extracts 5m data and plots it with a cleaned-up x-axis for one or more idents."""
    idents = [ident] if isinstance(ident, str) else ident
    
    plt.figure(figsize=(12, 5))
    
    for i in idents:
        data = zarr_store['5m'].sel(day=day, qVar=qVar, ident=i).to_pandas()
        plt.plot(data.index, data.values, label=i, linewidth=1.5)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12)) 
    plt.xticks(rotation=45)
    
    plt.title(f"{', '.join(idents)}: {qVar} ({day})", loc='left', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_zarr_range(zarr_store, start_day, end_day, qVar='quote.mark', ident='AAPL'):
    """Plots a range of days for one or more idents, handling missing dates."""
    idents = [ident] if isinstance(ident, str) else ident
    
    plt.figure(figsize=(14, 6))
    
    for i in idents:
        data = zarr_store['5m'].sel(
            day=slice(start_day, end_day), 
            qVar=qVar, 
            ident=i
        ).stack(timeline=('day', 'time')).to_pandas().dropna()
        
        x_labels = [f"{d} {t}" for d, t in data.index]
        plt.plot(x_labels, data.values, label=i, linewidth=1)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    
    plt.title(f"{', '.join(idents)}: {qVar} ({start_day} to {end_day})", loc='left', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()