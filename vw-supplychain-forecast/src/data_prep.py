"""
Load CSV, fill missing, compute returns if needed, and save cleaned CSV.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils import save_obj

DATA_IN = "data/vw_supplychain.csv"
DATA_OUT = "data/vw_supplychain_cleaned.csv"

def prepare():
    df = pd.read_csv(DATA_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # If user provided Open/Close but not Return, compute close returns
    # We'll detect columns ending with "_Return" or ".Return" or "_Return"
    cols = df.columns.tolist()
    # Heuristic: If vw_returns not present but vw close exists, compute it
    if 'vw_returns' not in df.columns:
        # Try to find VW close column "066570.KQ_Close" or "VW.DE_Close" etc.
        # We'll attempt to find a column that contains "vw" or "VW" and "Close"
        candidates = [c for c in cols if ('vw' in c.lower() or 'VW' in c) and 'close' in c.lower()]
        if len(candidates) == 1:
            close_col = candidates[0]
            df['vw_returns'] = df[close_col].pct_change()
        else:
            # fallback: if there's a column named "066570.KQ_Close" or "FR.PA_Close" for VW - user must ensure target exists
            pass

    # Fill missing returns by forward/backfill small gaps
    ret_cols = [c for c in df.columns if c.lower().endswith('_return') or c.lower().endswith('.return') or 'return' in c.lower()]
    for c in ret_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Simple imputation: forward then backward
    df[ret_cols] = df[ret_cols].ffill().bfill()

    # Drop rows with any NaN in target or after imputation
    df = df.dropna(subset=['vw_returns']).reset_index(drop=True)

    # Save cleaned
    df.to_csv(DATA_OUT, index=False)
    print(f"Saved cleaned CSV to {DATA_OUT}")

if __name__ == "__main__":
    prepare()

