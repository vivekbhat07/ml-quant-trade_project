"""
Create lag features: for each supplier return column, we add lags 1..L.
Also optional rolling mean / vol
"""
import pandas as pd
from pathlib import Path

DATA_IN = "data/vw_supplychain_cleaned.csv"
DATA_OUT = "data/vw_features.csv"

LAGS = 5  # number of past days to include per feature
ROLL_WINDOWS = [3]  # optional rolling features

def make_features():
    df = pd.read_csv(DATA_IN, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # identify return columns (exclude target)
    ret_cols = [c for c in df.columns if c.lower().endswith('_return') or 'return' in c.lower()]
    ret_cols = [c for c in ret_cols if c != 'vw_returns']
    features = df[['Date', 'vw_returns']].copy()

    for col in ret_cols:
        for lag in range(1, LAGS+1):
            features[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in ROLL_WINDOWS:
            features[f"{col}_rmean_{w}"] = df[col].rolling(window=w).mean().shift(1)
            features[f"{col}_rstd_{w}"] = df[col].rolling(window=w).std().shift(1)

    # Include bond_yeld as is and optionally its lag
    if 'bond_yeld' in df.columns:
        features['bond_yeld'] = df['bond_yeld']
        features['bond_yeld_lag1'] = df['bond_yeld'].shift(1)

    # drop rows with nan (due to lagging)
    features = features.dropna().reset_index(drop=True)
    features.to_csv(DATA_OUT, index=False)
    print(f"Saved features to {DATA_OUT}")

if __name__ == "__main__":
    make_features()
