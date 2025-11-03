"""
Simple trading backtester using model predictions.
Strategy:
- If predicted return > threshold -> go long 1 share
- If predicted return < -threshold -> go short 1 share
- Else hold / flat
No transaction costs included (you can add easily)
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from models import evaluate_model

MODELS_DIR = "models"
FEATURES_CSV = "data/vw_features.csv"
OUTPUTS_DIR = "outputs"
Path(OUTPUTS_DIR).mkdir(exist_ok=True)

def run_backtest(model_path, scaler_path, test_start, threshold=0.0):
    df = pd.read_csv(FEATURES_CSV, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # test window defined by model name
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # find test period - model filename contains test_start e.g. en_2017-06-01.joblib
    # We'll assume the saved model was trained with scaler and test month immediately after validation; 
    # to backtest we will locate that test month in features csv
    ts = Path(model_path).stem.split('_')[-1]
    test_month = pd.to_datetime(ts)
    test_start = test_month
    test_end = test_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)

    mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    X_test = df.loc[mask].drop(columns=['Date','vw_returns'])
    y_test = df.loc[mask]['vw_returns']

    X_test_s = scaler.transform(X_test)
    preds = model.predict(X_test_s)

    # simple P&L: assume 1 unit position, pnl = predicted direction * realized return
    positions = np.where(preds > threshold, 1, np.where(preds < -threshold, -1, 0))
    pnl_series = positions * y_test.values  # percent returns per day
    cum_pnl = (1 + pnl_series).cumprod() - 1

    out = pd.DataFrame({
        'Date': df.loc[mask, 'Date'],
        'pred': preds,
        'pos': positions,
        'real': y_test.values,
        'pnl': pnl_series,
        'cum_pnl': cum_pnl
    })
    return out

if __name__ == "__main__":
    # example: run backtest for earliest model
    model_files = sorted(Path(MODELS_DIR).glob("en_*.joblib"))
    if len(model_files) == 0:
        print("No models found. Run rolling_evaluation first.")
        exit(1)
    model_path = str(model_files[0])
    scaler_path = model_path.replace("en_", "scaler_")
    out = run_backtest(model_path, scaler_path, None, threshold=0.0025)
    out.to_csv("outputs/backtest_example.csv", index=False)
    print("Saved backtest example to outputs/backtest_example.csv")
