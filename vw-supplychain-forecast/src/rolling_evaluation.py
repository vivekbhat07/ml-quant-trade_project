"""
Reproduces paper-style result table:
- Forecast horizons h ∈ {1, 5, 20}
- Models: ElasticNet, XGBoost, LightGBM
- Reports RMSE and Estimation Time
"""

import pandas as pd
import numpy as np
import time
from utils import scale_train_val_test
from models import train_elasticnet, train_xgboost, train_lightgbm, evaluate_model
import os

FEATURES_CSV = "data/vw_features.csv"
OUTPUTS_DIR = "outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

FORECAST_HORIZONS = [1, 5, 20]

def simulate_forecast_shift(y, h):
    """Shift target by h days forward (predict future returns)."""
    return y.shift(-h)

def run_results_table():
    df = pd.read_csv(FEATURES_CSV, parse_dates=['Date']).sort_values('Date')
    X = df.drop(columns=['Date', 'vw_returns'])
    y = df['vw_returns']

    results_rmse = pd.DataFrame(columns=["model", *FORECAST_HORIZONS])
    results_time = pd.DataFrame(columns=["model", *FORECAST_HORIZONS])

    models = {
        "Elastic Net": train_elasticnet,
        "XGBoost": train_xgboost,
        "LightGBM": train_lightgbm,
    }

    for model_name, model_func in models.items():
        rmse_row = {"model": model_name}
        time_row = {"model": model_name}

        for h in FORECAST_HORIZONS:
            y_shifted = simulate_forecast_shift(y, h).dropna()
            X_cut = X.iloc[:len(y_shifted)]

            # Split 80/20 time-wise (no shuffle)
            split_idx = int(0.8 * len(X_cut))
            X_train, X_test = X_cut.iloc[:split_idx], X_cut.iloc[split_idx:]
            y_train, y_test = y_shifted.iloc[:split_idx], y_shifted.iloc[split_idx:]

            # Scale
            X_train_s, _, X_test_s, _ = scale_train_val_test(X_train, X_train, X_test)

            # Train + time
            start = time.time()
            model = model_func(X_train_s, y_train.values)
            elapsed = time.time() - start

            # Evaluate
            eval_result = evaluate_model(model, X_test_s, y_test.values)
            rmse_row[h] = round(eval_result["rmse"], 4)
            time_row[h] = round(elapsed, 2)

        results_rmse = pd.concat([results_rmse, pd.DataFrame([rmse_row])], ignore_index=True)
        results_time = pd.concat([results_time, pd.DataFrame([time_row])], ignore_index=True)

    # Save results
    results_rmse.to_csv(OUTPUTS_DIR + "table_rmse.csv", index=False)
    results_time.to_csv(OUTPUTS_DIR + "table_time.csv", index=False)

    # Print table
    print("\n" + "=" * 60)
    print("Prediction Performance on Test Set (RMSE)")
    print(results_rmse.to_string(index=False))
    print("\nEstimation Time (Seconds)")
    print(results_time.to_string(index=False))
    print("=" * 60)

    # Combine and save LaTeX-like format
    combined = results_rmse.copy()
    for i, row in results_time.iterrows():
        combined.loc[i, "model"] = f"{combined.loc[i, 'model']}  (time in s)"
        for h in FORECAST_HORIZONS:
            combined.loc[i, h] = f"{results_rmse.loc[i, h]} / {results_time.loc[i, h]}"
    combined.to_csv(OUTPUTS_DIR + "paper_style_table.csv", index=False)
    print("\nSaved paper-style table to outputs/paper_style_table.csv")

if __name__ == "__main__":
    run_results_table()
