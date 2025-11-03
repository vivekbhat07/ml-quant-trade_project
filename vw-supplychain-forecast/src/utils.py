import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def save_obj(obj, path):
    joblib.dump(obj, path)

def load_obj(path):
    return joblib.load(path)

def scale_train_val_test(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler

