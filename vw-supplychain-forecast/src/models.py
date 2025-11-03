import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_elasticnet(X_train, y_train, alpha=0.01, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, params=None):
    if params is None:
        params = {'n_estimators': 400, 'num_leaves': 31, 'learning_rate': 0.05}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    
    # Compute RMSE manually for compatibility with all sklearn versions
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    
    mae = mean_absolute_error(y, preds)
    return {'rmse': rmse, 'mae': mae, 'preds': preds}
