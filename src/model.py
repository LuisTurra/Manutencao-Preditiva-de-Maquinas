# src/model.py
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train(X_train, y_train):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))