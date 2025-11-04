# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():
    train = pd.read_csv("data/train_FD001.txt", sep='\s+', header=None, engine='python')
    test = pd.read_csv("data/test_FD001.txt", sep='\s+', header=None, engine='python')
    rul = pd.read_csv("data/RUL_FD001.txt", header=None)

    # Limpar colunas vazias
    train = train.iloc[:, :26]
    test = test.iloc[:, :26]

    cols = ['unit', 'cycle'] + [f'setting{i}' for i in range(1,4)] + [f'sensor{i}' for i in range(1,22)]
    train.columns = cols
    test.columns = cols

    return train, test, rul.values.flatten()

def add_rul(train):
    max_cycle = train.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']
    train = train.merge(max_cycle, on='unit')
    train['RUL'] = train['max_cycle'] - train['cycle']
    return train.drop('max_cycle', axis=1)

def feature_engineering(df):
    sensor_cols = [f'sensor{i}' for i in range(1,22)]
    for col in sensor_cols:
        df[f'{col}_roll'] = df.groupby('unit')[col].transform(lambda x: x.rolling(10, min_periods=1).mean())
    return df

def scale(train, test):
    cols = [c for c in train.columns if c not in ['unit', 'cycle', 'RUL']]
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train[cols]), columns=cols)
    X_test = pd.DataFrame(scaler.transform(test[cols]), columns=cols)
    return X_train, X_test, scaler, cols