import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv("data/Online Retail.csv", encoding="latin1")
    df.columns = df.columns.str.strip()
    return df


# -----------------------------
# PREPROCESS DATA
# -----------------------------
def preprocess_data(df):
    df = df.copy()

    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')

    df = df.dropna(subset=['InvoiceDate'])
    df = df[df['Quantity'] > 0]

    df['Sales'] = df['Quantity']

    df = df.groupby('InvoiceDate')['Sales'].sum().reset_index()
    df.rename(columns={'InvoiceDate': 'Date'}, inplace=True)

    # Features
    df['lag_1'] = df['Sales'].shift(1)
    df['lag_2'] = df['Sales'].shift(2)
    df['rolling_mean'] = df['Sales'].rolling(3).mean()

    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday

    df = df.dropna()

    return df


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(df):

    features = ['lag_1', 'lag_2', 'rolling_mean', 'day', 'month', 'weekday']

    X = df[features]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=4)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    # Best model
    if xgb_rmse < rf_rmse:
        best_model = xgb
        model_name = "XGBoost"
    else:
        best_model = rf
        model_name = "Random Forest"

    return best_model, df, features, rf_rmse, xgb_rmse, model_name