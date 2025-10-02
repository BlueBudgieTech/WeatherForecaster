import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import ttk
import joblib
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

def ensure_datetime_index(df, date_col='date', freq='D'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.sort_values(date_col).set_index(date_col)
    idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(idx)
    return df


def prepare_data(df, lags=[1,2,3,7,14,30], rolling_windows=[3,7,14]):
    df = ensure_datetime_index(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate(limit=3).ffill().bfill()

    for lag in lags:
        df[f'rain_lag_{lag}'] = df['rainfall'].shift(lag)
        for col in ['temperature', 'humidity', 'windspeed']:
            if col in df.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    for w in rolling_windows:
        df[f'rain_roll_mean_{w}'] = df['rainfall'].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'rain_roll_std_{w}'] = df['rainfall'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek

    df = df.dropna(subset=['rain_lag_1'])
    X = df.drop(columns=['rainfall'])
    y = df['rainfall']
    return X, y, df

def train_random_forest(X, y, test_size=0.2, random_state=42):
    split_idx = int(len(X)*(1-test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, scaler, mae, rmse

def forecast_next_days(df_full, model, scaler=None, days=30):
    df = df_full.copy()
    forecasts = []
    last_date = df.index.max()

    for i in range(days):
        next_date = last_date + pd.Timedelta(1, unit='D')
        row = {}
        for lag in [1,2,3,7,14,30]:
            row[f'rain_lag_{lag}'] = df['rainfall'].iloc[-lag] if len(df) >= lag else df['rainfall'].iloc[-1]
            for col in ['temperature','humidity','windspeed']:
                key = f'{col}_lag_{lag}'
                if key in df.columns:
                    row[key] = df[col].iloc[-lag] if len(df) >= lag else df[col].iloc[-1]

        for w in [3,7,14]:
            row[f'rain_roll_mean_{w}'] = df['rainfall'].iloc[-w:].mean()
            row[f'rain_roll_std_{w}'] = df['rainfall'].iloc[-w:].std()

        row['dayofyear'] = next_date.dayofyear
        row['month'] = next_date.month
        row['dayofweek'] = next_date.dayofweek

        row_df = pd.DataFrame(row, index=[next_date])
        feature_cols = [c for c in df.columns if c != 'rainfall']
        for c in feature_cols:
            if c not in row_df.columns:
                row_df[c] = 0.0
        row_df = row_df[feature_cols]

        X_scaled = scaler.transform(row_df) if scaler else row_df
        pred = model.predict(X_scaled)[0]
        forecasts.append((next_date, float(pred)))

        new_row_df = pd.DataFrame([ {**row, 'rainfall': pred} ], index=[next_date])
        for c in df.columns:
            if c not in new_row_df.columns:
                new_row_df[c] = df[c].iloc[-1]
        df = pd.concat([df, new_row_df], ignore_index=False)
        last_date = next_date

    forecast_df = pd.DataFrame(forecasts, columns=['Date','Predicted Rainfall'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)
    return forecast_df

def show_forecast_gui(forecast_df):
    root = tk.Tk()
    root.title('Rainfall Forecast')

    tree = ttk.Treeview(root, columns=['Date','Predicted Rainfall'], show='headings')
    tree.heading('Date', text='Date')
    tree.heading('Predicted Rainfall', text='Predicted Rainfall')
    tree.column('Date', width=150)
    tree.column('Predicted Rainfall', width=150)

    for index, row in forecast_df.iterrows():
        tree.insert('', tk.END, values=(index.strftime('%Y-%m-%d'), row['Predicted Rainfall']))

    tree.pack(expand=True, fill=tk.BOTH)
    root.mainloop()

def make_lstm_train_data(df, feature_cols, target_col='rainfall', lookback=30):
    values = df[feature_cols + [target_col]].values
    if len(values) <= lookback:
        raise ValueError(f"Not enough data for LSTM: {len(values)} rows, lookback={lookback}")
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i, :-1])
        y.append(values[i, -1])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    if os.path.exists('weather.csv'):
        raw = pd.read_csv('weather.csv')
        X, y, df_full = prepare_data(raw)

        rf_model, scaler, mae, rmse = train_random_forest(X, y)
        print(f"RandomForest -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        forecast = forecast_next_days(df_full, rf_model, scaler=scaler, days=30)

        show_forecast_gui(forecast)

    else:
        print('Please provide weather.csv with columns: date,rainfall,temperature,humidity,windspeed')

