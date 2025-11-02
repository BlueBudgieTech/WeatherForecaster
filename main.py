

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import calendar
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def ensure_datetime_index(df, date_col='date', freq='D'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
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
        for col in ['temperature','humidity','windspeed']:
            if col in df.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    for w in rolling_windows:
        df[f'rain_roll_mean_{w}'] = df['rainfall'].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'rain_roll_std_{w}'] = df['rainfall'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)

    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek

    df = df.dropna(subset=['rain_lag_1'])
    
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols.remove('rainfall')
    X = df[feature_cols]
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

        new_row_df = pd.DataFrame([{**row, 'rainfall': pred}], index=[next_date])
        for c in df.columns:
            if c not in new_row_df.columns:
                new_row_df[c] = df[c].iloc[-1]
        df = pd.concat([df, new_row_df], ignore_index=False)
        last_date = next_date

    forecast_df = pd.DataFrame(forecasts, columns=['Date','Predicted Rainfall'])
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df.set_index('Date', inplace=True)
    return forecast_df


def recommend_daily_crops(rainfall):
    if rainfall < 2:
        return "Millet, Sorghum, Barley"
    elif rainfall < 5:
        return "Maize, Wheat, Pulses"
    else:
        return "Rice, Sugarcane, Banana"

def generate_crop_plan(df):
    df['month'] = df.index.month
    month_to_season = {12:'Winter',1:'Winter',2:'Winter',
                       3:'Spring',4:'Spring',5:'Spring',
                       6:'Summer',7:'Summer',8:'Summer',
                       9:'Autumn',10:'Autumn',11:'Autumn'}
    seasonal_avg = {}
    for season in ['Winter','Spring','Summer','Autumn']:
        months = [m for m,s in month_to_season.items() if s==season]
        seasonal_avg[season] = df[df['month'].isin(months)]['rainfall'].mean()
    crop_plan = {}
    for season, rain_avg in seasonal_avg.items():
        if rain_avg < 2:
            crop_plan[season] = ['Millet', 'Sorghum', 'Barley']
        elif rain_avg < 5:
            crop_plan[season] = ['Maize','Wheat','Pulses']
        else:
            crop_plan[season] = ['Rice','Sugarcane','Banana']
    return crop_plan


def calculate_irrigation(rainfall, soil_moisture, crop_type):
    crop_water_req = {'Rice': 5, 'Maize': 3, 'Wheat': 2, 'Millet': 1.5,
                      'Sorghum':1.5, 'Barley':2, 'Pulses':2, 'Sugarcane':5, 'Banana':5}
    base_req = crop_water_req.get(crop_type, 2)
    effective_rain = min(rainfall, base_req)
    moisture_deficit = max(0, 50 - soil_moisture)/50
    irrigation_needed = (base_req - effective_rain) * moisture_deficit
    return max(0, round(irrigation_needed,2))

def add_irrigation_to_forecast(forecast_df, soil_moisture=30):
    forecast_df['Irrigation Needed (mm)'] = forecast_df.apply(
        lambda row: calculate_irrigation(
            row['Predicted Rainfall'],
            soil_moisture,
            recommend_daily_crops(row['Predicted Rainfall']).split(',')[0].strip()
        ), axis=1
    )
    return forecast_df


def show_dashboard_responsive(df_full, forecast_df, soil_moisture=30):
    forecast_df['Recommended Crops'] = forecast_df['Predicted Rainfall'].apply(recommend_daily_crops)
    forecast_df = add_irrigation_to_forecast(forecast_df, soil_moisture)
    crop_plan = generate_crop_plan(df_full)

    root = tk.Tk()
    root.title("Smart Farming Dashboard v18")
    root.geometry("1920x1080")

    root.grid_rowconfigure(0, weight=3)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=4)
    root.grid_columnconfigure(1, weight=1)

    # Graph frame
    graph_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
    graph_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    fig, axes = plt.subplots(2,1,figsize=(8,6))
    fig.tight_layout(pad=3.0)

    # Monthly averages
    monthly_avg_actual = df_full.groupby(df_full.index.month)['rainfall'].mean()
    monthly_avg_forecast = forecast_df.groupby(forecast_df.index.month)['Predicted Rainfall'].mean()
    month_names = [calendar.month_name[m] for m in range(1,13)]
    monthly_avg_all = pd.Series(data=np.zeros(12), index=range(1,13))
    monthly_avg_all.update(monthly_avg_actual)
    monthly_avg_all.update(monthly_avg_forecast)
    colors = ['orange' if m in monthly_avg_forecast.index else 'skyblue' for m in range(1,13)]
    axes[0].bar(month_names, monthly_avg_all.values, color=colors)
    axes[0].set_title("Average Rainfall per Month", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Daily rainfall current month
    current_month = pd.Timestamp.now().month
    daily_data = pd.concat([df_full[['rainfall']].rename(columns={'rainfall':'Rainfall'}),
                            forecast_df.rename(columns={'Predicted Rainfall':'Rainfall'})])
    daily_data_current = daily_data[daily_data.index.month == current_month]
    historical = daily_data_current[daily_data_current.index <= df_full.index.max()]
    forecasted = daily_data_current[daily_data_current.index > df_full.index.max()]
    axes[1].plot(historical.index, historical['Rainfall'], marker='o', label='Actual', color='blue')
    if not forecasted.empty:
        axes[1].plot(forecasted.index, forecasted['Rainfall'], marker='o', label='Forecast', color='orange')
    axes[1].set_title(f"Daily Rainfall - {pd.Timestamp.now().strftime('%B')}", fontsize=12)
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Forecast Table
    table_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
    table_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)

    tk.Label(table_frame, text="30-Day Forecast & Crop Recommendations", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky="w", pady=5)
    columns = ("Date","Predicted Rainfall","Recommended Crops","Irrigation Needed (mm)")
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=130 if col=="Recommended Crops" else 100, anchor='center')
    tree.grid(row=1, column=0, sticky="nsew")

    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=1, column=1, sticky='ns')

    for idx,row in forecast_df.iterrows():
        tree.insert('', tk.END, values=(
            idx.strftime('%Y-%m-%d'),
            f"{row['Predicted Rainfall']:.2f}",
            row['Recommended Crops'],
            f"{row['Irrigation Needed (mm)']:.2f}"
        ))

    # Color-code irrigation
    for child in tree.get_children():
        irrigation = float(tree.item(child)['values'][3])
        if irrigation > 3:
            tree.item(child, tags=('high',))
        elif irrigation > 0:
            tree.item(child, tags=('medium',))
        else:
            tree.item(child, tags=('low',))
    tree.tag_configure('high', background='tomato')
    tree.tag_configure('medium', background='yellow')
    tree.tag_configure('low', background='lightgreen')

    # Seasonal crop plan (compact)
    plan_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
    plan_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

    tk.Label(plan_frame, text="Seasonal Crop Plan (Rotation Applied)", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(3,2))
    inner_frame = tk.Frame(plan_frame)
    inner_frame.pack(anchor='w', fill='x')
    for season, crops in crop_plan.items():
        tk.Label(inner_frame, text=f"{season}: ", font=('Arial', 9, 'bold')).pack(side='left', padx=(0,2))
        tk.Label(inner_frame, text=', '.join(crops), font=('Arial', 9)).pack(side='left', padx=(0,10))

    root.mainloop()

if __name__ == '__main__':
    if os.path.exists('weather.csv'):
        raw = pd.read_csv('weather.csv')
        X, y, df_full = prepare_data(raw)
        rf_model, scaler, mae, rmse = train_random_forest(X, y)
        print(f"RandomForest -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        forecast = forecast_next_days(df_full, rf_model, scaler=scaler, days=30)
        show_dashboard_responsive(df_full, forecast, soil_moisture=30)
    else:
        print('Please provide weather.csv with columns: date,rainfall,temperature,humidity,windspeed')
