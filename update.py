import os
import pandas as pd
import requests
from datetime import datetime

# -------------------- Configuration --------------------
CSV_FILE = "weather.csv"
API_KEY = "a280d32deb0f483dba1121252250210"  # Replace with your API key
LOCATION = "Chennai,India"    # Replace with your location
API_URL = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={LOCATION}"

# -------------------- Ensure CSV Exists --------------------
if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    df = pd.DataFrame(columns=["date", "rainfall", "temperature", "humidity", "windspeed"])
    df.to_csv(CSV_FILE, index=False)

# -------------------- Fetch Weather Data --------------------
try:
    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Extract relevant fields
    new_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "rainfall": data["current"].get("precip_mm", 0.0),
        "temperature": data["current"].get("temp_c", 0.0),
        "humidity": data["current"].get("humidity", 0.0),
        "windspeed": data["current"].get("wind_kph", 0.0)
    }

except Exception as e:
    print(f"Error fetching weather data: {e}")
    # Fallback: insert zeros
    new_row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "rainfall": 0.0,
        "temperature": 0.0,
        "humidity": 0.0,
        "windspeed": 0.0
    }

# -------------------- Update CSV --------------------
df = pd.read_csv(CSV_FILE)

# Prevent duplicate dates
if new_row["date"] not in df["date"].values:
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"Weather data updated for {new_row['date']}")
else:
    print(f"Data for {new_row['date']} already exists. No update made.")
