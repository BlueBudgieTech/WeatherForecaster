import pandas as pd
import requests
from datetime import datetime, timedelta

CSV_FILE = "weather.csv"

# Ask for user input
city = input("Enter city name (e.g. London or Dubai): ").strip()

# Geocode using Open-Meteo API
geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.requote_uri(city)}"
geo_resp = requests.get(geo_url, timeout=10)
geo_resp.raise_for_status()
geo_data = geo_resp.json()

if "results" not in geo_data or len(geo_data["results"]) == 0:
    raise ValueError(f"Could not find location: {city}")

top = geo_data["results"][0]
lat = top["latitude"]
lon = top["longitude"]
resolved_name = top.get("name", "").strip()
country = top.get("country", "").strip()

city_key = f"{resolved_name}, {country}" if country else resolved_name
print(f"\nFound location: {city_key} ({lat}, {lon})")

# Define 6-month date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=180)
print(f"\nFetching data from {start_date} to {end_date}...\n")

# Fetch historical data from Open-Meteo
api_url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={lat}&longitude={lon}"
    f"&start_date={start_date}&end_date={end_date}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "wind_speed_10m_max,relative_humidity_2m_max"
    "&timezone=auto"
)

try:
    resp = requests.get(api_url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "daily" not in data or "time" not in data["daily"]:
        raise ValueError("No daily data returned by API.")

    dates = data["daily"]["time"]
    rain = data["daily"].get("precipitation_sum", [])
    temp_max = data["daily"].get("temperature_2m_max", [])
    temp_min = data["daily"].get("temperature_2m_min", [])
    humidity = data["daily"].get("relative_humidity_2m_max", [])
    windspeed = data["daily"].get("wind_speed_10m_max", [])

    # Build the dataset for the current city
    rows = []
    for i, d in enumerate(dates):
        date_str = datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
        avg_temp = None
        if i < len(temp_max) and i < len(temp_min):
            avg_temp = (temp_max[i] + temp_min[i]) / 2
        elif i < len(temp_max):
            avg_temp = temp_max[i]
        elif i < len(temp_min):
            avg_temp = temp_min[i]

        rows.append({
            "date": date_str,
            "rainfall": rain[i] if i < len(rain) else None,
            "temperature": avg_temp,
            "humidity": humidity[i] if i < len(humidity) else None,
            "windspeed": windspeed[i] if i < len(windspeed) else None
        })

    df = pd.DataFrame(rows, columns=["date", "rainfall", "temperature", "humidity", "windspeed"])

    # Overwrite the CSV file completely
    df.to_csv(CSV_FILE, index=False)
    print(f"Weather data for {city_key} written to {CSV_FILE} ({len(df)} rows).")

except Exception as e:
    print(f"Error fetching weather data: {e}")
