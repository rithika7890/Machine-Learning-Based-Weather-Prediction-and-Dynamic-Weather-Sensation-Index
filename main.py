import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os
import threading
from tkinter import *
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# --- Configuration ---
API_KEY = os.environ['OPENWEATHERMAP_API_KEY']  
MODEL_DIR = "weather_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_TRAIN_TIMES = {}
HISTORICAL_DAYS = 7

def fetch_api_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def get_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = fetch_api_data(url)
    if data:
        main = data.get("main", {})
        wind = data.get("wind", {})
        clouds = data.get("clouds", {})
        rain = data.get("rain", {})
        snow = data.get("snow", {})

        return {
            "temp": main.get("temp", np.nan),
            "humidity": main.get("humidity", np.nan),
            "pressure": main.get("pressure", np.nan),
            "wind_speed": wind.get("speed", np.nan),
            "cloudiness": clouds.get("all", np.nan),
            "temp_min": main.get("temp_min", np.nan),
            "temp_max": main.get("temp_max", np.nan),
            "feels_like": main.get("feels_like", np.nan),
            "sea_level": main.get("sea_level", np.nan),
            "grnd_level": main.get("grnd_level", np.nan),
            "visibility": data.get("visibility", np.nan),
            "rain_1h": rain.get("1h", 0) if rain else 0,
            "snow_1h": snow.get("1h", 0) if snow else 0,
        }
    return None

def download_historical_data(city, days=HISTORICAL_DAYS):
    coords = get_city_coordinates(city)
    if not coords:
        return None

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={coords['lat']}&lon={coords['lon']}&type=hour&start={int(start_time.timestamp())}&end={int(end_time.timestamp())}&appid={API_KEY}"

    data = fetch_api_data(url)

    if data and 'list' in data:
        historical_data = []
        for entry in data['list']:
            try:
                historical_data.append({
                    "date": datetime.fromtimestamp(entry['dt'], tz=timezone.utc).strftime("%Y-%m-%d"),
                    "temp": entry.get('main', {}).get('temp', np.nan) - 273.15,
                    "humidity": entry.get('main', {}).get('humidity', np.nan),
                    "pressure": entry.get('main', {}).get('pressure', np.nan),
                    "wind_speed": entry.get('wind', {}).get('speed', np.nan),
                    "cloudiness": entry.get('clouds', {}).get('all', np.nan),
                })
            except (TypeError, KeyError):
                continue

        historical_df = pd.DataFrame(historical_data)
        return historical_df
    else:
        return None

def get_city_coordinates(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    data = fetch_api_data(url)
    if data and len(data) > 0:
        return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
    return None

def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return "spring"
    elif 6 <= month <= 8:
        return "summer"
    elif 9 <= month <= 11:
        return "autumn"
    else:
        return "winter"

def calculate_dwsi(weather_data, season="average", latitude=None):
    temp = weather_data["temp"]
    humidity = weather_data["humidity"]
    wind_speed = weather_data["wind_speed"]
    cloudiness = weather_data["cloudiness"]
    feels_like = weather_data["feels_like"]
    pressure = weather_data["pressure"]
    temp_min = weather_data["temp_min"]
    temp_max = weather_data["temp_max"]
    sea_level = weather_data["sea_level"] if weather_data["sea_level"] else pressure
    grnd_level = weather_data["grnd_level"] if weather_data["grnd_level"] else pressure
    visibility = weather_data["visibility"]
    rain_1h = weather_data["rain_1h"]
    snow_1h = weather_data["snow_1h"]

    season_weights = {
        "spring": {"temp": 0.20, "humidity": 0.15, "wind": 0.08, "pressure": 0.04, "cloud": 0.05, "visibility": 0.05, "rain": 0.12, "snow": 0.01, "sea": 0.04, "grnd": 0.04, "feels_like": 0.22},
        "summer": {"temp": 0.30, "humidity": 0.20, "wind": 0.05, "pressure": 0.03, "cloud": 0.03, "visibility": 0.04, "rain": 0.10, "snow": 0.00, "sea": 0.03, "grnd": 0.03, "feels_like": 0.19},
        "autumn": {"temp": 0.22, "humidity": 0.18, "wind": 0.10, "pressure": 0.06, "cloud": 0.06, "visibility": 0.06, "rain": 0.15, "snow": 0.03, "sea": 0.05, "grnd": 0.05, "feels_like": 0.14},
        "winter": {"temp": 0.15, "humidity": 0.12, "wind": 0.12, "pressure": 0.08, "cloud": 0.08, "visibility": 0.08, "rain": 0.05, "snow": 0.20, "sea": 0.06, "grnd": 0.06, "feels_like": 0.00},
        "average": {"temp": 0.24, "humidity": 0.16, "wind": 0.1, "pressure": 0.05, "cloud": 0.05, "visibility": 0.05, "rain": 0.1, "snow": 0.05, "sea": 0.05, "grnd": 0.05, "feels_like": 0.1}
    }
    weights = season_weights.get(season, season_weights["average"])

    temp_variability = np.std([temp, temp_min, temp_max, feels_like])
    temp_impact = min(temp_variability / 10, 1)

    if humidity < 30:
        humidity_impact = 0.5 + (30 - humidity) / 140
    elif humidity > 70:
        humidity_impact = 0.5 + (humidity - 70) / 140
    else:
        humidity_impact = (humidity - 50) / 50
    humidity_impact = max(0, min(humidity_impact, 1))

    wind_impact = min(wind_speed / 20, 1)

    pressure_impact = np.abs(pressure - 1013.25) / 50
    sea_level_impact = np.abs(sea_level - 1013.25) / 50
    grnd_level_impact = np.abs(grnd_level - 1013.25) / 50

    cloudiness_impact = cloudiness / 100
    visibility_impact = max((10000 - visibility) / 10000, 0)
    rain_impact = min(rain_1h / 10, 1)
    snow_impact = min(snow_1h / 5, 1)

    uv_proxy = 0
    if latitude is not None:
        solar_angle = np.cos(np.radians(latitude))
        uv_proxy = max(0, solar_angle)
        uv_proxy = min(uv_proxy, 1)

    air_quality_proxy = 0

    dwsi = (
        weights["temp"] * temp_impact +
        weights["feels_like"]* temp_impact +
        weights["humidity"] * humidity_impact +
        weights["wind"] * wind_impact +
        weights["pressure"] * pressure_impact +
        weights["cloud"] * cloudiness_impact +
        weights["visibility"] * visibility_impact +
        weights["rain"] * rain_impact +
        weights["snow"] * snow_impact +
        weights["sea"] * sea_level_impact +
        weights["grnd"] * grnd_level_impact
    )

    dwsi = min(max(dwsi, 0), 1)
    return dwsi

def get_model_path(city, target):
    return os.path.join(MODEL_DIR, f"{city}_{target}_model.pkl")

def train_model(city, target, features):
    model_path = get_model_path(city, target)
    now = datetime.now()

    if city in MODEL_TRAIN_TIMES and os.path.exists(model_path) and now - MODEL_TRAIN_TIMES[city] < timedelta(days=7):
        return True

    historical_df = download_historical_data(city)
    if historical_df is None or historical_df.empty:
        return False

    def calculate_historical_dwsi(row):
        weather_data = {
            "temp": row["temp"],
            "humidity": row["humidity"],
            "wind_speed": row["wind_speed"],
            "cloudiness": row["cloudiness"],
            "pressure": row["pressure"],
            "feels_like": row["temp"],  
            "temp_min": row["temp"],  
            "temp_max": row["temp"],  
            "sea_level": row["pressure"],  
            "grnd_level": row["pressure"],  
            "visibility": 10000,  
            "rain_1h": 0,  
            "snow_1h": 0  
        }
        date = datetime.strptime(row["date"], "%Y-%m-%d")
        season = get_season(date)

        coords = get_city_coordinates(city)
        latitude = coords['lat'] if coords else None

        return calculate_dwsi(weather_data, season=season, latitude=latitude)

    historical_df["dwsi"] = historical_df.apply(calculate_historical_dwsi, axis=1)
    historical_df = historical_df.fillna(historical_df.mean(numeric_only=True))

    if not all(col in historical_df.columns for col in features + [target]):
        return False

    X_train, X_test, y_train, y_test = train_test_split(historical_df[features], historical_df[target], test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    MODEL_TRAIN_TIMES[city] = now
    return True

def load_or_train_model(city, target, features):
    model_path = get_model_path(city, target)
    if os.path.exists(model_path):
        return joblib.load(model_path)

    if train_model(city, target, features):
        return joblib.load(model_path)
    return None

def predict_current(city, target, features):
    model = load_or_train_model(city, target, features)
    if not model:
        return "Could not load or train the model."

    weather_data = get_weather_data(city)
    if weather_data:
        
        feature_values = np.array([[weather_data.get(f, np.nan) for f in features]])
        try:
            predicted_value = model.predict(feature_values)[0]
            return f"{predicted_value:.2f}"
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error during prediction."

    return "Weather data unavailable for prediction."

def predict_dwsi(city):
    features = ["temp", "humidity", "pressure", "wind_speed", "cloudiness",
                "temp_min", "temp_max", "feels_like", "sea_level", "grnd_level",
                "visibility", "rain_1h", "snow_1h"]
    return predict_current(city, "dwsi", features)

def predict_future(city, target, features):
    model = load_or_train_model(city, target, features)
    if not model:
        return "Could not load or train the model."

    weather_data = get_weather_data(city)
    if weather_data:
        feature_values = np.array([[weather_data.get(f, np.nan) for f in features]])
        try:
            predicted_value = model.predict(feature_values)[0]
            return f"{predicted_value:.2f}°C"
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error during prediction."

    return "Weather data unavailable for prediction."

def predict_future_temperature(city):
    features = ["humidity", "pressure", "wind_speed", "cloudiness"]
    return predict_future(city, "temp", features)

def plot_temperature_data(historical_df, current_temp, predicted_temp):
    fig, ax = plt.subplots(figsize=(12, 4))

    dates = pd.to_datetime(historical_df['date'])
    historical_df['temp'] = pd.to_numeric(historical_df['temp'])

    today = datetime.now().date()
    today = pd.to_datetime(today)
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df_filtered = historical_df[historical_df['date'].dt.date < today.date()]

    ax.plot(historical_df_filtered['date'], historical_df_filtered['temp'], label='Historical Temperature', marker='o', linestyle='-', markersize=4, linewidth=1, color='blue')

    ax.scatter(today, current_temp, color='red', label='Current Temperature', s=100, zorder=5)
    ax.scatter(today, predicted_temp, color='green', label='Predicted Temperature (for next few hours)', s=100, marker='x', zorder=5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Forecast', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter('{x:.1f}')
    min_date = min(dates)
    max_date = max(dates)
    date_range = max_date - min_date
    padding = date_range * 0.05
    ax.set_xlim([min_date - padding, today + padding])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    ax.yaxis.set_major_formatter('{x:.1f}')
    return fig, ax

def display_graph(fig):
    global graph_widget
    if graph_widget:
        for widget in graph_widget.winfo_children():
            widget.destroy()
        graph_widget.destroy()

    graph_widget = Frame(root)
    graph_widget.grid(row=5, column=0, columnspan=2, padx=20, pady=20)

    canvas = FigureCanvasTkAgg(fig, master=graph_widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

def get_forecast_prediction():
    city = city_entry.get()
    if not city:
        display_output("Enter a city name.")
        return

    historical_df = download_historical_data(city)
    if historical_df is None or historical_df.empty:
        display_output("No historical data to display.")
        return

    weather_data = get_weather_data(city)
    if not weather_data:
        display_output("Could not retrieve current weather data.")
        return
    current_temp = weather_data['temp']

    prediction_result = predict_future_temperature(city)

    try:
        predicted_temp = float(prediction_result.strip("°C"))
    except:
        predicted_temp = current_temp

    fig, ax = plot_temperature_data(historical_df, current_temp, predicted_temp)
    display_graph(fig)

def predict_dwsi_weather_impact(city, season="average", latitude=None):
    weather_data = get_weather_data(city)

    if not weather_data:
        return "Weather data unavailable."

    # Use the predicted DWSI from the trained model, if available, otherwise calculate it manually
    dwsi_prediction = predict_dwsi(city)
    try:
        dwsi = float(dwsi_prediction)  #DWSI calculated with trained model
    except:
        dwsi = calculate_dwsi(weather_data, season, latitude) #DWSI calculated manually

    if dwsi < 0.2:
        impact = "Very Comfortable ☀️"
    elif dwsi < 0.4:
        impact = "Slightly Uncomfortable 🌤️"
    elif dwsi < 0.6:
        impact = "Moderately Disruptive 🌦️"
    elif dwsi < 0.8:
        impact = "Highly Uncomfortable 🥵"
    else:
        impact = "Extreme Weather Alert 🚨"

    return f"DWSI: {dwsi:.2f} → {impact}"

# --- GUI Setup ---
def get_weather_prediction():
    city = city_entry.get()
    if not city:
        display_output("Enter a city name.")
        return

    coords = get_city_coordinates(city)
    latitude = coords['lat'] if coords else None
    season = get_season(datetime.now())

    threading.Thread(target=lambda: display_weather_info(city, season, latitude)).start()

def display_weather_info(city, season, latitude):
    weather_data = get_weather_data(city)
    if not weather_data:
        display_output("Could not retrieve weather data.")
        return

    predicted_temp = predict_future_temperature(city)
    dwsi_impact = predict_dwsi_weather_impact(city, season, latitude)

    message = f"Current Temperature: {weather_data['temp']}°C\n"
    message += f"Predicted Temperature (for next few hours): {predicted_temp}\n"
    message += f"{dwsi_impact}\n"

    display_output(message)

def display_output(message):
    output_text.delete("1.0", END)
    output_text.insert("1.0", message)

root = Tk()
root.title("Weather Prediction")
graph_widget = None
Label(root, text="Enter City:").grid(row=0, column=0, padx=10, pady=10, sticky=W)
city_entry = Entry(root, width=30)
city_entry.grid(row=0, column=1, padx=10, pady=10, sticky=E)
Button(root, text="Get Weather", command=get_weather_prediction).grid(row=1, column=0, columnspan=2, pady=10)
Button(root, text="Show Forecast", command=get_forecast_prediction).grid(row=2, column=0, columnspan=2, pady=10)
output_text = scrolledtext.ScrolledText(root, width=70, height=10)
output_text.grid(row=3, column=0, columnspan=2, padx=20, pady=20)
root.mainloop()