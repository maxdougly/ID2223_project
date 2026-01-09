"""
Utility functions for data retrieval: OpenMeteo weather, elprisetjustnu.se prices,
and Entsoe energy production.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import matplotlib.pyplot as plt
import seaborn as sns
import os


def setup_openmeteo_session():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def get_historical_weather(start_date, end_date, latitude=59.33, longitude=18.07):
    """Fetch historical weather data from OpenMeteo for Stockholm (SE3)."""
    client = setup_openmeteo_session()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum"
        ],
        "timezone": "Europe/Stockholm"
    }

    try:
        responses = client.weather_api(url, params=params)
        response = responses[0]

        daily = response.Daily()
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )

        weather_data = pd.DataFrame({
            "date": dates,
            "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m_max": daily.Variables(4).ValuesAsNumpy(),
            "wind_direction_10m_dominant": daily.Variables(5).ValuesAsNumpy(),
            "shortwave_radiation_sum": daily.Variables(6).ValuesAsNumpy()
        })

        weather_data['date'] = weather_data['date'].dt.date
        return weather_data

    except Exception as e:
        print(f"Error fetching historical weather data: {e}")
        return pd.DataFrame()


def get_weather_forecast(days_ahead=10, latitude=59.33, longitude=18.07):
    """Fetch weather forecast from OpenMeteo for Stockholm."""
    client = setup_openmeteo_session()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum"
        ],
        "timezone": "Europe/Stockholm",
        "forecast_days": days_ahead
    }

    try:
        responses = client.weather_api(url, params=params)
        response = responses[0]

        daily = response.Daily()
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )

        forecast_data = pd.DataFrame({
            "date": dates,
            "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m_max": daily.Variables(4).ValuesAsNumpy(),
            "wind_direction_10m_dominant": daily.Variables(5).ValuesAsNumpy(),
            "shortwave_radiation_sum": daily.Variables(6).ValuesAsNumpy()
        })

        forecast_data['date'] = forecast_data['date'].dt.date
        return forecast_data

    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        return pd.DataFrame()


def get_electricity_prices(start_date, end_date, region="SE3"):
    """Fetch electricity prices from elprisetjustnu.se API, aggregated to daily mean/min/max/std."""
    prices = []

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    current_date = start
    while current_date <= end:
        date_str = current_date.strftime('%Y/%m-%d')
        url = f"https://www.elprisetjustnu.se/api/v1/prices/{date_str}_{region}.json"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for entry in data:
                    prices.append({
                        'timestamp': pd.to_datetime(entry['time_start']),
                        'price_sek_kwh': entry['SEK_per_kWh']
                    })
            else:
                print(f"Warning: Failed to fetch data for {current_date.date()}, status: {response.status_code}")
        except Exception as e:
            print(f"Error fetching electricity prices for {current_date.date()}: {e}")

        current_date += timedelta(days=1)

    if prices:
        df = pd.DataFrame(prices)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['date'] = df['timestamp'].dt.date
        df_daily = df.groupby('date')['price_sek_kwh'].agg(['mean', 'min', 'max', 'std']).reset_index()
        df_daily.columns = ['date', 'price_sek_kwh_mean', 'price_sek_kwh_min', 'price_sek_kwh_max', 'price_sek_kwh_std']
        return df_daily
    else:
        return pd.DataFrame()


def get_entsoe_generation_data(start_date, end_date, api_key):
    """Fetch energy generation data from Entsoe Transparency Platform (placeholder implementation)."""
    print("Note: Entsoe integration requires API key and additional configuration.")
    print("For now, returning empty DataFrame. Implement full Entsoe integration as needed.")
    return pd.DataFrame()


def plot_electricity_price_forecast(historical_df, forecast_df, city="Stockholm"):
    plt.figure(figsize=(14, 6))

    if not historical_df.empty:
        plt.plot(historical_df['date'], historical_df['price_sek_kwh_mean'],
                label='Historical Price', color='blue', linewidth=2)

    if not forecast_df.empty:
        plt.plot(forecast_df['date'], forecast_df['predicted_price'],
                label='Predicted Price', color='orange', linewidth=2, linestyle='--')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (SEK/kWh)', fontsize=12)
    plt.title(f'Electricity Price Forecast - {city}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def delete_feature_groups(fs, feature_group_names):
    for fg_name in feature_group_names:
        try:
            fg = fs.get_feature_group(fg_name)
            fg.delete()
            print(f"Deleted feature group: {fg_name}")
        except Exception as e:
            print(f"Could not delete {fg_name}: {e}")


def delete_feature_views(fs, feature_view_names):
    for fv_name in feature_view_names:
        try:
            fv = fs.get_feature_view(fv_name)
            fv.delete()
            print(f"Deleted feature view: {fv_name}")
        except Exception as e:
            print(f"Could not delete {fv_name}: {e}")


def purge_project(project):
    """Clean up all Hopsworks artifacts (use with caution!)."""
    fs = project.get_feature_store()

    try:
        for fv in fs.get_feature_views():
            try:
                fv.delete()
                print(f"Deleted feature view: {fv.name}")
            except:
                pass
    except Exception as e:
        print(f"Error deleting feature views: {e}")

    try:
        for fg in fs.get_feature_groups():
            try:
                fg.delete()
                print(f"Deleted feature group: {fg.name}")
            except:
                pass
    except Exception as e:
        print(f"Error deleting feature groups: {e}")


def check_file_path(file_path):
    return os.path.exists(file_path)
