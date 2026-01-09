#!/usr/bin/env python3
"""
Inference Pipeline - Generate 7-day electricity price forecast

Works in both local and production modes:
  --mode local       : Load model from local files
  --mode production  : Load from Hopsworks Model Registry

Model types:
  --model-type xgboost : Use XGBoost model (default)
  --model-type lstm    : Use LSTM model

Usage:
    python pipelines/inference_pipeline.py --mode local --days 7 --model-type xgboost
    python pipelines/inference_pipeline.py --mode production --days 7 --model-type lstm
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import json
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from functions.util import get_weather_forecast
from functions.storage_factory import get_storage, detect_mode


def load_model_local(experiment_name='default'):
    model_dir = f"data/models/electricity_price_xgboost_{experiment_name}"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found: {model_dir}\n"
            f"Train a model first: python pipelines/training_pipeline.py --mode local"
        )

    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "model.json"))

    with open(os.path.join(model_dir, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)

    print(f"  Model loaded from: {model_dir}")
    return model, feature_names


def load_model_hopsworks(storage):
    mr = storage.get_model_registry()

    model_name = "electricity_price_xgboost"
    model = mr.get_model(model_name, version=1)

    # Download model files
    model_dir = model.download()

    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(model_dir, "model.json"))

    with open(os.path.join(model_dir, "feature_names.json"), 'r') as f:
        feature_names = json.load(f)

    print(f"  Model loaded from Hopsworks: {model_name}")
    return xgb_model, feature_names


def load_lstm_model_local(experiment_name='default'):
    model_dir = f"data/models/electricity_price_lstm_{experiment_name}"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"LSTM model not found: {model_dir}\n"
            f"Train a model first: python pipelines/training_pipeline.py --mode local --model-type lstm"
        )

    # Import TensorFlow
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

    model = keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))

    with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config = json.load(f)

    fn_path = os.path.join(model_dir, "feature_names.json")
    if os.path.exists(fn_path):
        with open(fn_path, "r") as f:
            config["feature_names"] = json.load(f)
    else:
        if "feature_names" not in config:
            raise FileNotFoundError(f"Missing feature_names.json in {model_dir} (needed for LSTM inference)")

    print(f"  LSTM model loaded from: {model_dir}")
    print(f"     Lookback: {config['lookback']} days, Features: {config['n_features']}")

    return model, scaler, config


def load_lstm_model_hopsworks(storage):
    mr = storage.get_model_registry()

    model_name = "electricity_price_lstm"
    model_meta = mr.get_model(model_name, version=1)

    # Download model files
    model_dir = model_meta.download()

    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

    model = keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))

    with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, "config.json"), 'r') as f:
        config = json.load(f)
        fn_path = os.path.join(model_dir, "feature_names.json")

    if os.path.exists(fn_path):
        with open(fn_path, "r") as f:
            config["feature_names"] = json.load(f)
    else:
        if "feature_names" not in config:
            raise FileNotFoundError(f"Missing feature_names.json in downloaded model dir {model_dir}")

    print(f"  LSTM model loaded from Hopsworks: {model_name}")
    print(f"     Lookback: {config['lookback']} days, Features: {config['n_features']}")

    return model, scaler, config


def prepare_forecast_features(weather_forecast_df, historical_df, feature_names):
    """Prepare features for forecast with weather data and lag features"""
    forecast_df = weather_forecast_df.copy()
    forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.tz_localize(None)

    historical_df = historical_df.copy()
    historical_df['date'] = pd.to_datetime(historical_df['date']).dt.tz_localize(None)

    forecast_df['hour'] = forecast_df['date'].dt.hour
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['is_weekend'] = (forecast_df['day_of_week'] >= 5).astype(int)

    forecast_df['temp_squared'] = forecast_df['temperature_2m_mean'] ** 2
    forecast_df['wind_temp_interaction'] = (
        forecast_df['wind_speed_10m_max'] * forecast_df['temperature_2m_mean']
    )

    recent_prices = historical_df['price_sek_kwh_mean'].tail(200).values

    if len(recent_prices) >= 24:
        forecast_df['price_lag_1d'] = recent_prices[-24]
    else:
        forecast_df['price_lag_1d'] = recent_prices[-1]

    if len(recent_prices) >= 168:
        forecast_df['price_lag_7d'] = recent_prices[-168]
    else:
        forecast_df['price_lag_7d'] = recent_prices[-1]

    if len(recent_prices) >= 168:
        forecast_df['price_rolling_mean_7d'] = np.mean(recent_prices[-168:])
        forecast_df['price_rolling_std_7d'] = np.std(recent_prices[-168:])
    else:
        forecast_df['price_rolling_mean_7d'] = np.mean(recent_prices)
        forecast_df['price_rolling_std_7d'] = np.std(recent_prices)

    for feature in feature_names:
        if feature not in forecast_df.columns:
            forecast_df[feature] = 0

    return forecast_df[feature_names]


def create_forecast_visualization(forecast_df, output_path='outputs/forecast.png'):
    os.makedirs('outputs', exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(forecast_df['date'], forecast_df['predicted_price'],
            label='7-Day Forecast', color='#E63946', linewidth=3,
            marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)

    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (SEK/kWh)', fontsize=13, fontweight='bold')
    ax.set_title('Electricity Price Forecast - Stockholm (SE3)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    for i, (date, price) in enumerate(zip(forecast_df['date'], forecast_df['predicted_price'])):
        ax.text(date, price + 0.01, f'{price:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Forecast chart saved: {output_path}")

    return output_path


def create_comparison_visualization(historical_df, output_path='outputs/predicted_vs_actual.png'):
    os.makedirs('outputs', exist_ok=True)

    tracking_file = 'outputs/prediction_tracking.csv'
    if not os.path.exists(tracking_file):
        print(f"  No tracking data yet. Run daily to build comparison history.")
        return None

    tracking_df = pd.read_csv(tracking_file)
    tracking_df['target_date'] = (
        pd.to_datetime(tracking_df['target_date'], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    tracking_df['prediction_date'] = (
        pd.to_datetime(tracking_df['prediction_date'], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Get actual prices from historical data
    historical_df['date'] = (
        pd.to_datetime(historical_df['date'], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    actual_prices = historical_df[['date', 'price_sek_kwh_mean']].copy()
    actual_prices.columns = ['target_date', 'actual_price']

    comparison_df = tracking_df.merge(actual_prices, on='target_date', how='inner')

    if comparison_df.empty:
        print(f"  No actual data available yet for comparison.")
        return None

    comparison_df = comparison_df.sort_values(['target_date', 'prediction_date'])
    comparison_df = comparison_df.drop_duplicates('target_date', keep='last')
    comparison_df = comparison_df.sort_values('target_date')

    os.makedirs('outputs', exist_ok=True)

    comparison_out = comparison_df[['target_date', 'predicted_price', 'actual_price', 'prediction_date']].copy()

    comparison_out['target_date'] = pd.to_datetime(comparison_out['target_date']).dt.strftime('%Y-%m-%d')
    comparison_out['prediction_date'] = pd.to_datetime(comparison_out['prediction_date']).dt.strftime('%Y-%m-%d')

    comparison_out.to_csv('outputs/comparison_timeseries.csv', index=False)

    comparison_df['error'] = comparison_df['predicted_price'] - comparison_df['actual_price']
    comparison_df['abs_error'] = comparison_df['error'].abs()
    mae = comparison_df['abs_error'].mean()
    rmse = np.sqrt((comparison_df['error'] ** 2).mean())

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(comparison_df['target_date'], comparison_df['predicted_price'],
            label='Predicted', color='#E63946', linewidth=3,
            marker='o', markersize=8, alpha=0.9)
    ax.plot(comparison_df['target_date'], comparison_df['actual_price'],
            label='Actual', color='#2E86AB', linewidth=3,
            marker='s', markersize=8, alpha=0.9)

    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (SEK/kWh)', fontsize=13, fontweight='bold')
    ax.set_title('Predicted vs Actual Electricity Prices - Stockholm (SE3)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45, ha='right')

    metrics_text = f'MAE: {mae:.4f} SEK/kWh | RMSE: {rmse:.4f} SEK/kWh | Days: {len(comparison_df)}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Comparison chart saved: {output_path}")
    print(f"     MAE: {mae:.4f} SEK/kWh | RMSE: {rmse:.4f} SEK/kWh | {len(comparison_df)} days")

    return output_path


def save_predictions_for_tracking(forecast_df, prediction_date):
    tracking_file = 'outputs/prediction_tracking.csv'

    tracking_data = forecast_df[['date', 'predicted_price']].copy()
    tracking_data.columns = ['target_date', 'predicted_price']
    tracking_data['prediction_date'] = prediction_date
    tracking_data['prediction_date'] = pd.to_datetime(tracking_data['prediction_date'])
    tracking_data['target_date'] = pd.to_datetime(tracking_data['target_date'])

    if os.path.exists(tracking_file):
        existing_df = pd.read_csv(tracking_file)
        existing_df['prediction_date'] = pd.to_datetime(existing_df['prediction_date'])
        existing_df['target_date'] = pd.to_datetime(existing_df['target_date'])

        existing_df = existing_df[~existing_df['target_date'].isin(tracking_data['target_date'])]

        combined_df = pd.concat([existing_df, tracking_data], ignore_index=True)
        combined_df = combined_df.sort_values(['target_date', 'prediction_date'])
        combined_df.to_csv(tracking_file, index=False)
    else:
        tracking_data.to_csv(tracking_file, index=False)

    print(f"  Predictions saved to tracking file: {tracking_file}")


def predict_with_lstm(lstm_model, scaler, config, historical_df, weather_forecast_df):
    """Generate predictions using LSTM model with iterative forecasting"""
    lookback = config['lookback']
    feature_names = config.get('feature_names', historical_df.columns.tolist())

    if 'date' not in historical_df.columns:
        raise ValueError("Historical data must have 'date' column")

    historical_df = historical_df.sort_values('date').reset_index(drop=True)

    forecast_df = weather_forecast_df.copy()
    forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.tz_localize(None)

    historical_df = historical_df.copy()
    historical_df['date'] = pd.to_datetime(historical_df['date']).dt.tz_localize(None)

    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['is_weekend'] = (forecast_df['day_of_week'] >= 5).astype('int32')
    forecast_df['day_of_year'] = forecast_df['date'].dt.dayofyear

    forecast_df['temp_squared'] = forecast_df['temperature_2m_mean'] ** 2
    forecast_df['wind_temp_interaction'] = (
        forecast_df['wind_speed_10m_max'] * forecast_df['temperature_2m_mean']
    )

    last_price = historical_df['price_sek_kwh_mean'].iloc[-1]
    forecast_df['price_sek_kwh_mean'] = last_price
    forecast_df['price_lag_1d'] = last_price
    forecast_df['price_lag_7d'] = last_price
    forecast_df['price_rolling_mean_7d'] = last_price
    forecast_df['price_rolling_std_7d'] = 0.0

    feature_cols = config['feature_names']

    for col in feature_cols:
        if col not in historical_df.columns:
            historical_df[col] = 0.0

    for col in feature_cols:
        if col not in forecast_df.columns:
            if col in historical_df.columns:
                forecast_df[col] = historical_df[col].mean()
            else:
                forecast_df[col] = 0.0

    col_order = ['date'] + feature_cols

    print(f"  Debug: historical_df shape before selection: {historical_df.shape}")
    print(f"  Debug: forecast_df shape before selection: {forecast_df.shape}")
    print(f"  Debug: col_order length: {len(col_order)}")
    print(f"  Debug: Missing in historical_df: {set(col_order) - set(historical_df.columns)}")
    print(f"  Debug: Missing in forecast_df: {set(col_order) - set(forecast_df.columns)}")

    forecast_df = forecast_df[col_order].copy()
    historical_df = historical_df[col_order].copy()

    historical_df['date'] = pd.to_datetime(historical_df['date']).astype('datetime64[ns]')
    forecast_df['date'] = pd.to_datetime(forecast_df['date']).astype('datetime64[ns]')

    print(f"  Aligned dataframes: historical ({historical_df.shape[0]} rows), forecast ({forecast_df.shape[0]} rows)")

    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    data_df = combined_df[feature_cols].copy()
    data_df = data_df.replace([np.inf, -np.inf], np.nan)
    data_df = data_df.fillna(method="ffill").fillna(method="bfill")
    data_df = data_df.fillna(data_df.mean(numeric_only=True))

    if data_df.isna().any().any():
        bad = data_df.isna().sum().sort_values(ascending=False)
        raise ValueError(f"NaNs still present after cleaning:\n{bad[bad > 0].head(20)}")

    data = data_df.values

    data_scaled = scaler.transform(data)

    hist_len = len(historical_df)

    target_col = 'price_sek_kwh_mean'
    target_idx = feature_cols.index(target_col)

    predictions = []

    for i in range(len(forecast_df)):
        end_idx = hist_len + i
        start_idx = end_idx - lookback

        if start_idx < 0:
            print(f"  Warning: Not enough historical data for full lookback window")
            start_idx = 0

        sequence = data_scaled[start_idx:end_idx, :]

        if len(sequence) < lookback:
            padding = np.zeros((lookback - len(sequence), sequence.shape[1]))
            sequence = np.vstack([padding, sequence])

        sequence = sequence.reshape(1, lookback, -1)

        pred_scaled = lstm_model.predict(sequence, verbose=0)[0][0]

        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, target_idx] = pred_scaled
        pred_denorm = scaler.inverse_transform(dummy)[0, target_idx]

        predictions.append(pred_denorm)

        if i < len(forecast_df) - 1:
            data_scaled[end_idx, target_idx] = pred_scaled

    result_df = forecast_df[['date']].copy()
    result_df['predicted_price'] = predictions

    return result_df


def main():
    parser = argparse.ArgumentParser(description='Inference Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'production'],
        default=None,
        help='Storage mode. Auto-detects if not specified.'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to forecast (excluding today)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Experiment name (for local mode)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['xgboost', 'lstm'],
        default='xgboost',
        help='Model type: xgboost or lstm (default: xgboost)'
    )
    parser.add_argument(
        '--latitude',
        type=float,
        default=59.33,
        help='Latitude for weather forecast'
    )
    parser.add_argument(
        '--longitude',
        type=float,
        default=18.07,
        help='Longitude for weather forecast'
    )

    args = parser.parse_args()

    mode = args.mode if args.mode else detect_mode()
    print(f"\n{'='*70}")
    print(f"INFERENCE PIPELINE - Mode: {mode.upper()}, Model: {args.model_type.upper()}")
    print(f"{'='*70}")
    print(f"Forecast horizon: {args.days} days")

    print(f"\n[1/5] Loading {args.model_type.upper()} model from {mode} storage...")

    if args.model_type == 'xgboost':
        if mode == 'local':
            model, feature_names = load_model_local(args.experiment_name)
            scaler = None
            config = None
        else:
            storage = get_storage(mode)
            model, feature_names = load_model_hopsworks(storage)
            scaler = None
            config = None
    else:  # lstm
        if mode == 'local':
            model, scaler, config = load_lstm_model_local(args.experiment_name)
            feature_names = config.get('feature_names', [])
        else:
            storage = get_storage(mode)
            model, scaler, config = load_lstm_model_hopsworks(storage)
            feature_names = config.get('feature_names', [])

    print(f"\n[2/5] Loading historical data...")
    storage = get_storage(mode)
    fs = storage.get_feature_store()

    electricity_fg = fs.get_or_create_feature_group(name="electricity_price", version=1)
    historical_df = electricity_fg.read()
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df = historical_df.sort_values('date')

    print(f"  Loaded {len(historical_df)} historical records")
    print(f"  Latest data: {historical_df['date'].max()}")

    print(f"\n[3/5] Fetching weather forecast...")
    weather_forecast = get_weather_forecast(
        days_ahead=args.days + 3,
        latitude=args.latitude,
        longitude=args.longitude
    )
    weather_forecast['date'] = pd.to_datetime(weather_forecast['date'])

    tomorrow = (datetime.now() + timedelta(days=1)).date()
    weather_forecast = weather_forecast[weather_forecast['date'].dt.date >= tomorrow].head(args.days)

    print(f"  Retrieved {len(weather_forecast)} days of forecast")

    print(f"\n[4/5] Generating predictions...")

    if args.model_type == 'xgboost':
        forecast_features = prepare_forecast_features(weather_forecast, historical_df, feature_names)
        predictions = model.predict(xgb.DMatrix(forecast_features))

        forecast_df = weather_forecast[['date']].copy()
        forecast_df['predicted_price'] = predictions
    else:
        forecast_df = predict_with_lstm(model, scaler, config, historical_df, weather_forecast)

    print(f"  Generated {len(forecast_df)} predictions")
    print(f"\n  Forecast Summary:")
    print(f"     Date range: {forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")
    print(f"     Avg price:  {forecast_df['predicted_price'].mean():.3f} SEK/kWh")
    print(f"     Min price:  {forecast_df['predicted_price'].min():.3f} SEK/kWh")
    print(f"     Max price:  {forecast_df['predicted_price'].max():.3f} SEK/kWh")

    print(f"\n[5/5] Saving results...")

    os.makedirs("outputs", exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    csv_path = f"outputs/forecast_{today_str}.csv"
    forecast_df.to_csv(csv_path, index=False)
    print(f"  Forecast saved: {csv_path}")

    prediction_date = datetime.now().date()
    save_predictions_for_tracking(forecast_df, prediction_date)

    forecast_chart_path = f"outputs/forecast_{today_str}.png"
    create_forecast_visualization(forecast_df, forecast_chart_path)

    comparison_chart_path = f"outputs/predicted_vs_actual_{today_str}.png"
    create_comparison_visualization(historical_df, comparison_chart_path)

    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  - Forecast data: {csv_path}")
    print(f"  - Forecast chart: {forecast_chart_path}")
    if os.path.exists(comparison_chart_path):
        print(f"  - Comparison chart: {comparison_chart_path}")
    print(f"\nRun daily to build prediction vs actual comparison history!")


if __name__ == '__main__':
    main()
