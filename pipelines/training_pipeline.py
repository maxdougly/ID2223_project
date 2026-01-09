#!/usr/bin/env python3
"""
Training Pipeline - Train XGBoost or LSTM model

Works in both local and production modes:
  --mode local       : Load from local Parquet, save model locally
  --mode production  : Load from Hopsworks, save to Model Registry

Model types:
  --model-type xgboost : Train XGBoost regression model (default)
  --model-type lstm    : Train LSTM neural network model

Usage:
    python pipelines/training_pipeline.py --mode local --model-type xgboost
    python pipelines/training_pipeline.py --mode production --model-type lstm
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import json
import os
import pickle
from datetime import datetime
from functions.storage_factory import get_storage, detect_mode


def prepare_training_data(df):
    """Prepare XGBoost training data with time-based split"""
    # Target variable
    target_col = 'price_sek_kwh_mean'

    # Features (exclude target and date)
    feature_cols = [col for col in df.columns if col not in [target_col, 'date']]

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split (no shuffle to prevent data leakage)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


def prepare_lstm_sequences(df, lookback=14):
    """Prepare LSTM sequences with normalization and lookback window"""
    # Target variable
    target_col = 'price_sek_kwh_mean'

    # Features (exclude date)
    feature_cols = [col for col in df.columns if col not in ['date']]

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Extract features and target
    data = df[feature_cols].copy()

    # Clean data before scaling
    print(f"  Cleaning data...")
    print(f"    NaN values before: {data.isna().sum().sum()}")

    data = data.fillna(method='ffill').fillna(method='bfill')
    data = data.fillna(data.mean())

    print(f"    NaN values after: {data.isna().sum().sum()}")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean())

    # Convert to numpy
    data = data.values

    # Scale data (LSTM works better with normalized data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Verify no NaN in scaled data
    if np.isnan(data_scaled).any():
        raise ValueError("NaN values present after scaling! Cannot train model.")

    # Create sequences
    X_sequences = []
    y_sequences = []

    target_idx = feature_cols.index(target_col)

    for i in range(lookback, len(data_scaled)):
        # X: last 'lookback' days of all features
        X_sequences.append(data_scaled[i-lookback:i, :])
        # y: target price for current day
        y_sequences.append(data_scaled[i, target_idx])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Time-based split (80/20)
    split_idx = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_train = y_sequences[:split_idx]
    y_test = y_sequences[split_idx:]

    print(f"  Training sequences: {len(X_train)}")
    print(f"  Test sequences: {len(X_test)}")
    print(f"  Sequence shape: {X_train.shape}")
    print(f"  Lookback window: {lookback} days")
    print(f"  Features per timestep: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost regression model with early stopping"""
    print("  Training XGBoost model...")

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        'model_type': 'xgboost',
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'n_training_samples': int(len(X_train)),
        'n_test_samples': int(len(X_test)),
        'best_iteration': int(
            model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        ),
    }

    return model, metrics


def train_lstm_model(X_train, y_train, X_test, y_test, scaler):
    """Train stacked LSTM model with dropout and early stopping"""
    print("  Building LSTM architecture...")

    # Import TensorFlow/Keras
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError(
            "TensorFlow not installed. Install with: pip install tensorflow"
        )

    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Build LSTM model
    model = keras.Sequential([
        # First LSTM layer with return sequences
        layers.LSTM(
            units=64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            input_shape=(X_train.shape[1], X_train.shape[2])
        ),

        # Second LSTM layer
        layers.LSTM(
            units=64,
            dropout=0.2,
            recurrent_dropout=0.2
        ),

        # Dense output layer
        layers.Dense(units=1)
    ])

    # Compile model with gradient clipping
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    print(f"  LSTM Architecture:")
    print(f"    - Layer 1: LSTM(64 units, dropout=0.2, recurrent_dropout=0.2)")
    print(f"    - Layer 2: LSTM(64 units, dropout=0.2, recurrent_dropout=0.2)")
    print(f"    - Output: Dense(1 unit)")
    print(f"    - Optimizer: Adam, Loss: MSE")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0
    )

    print("  Training LSTM model...")
    print("  (Monitoring epoch progress...)\n")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1  # Show epoch progress
    )

    # Predictions
    y_pred_train_scaled = model.predict(X_train, verbose=0).flatten()
    y_pred_test_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform predictions back to original scale
    # Create dummy arrays with same shape as original data for inverse transform
    n_features = X_train.shape[2]

    # For training predictions
    dummy_train = np.zeros((len(y_pred_train_scaled), n_features))
    dummy_train[:, 0] = y_pred_train_scaled  # Assuming target is first feature after scaling
    y_pred_train = scaler.inverse_transform(dummy_train)[:, 0]

    dummy_train_actual = np.zeros((len(y_train), n_features))
    dummy_train_actual[:, 0] = y_train
    y_train_actual = scaler.inverse_transform(dummy_train_actual)[:, 0]

    # For test predictions
    dummy_test = np.zeros((len(y_pred_test_scaled), n_features))
    dummy_test[:, 0] = y_pred_test_scaled
    y_pred_test = scaler.inverse_transform(dummy_test)[:, 0]

    dummy_test_actual = np.zeros((len(y_test), n_features))
    dummy_test_actual[:, 0] = y_test
    y_test_actual = scaler.inverse_transform(dummy_test_actual)[:, 0]

    # Calculate metrics
    metrics = {
        'model_type': 'lstm',
        'train_rmse': float(np.sqrt(mean_squared_error(y_train_actual, y_pred_train))),
        'train_mae': float(mean_absolute_error(y_train_actual, y_pred_train)),
        'train_r2': float(r2_score(y_train_actual, y_pred_train)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test_actual, y_pred_test))),
        'test_mae': float(mean_absolute_error(y_test_actual, y_pred_test)),
        'test_r2': float(r2_score(y_test_actual, y_pred_test)),
        'n_training_samples': int(len(X_train)),
        'n_test_samples': int(len(X_test)),
        'epochs_trained': int(len(history.history['loss'])),
        'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
    }

    return model, metrics


def save_xgboost_model_local(model, feature_names, metrics, experiment_name='default'):
    model_dir = f"data/models/electricity_price_xgboost_{experiment_name}"
    os.makedirs(model_dir, exist_ok=True)

    model.save_model(os.path.join(model_dir, "model.json"))

    with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f)

    with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model saved to: {model_dir}")
    return model_dir


def save_lstm_model_local(model, scaler, feature_names, metrics, experiment_name='default', lookback=14):
    model_dir = f"data/models/electricity_price_lstm_{experiment_name}"
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(model_dir, "lstm_model.keras"))

    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f)

    config = {
        'lookback': lookback,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        **metrics
    }
    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Model saved to: {model_dir}")
    return model_dir


def save_xgboost_model_hopsworks(model, feature_names, metrics, storage, experiment_name='default'):
    mr = storage.get_model_registry()

    import tempfile
    import shutil
    model_dir = tempfile.mkdtemp()

    try:
        model.save_model(os.path.join(model_dir, "model.json"))

        with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
            json.dump(feature_names, f)

        with open(os.path.join(model_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

        model_name = f"electricity_price_xgboost"
        trained_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        registered_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=(
                f"XGBoost model for electricity price prediction - {experiment_name} "
                f"(trained_at={trained_at})"
            ),
        )

        registered_model.save(model_dir)
        print(f"  Model saved to Hopsworks Model Registry: {model_name}")

    finally:
        shutil.rmtree(model_dir)


def save_lstm_model_hopsworks(model, scaler, feature_names, metrics, storage, experiment_name='default', lookback=14):
    mr = storage.get_model_registry()

    import tempfile
    import shutil
    model_dir = tempfile.mkdtemp()

    try:
        model.save(os.path.join(model_dir, "lstm_model.keras"))

        with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)

        with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
            json.dump(feature_names, f)

        config = {
            'lookback': lookback,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            **metrics
        }
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        model_name = f"electricity_price_lstm"
        trained_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        registered_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=(
                f"LSTM model for electricity price prediction - {experiment_name} "
                f"(trained_at={trained_at}, lookback={lookback})"
            ),
        )

        registered_model.save(model_dir)
        print(f"  Model saved to Hopsworks Model Registry: {model_name}")

    finally:
        shutil.rmtree(model_dir)


def main():
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'production'],
        default=None,
        help='Storage mode: local or production. Auto-detects if not specified.'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['xgboost', 'lstm'],
        default='xgboost',
        help='Model type: xgboost or lstm (default: xgboost)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Experiment name for organizing models'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=14,
        help='Lookback window for LSTM (default: 14 days)'
    )

    args = parser.parse_args()

    mode = args.mode if args.mode else detect_mode()
    print(f"\n{'='*70}")
    print(f"TRAINING PIPELINE - Mode: {mode.upper()}, Model: {args.model_type.upper()}")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment_name}")

    print(f"\n[1/4] Loading data from {mode} storage...")
    storage = get_storage(mode)
    fs = storage.get_feature_store()

    electricity_fg = fs.get_or_create_feature_group(
        name="electricity_price",
        version=1
    )

    df = electricity_fg.read()
    print(f"  Loaded {len(df)} records")

    # Hold out last 30 days for validation/test
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    cutoff_date = df['date'].max() - pd.Timedelta(days=30)
    df = df[df['date'] <= cutoff_date]
    print(f"  Using data up to {cutoff_date.date()} (excluding last 30 days)")
    print(f"  Training with {len(df)} records")

    print(f"\n[2/4] Preparing training data...")

    if args.model_type == 'xgboost':
        X_train, X_test, y_train, y_test, feature_names = prepare_training_data(df)
        scaler = None
    else:
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_lstm_sequences(
            df, lookback=args.lookback
        )

    print(f"\n[3/4] Training {args.model_type.upper()} model...")

    if args.model_type == 'xgboost':
        model, metrics = train_xgboost_model(X_train, y_train, X_test, y_test)
    else:
        model, metrics = train_lstm_model(X_train, y_train, X_test, y_test, scaler)

    print(f"\n  Model Performance:")
    print(f"     Train RMSE: {metrics['train_rmse']:.4f} SEK/kWh")
    print(f"     Test RMSE:  {metrics['test_rmse']:.4f} SEK/kWh")
    print(f"     Test MAE:   {metrics['test_mae']:.4f} SEK/kWh")
    print(f"     Test R²:    {metrics['test_r2']:.4f}")

    if args.model_type == 'lstm':
        print(f"     Epochs:     {metrics['epochs_trained']} (best: {metrics['best_epoch']})")

    print(f"\n[4/4] Saving model to {mode} storage...")

    if mode == 'local':
        if args.model_type == 'xgboost':
            model_path = save_xgboost_model_local(model, feature_names, metrics, args.experiment_name)
        else:
            model_path = save_lstm_model_local(
                model, scaler, feature_names, metrics, args.experiment_name, args.lookback
            )
    else:  # production
        if args.model_type == 'xgboost':
            save_xgboost_model_hopsworks(model, feature_names, metrics, storage, args.experiment_name)
        else:
            save_lstm_model_hopsworks(
                model, scaler, feature_names, metrics, storage, args.experiment_name, args.lookback
            )

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    if mode == 'local':
        print(f"  - Generate forecast: python pipelines/inference_pipeline.py --mode local --model-type {args.model_type}")
        if args.model_type == 'xgboost':
            print(f"  - View model: ls -lh {model_path}")
    else:
        print(f"  - Check Hopsworks Model Registry")
        print(f"  - Generate forecast: python pipelines/inference_pipeline.py --mode production --model-type {args.model_type}")


if __name__ == '__main__':
    main()
