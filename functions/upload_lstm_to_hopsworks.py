#!/usr/bin/env python3
"""
Upload existing LSTM model to Hopsworks Model Registry

Usage:
    python3 upload_lstm_to_hopsworks.py --model-dir <path_to_model>
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from functions.storage_factory import get_storage


def upload_lstm_model(model_dir, experiment_name='default'):
    print(f"\n{'='*70}")
    print(f"UPLOAD LSTM MODEL TO HOPSWORKS")
    print(f"{'='*70}")
    print(f"Model directory: {model_dir}")
    print(f"Experiment: {experiment_name}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    required_files = ['lstm_model.keras', 'scaler.pkl', 'config.json']
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file missing: {file_path}")

    print(f"\n[1/3] Loading model metadata...")

    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    print(f"  Model type: LSTM")
    print(f"  Lookback: {config.get('lookback', 14)} days")
    print(f"  Features: {config.get('n_features', 'unknown')}")
    print(f"  Test R²: {config.get('test_r2', 'unknown'):.4f}")
    print(f"  Test MAE: {config.get('test_mae', 'unknown'):.4f} SEK/kWh")

    metrics = {
        'train_rmse': float(config.get('train_rmse', 0)),
        'train_mae': float(config.get('train_mae', 0)),
        'train_r2': float(config.get('train_r2', 0)),
        'test_rmse': float(config.get('test_rmse', 0)),
        'test_mae': float(config.get('test_mae', 0)),
        'test_r2': float(config.get('test_r2', 0)),
        'lookback': int(config.get('lookback', 14)),
        'n_features': int(config.get('n_features', 0)),
        'epochs_trained': int(config.get('epochs_trained', 0)),
    }

    print(f"\n[2/3] Connecting to Hopsworks...")
    storage = get_storage('production')
    mr = storage.get_model_registry()
    print(f"  Connected to Hopsworks Model Registry")

    print(f"\n[3/3] Uploading model to Hopsworks...")

    temp_dir = tempfile.mkdtemp()

    try:
        for file in os.listdir(model_dir):
            src = os.path.join(model_dir, file)
            dst = os.path.join(temp_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        model_name = "electricity_price_lstm"
        trained_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        registered_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=(
                f"LSTM model for electricity price prediction - {experiment_name} "
                f"(trained_at={trained_at}, lookback={config.get('lookback', 14)}d)"
            ),
        )

        registered_model.save(temp_dir)

        print(f"  Model uploaded to Hopsworks Model Registry!")
        print(f"     Model name: {model_name}")
        print(f"     Version: {registered_model.version}")

    finally:
        shutil.rmtree(temp_dir)

    print(f"\n{'='*70}")
    print(f"UPLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  - Check model in Hopsworks UI: Model Registry → {model_name}")
    print(f"  - Run inference: python3 pipelines/inference_pipeline.py --mode production --model-type lstm --days 7")


def main():
    parser = argparse.ArgumentParser(description='Upload LSTM model to Hopsworks')
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Path to local LSTM model directory'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='default',
        help='Experiment name for organizing models'
    )

    args = parser.parse_args()

    upload_lstm_model(args.model_dir, args.experiment_name)


if __name__ == '__main__':
    main()
