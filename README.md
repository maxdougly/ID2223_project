# Electricity Price Predictor

This repository contains an end-to-end machine learning project in the Scalable Machine Learning course (ID2223). The project combines weather data with historical electricity prices to produce 7-day forecasts for the Stockholm region (SE3) and includes utilities for data collection, model training (XGBoost or LSTM), and daily batch inference with output visualization.

**Models:** The system supports two model architectures:
- XGBoost: Gradient boosting model using tree-based learning with hyperparameter tuning
- LSTM: Recurrent neural network with 14-day lookback window, using sequence-based learning with MinMaxScaler normalization

**Features:** The prediction models use 20+ engineered features combining raw data with derived patterns:
- Weather data (OpenMeteo API): temperature (mean, max, min), precipitation, wind speed, wind direction, solar radiation
- Electricity prices (elprisetjustnu.se API): daily mean, min, max, and standard deviation
- Temporal features: day of week, month, weekend indicator, day of year
- Lag features: 1-day and 7-day price history
- Rolling statistics: 7-day rolling mean and standard deviation of prices
- Interaction features: polynomial temperature terms, wind-temperature interactions

Authors: Max Dougly, Erik Forsell

---

## Important links

- Live app: [HuggingFace Space](https://huggingface.co/spaces/maxdougly/Electricity_price_predictor)
- Feature store: [Hopsworks Project](https://c.app.hopsworks.ai/p/1333397/view)
- Weather API: [OpenMeteo](https://open-meteo.com/)
- Price API: [elprisetjustnu.se](https://www.elprisetjustnu.se/api/v1/prices/)

---

## Components

### Data sources
- OpenMeteo API: Historical and forecast weather data (temperature, wind, precipitation, solar radiation).
- elprisetjustnu.se API: Swedish electricity spot prices for the SE3 region.

### Feature pipeline
Collects weather and price data from APIs, engineers features including temporal patterns, lag features (1-day, 7-day) and rolling statistics, and writes results to the feature store or local storage. Runs daily via GitHub Actions to backfill historical data from 9 days ago to 2 days ago.

### Feature store
Hopsworks feature store storing engineered features in the `electricity_price` feature group (version 1). Serves as the central repository for all processed features used in model training and inference. Project ID: 1333397.

### Model registry
Hopsworks Model Registry storing trained models (XGBoost and LSTM). Models are versioned and include performance metrics. The inference pipeline automatically retrieves the latest model version for daily predictions.

### Training pipeline
Trains models (XGBoost or LSTM) using historical features from the feature store. Creates train/test splits, tunes hyperparameters, evaluates model performance, and registers trained models to the model registry. Training can be run locally or in production mode.

### Inference pipeline
Generates 7-day price forecasts by loading the latest model from the registry, preparing forecast features from weather predictions, and producing predictions. Exports CSV and PNG outputs, and appends predictions to a tracking file for later comparison with actuals. Runs daily via GitHub Actions.

### Storage factory
Provides a unified interface for local Parquet storage and Hopsworks Feature Store access. Allows seamless switching between local development and production deployment without changing pipeline code.

### GitHub Actions workflow
Automates daily feature backfill and inference tasks at 18:00 UTC. Commits generated outputs (CSV and PNG files) to the repository and syncs them to the HuggingFace Space for visualization.

### HuggingFace Space
A Gradio web UI that displays forecast charts and prediction tracking. Outputs are synchronized to the Space on a daily schedule, providing an interactive interface for viewing electricity price predictions.

---

## Automated jobs and routines

| Task | Schedule | Description |
|------|----------|-------------|
| Feature backfill | Daily at 18:00 UTC | Collects data from 8 days ago to 1 day ago and appends to feature store. |
| Inference | Daily at 18:00 UTC | Generates 7-day forecast and comparison charts. |
| Commit outputs | Daily at 18:00 UTC | Commits CSV and PNG files to repository. |
| Upload to HuggingFace | Daily at 18:00 UTC | Syncs outputs folder to the HuggingFace Space. |

Workflow configuration: `.github/workflows/electricity-price-daily.yml`

---

## Quick start

A short quick start guide, the commands below are the minimum necessary to run the project in local mode.

### Installation

```bash
git clone https://github.com/E4Sell/sml_project.git
cd sml_project
pip install -r requirements.txt
```

### Local run (minimal)

```bash
# Collect recent data (local mode)
python pipelines/feature_backfill.py --mode local --start-date 2024-11-01

# Train a model (default is XGBoost; add --model-type lstm to train LSTM)
python pipelines/training_pipeline.py --mode local

# Generate a 7-day forecast
python pipelines/inference_pipeline.py --mode local --days 7
```
