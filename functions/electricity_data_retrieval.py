"""Retrieve electricity price data and predictions from Hopsworks Feature Store."""

import pandas as pd
from datetime import datetime, timedelta


def get_historical_data_for_date(feature_view, date_str, city="Stockholm"):
    """Retrieve historical electricity price data for specific date."""
    try:
        # Get batch data from feature view
        df = feature_view.get_batch_data()

        # Filter for specific date and city
        df['date'] = pd.to_datetime(df['date']).dt.date
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        filtered_df = df[(df['date'] == target_date) & (df['city'] == city)]

        # Sort by date
        filtered_df = filtered_df.sort_values('date')

        return filtered_df

    except Exception as e:
        print(f"Error retrieving historical data for {date_str}: {e}")
        return pd.DataFrame()


def get_historical_data_in_date_range(feature_view, start_date_str, end_date_str, city="Stockholm"):
    """Retrieve historical electricity price data for date range."""
    try:
        # Get batch data from feature view
        df = feature_view.get_batch_data()

        # Convert date column to datetime for filtering
        df['date'] = pd.to_datetime(df['date']).dt.date
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        # Filter by date range and city
        filtered_df = df[
            (df['date'] >= start_date) &
            (df['date'] <= end_date) &
            (df['city'] == city)
        ]

        # Sort by date
        filtered_df = filtered_df.sort_values('date')

        # Format date as string for consistency
        filtered_df['date'] = filtered_df['date'].astype(str)

        return filtered_df

    except Exception as e:
        print(f"Error retrieving historical data for range {start_date_str} to {end_date_str}: {e}")
        return pd.DataFrame()


def get_future_data_for_date(feature_view, model, date_str, city="Stockholm"):
    """Generate electricity price prediction for specific future date."""
    try:
        # Get forecast data from weather feature group
        # This assumes weather forecasts have been written to a feature group
        df = feature_view.get_batch_data()

        # Filter for the specific date
        df['date'] = pd.to_datetime(df['date']).dt.date
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()

        forecast_df = df[(df['date'] == target_date) & (df['city'] == city)]

        if forecast_df.empty:
            print(f"No forecast data available for {date_str}")
            return pd.DataFrame()

        # Drop non-feature columns
        feature_cols = [col for col in forecast_df.columns if col not in ['date', 'city']]
        X = forecast_df[feature_cols]

        # Generate prediction
        predictions = model.predict(X)

        result_df = pd.DataFrame({
            'date': [target_date],
            'predicted_price': predictions
        })

        result_df['date'] = result_df['date'].astype(str)

        return result_df

    except Exception as e:
        print(f"Error generating prediction for {date_str}: {e}")
        return pd.DataFrame()


def get_future_data_in_date_range(feature_view, model, start_date_str, end_date_str, city="Stockholm"):
    """Generate electricity price predictions for date range."""
    try:
        # Get forecast data
        df = feature_view.get_batch_data()

        # Convert and filter dates
        df['date'] = pd.to_datetime(df['date']).dt.date
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        forecast_df = df[
            (df['date'] >= start_date) &
            (df['date'] <= end_date) &
            (df['city'] == city)
        ]

        if forecast_df.empty:
            print(f"No forecast data available for range {start_date_str} to {end_date_str}")
            return pd.DataFrame()

        # Drop non-feature columns
        feature_cols = [col for col in forecast_df.columns if col not in ['date', 'city']]
        X = forecast_df[feature_cols]

        # Generate predictions
        predictions = model.predict(X)

        result_df = pd.DataFrame({
            'date': forecast_df['date'].values,
            'predicted_price': predictions
        })

        # Sort by date
        result_df = result_df.sort_values('date')
        result_df['date'] = result_df['date'].astype(str)

        return result_df

    except Exception as e:
        print(f"Error generating predictions for range {start_date_str} to {end_date_str}: {e}")
        return pd.DataFrame()


def get_predictions_from_feature_group(fs, fg_name="electricity_price_predictions", city="Stockholm", days=7):
    """Retrieve latest predictions from predictions feature group."""
    try:
        fg = fs.get_feature_group(fg_name)
        df = fg.read()

        # Filter for city and recent dates
        df['date'] = pd.to_datetime(df['date'])
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_df = df[(df['city'] == city) & (df['date'] >= cutoff_date)]

        # Sort by date
        filtered_df = filtered_df.sort_values('date')

        return filtered_df

    except Exception as e:
        print(f"Error retrieving predictions from feature group: {e}")
        return pd.DataFrame()
