#!/usr/bin/env python3
"""
Gradio UI for Electricity Price Predictor
Displays 7-day forecast and predicted vs actual comparison
"""

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import os

# Set matplotlib backend
plt.switch_backend('Agg')


def load_latest_forecast():
    """Load the most recent forecast data"""
    try:
        output_dir = Path("outputs")
        csv_files = list(output_dir.glob("forecast_*.csv"))

        if not csv_files:
            return None, "No forecast data available. Run inference pipeline first."

        # Get most recent forecast
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_csv)
        df['date'] = pd.to_datetime(df['date'])

        return df, None
    except Exception as e:
        return None, f"Error loading forecast: {str(e)}"


def load_comparison_data():
    """Load predicted vs actual comparison data"""
    try:
        tracking_file = Path("outputs/prediction_tracking.csv")

        if not tracking_file.exists():
            return None, "No tracking data available yet. Predictions will accumulate over time."

        df = pd.read_csv(tracking_file)
        df['target_date'] = pd.to_datetime(df['target_date'])
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])

        return df, None
    except Exception as e:
        return None, f"Error loading comparison data: {str(e)}"


def create_forecast_plot():
    """Generate forecast visualization"""
    df, error = load_latest_forecast()

    if df is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, error, ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot forecast
    ax.plot(df['date'], df['predicted_price'],
            label='7-Day Forecast', color='#E63946', linewidth=3,
            marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Add value labels
    for date, price in zip(df['date'], df['predicted_price']):
        ax.text(date, price + 0.01, f'{price:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (SEK/kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Electricity Price Forecast - Next 7 Days (SE3 Stockholm)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def create_comparison_plot():
    """Generate predicted vs actual comparison"""
    df, error = load_comparison_data()

    if df is None:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.text(0.5, 0.5, error, ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
        return fig

    # For now, just show prediction history (actuals to be added as they come in)
    fig, ax = plt.subplots(figsize=(14, 7))

    # Group by target_date and get latest prediction for each date
    latest_predictions = df.sort_values('prediction_date').groupby('target_date').last().reset_index()

    # Plot predicted prices
    ax.plot(latest_predictions['target_date'], latest_predictions['predicted_price'],
            label='Predicted', color='#E63946', linewidth=3,
            marker='o', markersize=8, alpha=0.9)

    # TODO: Add actual prices when available
    # This will be populated as days pass and actuals become available

    # Styling
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (SEK/kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Predicted vs Actual Electricity Prices (Growing Daily)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add info text
    info_text = f'Predictions tracked: {len(latest_predictions)} days\n(Actuals will be added as they become available)'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def get_forecast_table():
    """Get forecast data as table"""
    df, error = load_latest_forecast()

    if df is None:
        return pd.DataFrame({"Error": [error]})

    # Format for display
    display_df = df[['date', 'predicted_price']].copy()
    display_df.columns = ['Date', 'Predicted Price (SEK/kWh)']
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Predicted Price (SEK/kWh)'] = display_df['Predicted Price (SEK/kWh)'].round(4)

    return display_df


def get_last_update_time():
    """Get timestamp of last forecast update"""
    try:
        output_dir = Path("outputs")
        csv_files = list(output_dir.glob("forecast_*.csv"))

        if not csv_files:
            return "No forecasts available"

        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest_csv.stat().st_mtime)

        return f"Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}"
    except:
        return "Unknown"


def refresh_all():
    """Refresh all components"""
    return (
        create_forecast_plot(),
        create_comparison_plot(),
        get_forecast_table(),
        get_last_update_time()
    )


# Create Gradio Interface
with gr.Blocks(title="⚡ Electricity Price Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # ⚡ Electricity Price Predictor
    ### 7-Day Forecast for Stockholm (SE3) | XGBoost ML Model

    This system predicts Swedish electricity prices using weather data and machine learning.
    Forecasts are updated daily at midnight CET.
    """)

    # Last update timestamp
    update_time = gr.Markdown(get_last_update_time())

    # Refresh button
    refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")

    with gr.Tabs():
        with gr.Tab("📈 7-Day Forecast"):
            gr.Markdown("### Next 7 Days Predicted Prices")
            forecast_plot = gr.Plot(create_forecast_plot())

            gr.Markdown("### Forecast Data Table")
            forecast_table = gr.Dataframe(get_forecast_table())

        with gr.Tab("📊 Predicted vs Actual"):
            gr.Markdown("""
            ### Performance Tracking Over Time

            This chart grows daily as we collect actual prices and compare them with predictions.
            - **Red line**: Predicted prices (made 7 days in advance)
            - **Blue line**: Actual prices (added as they become available)
            """)
            comparison_plot = gr.Plot(create_comparison_plot())

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## How It Works

            ### Data Sources
            - **Weather**: Temperature, wind, solar radiation from OpenMeteo API
            - **Prices**: Historical electricity prices from elprisetjustnu.se
            - **Region**: SE3 (Stockholm, Sweden)

            ### Model
            - **Algorithm**: XGBoost Regression
            - **Features**: ~50 features including temporal patterns, weather data, and price lags
            - **Training**: Continuously updated with new data
            - **Horizon**: 7 days ahead predictions

            ### Automation
            - **Daily Data Collection**: 23:00 CET
            - **Daily Predictions**: 00:00 CET
            - **Weekly Retraining**: Sundays at 02:00 UTC

            ### Performance
            - **Typical RMSE**: ~0.12 SEK/kWh
            - **Typical MAE**: ~0.08 SEK/kWh
            - **R² Score**: ~0.93

            ---

            **GitHub**: [github.com/maxdougly/sml_project](https://github.com/maxdougly/sml_project)

            **Powered by**: Hopsworks Feature Store, XGBoost, Gradio
            """)

    # Refresh button handler
    refresh_btn.click(
        fn=refresh_all,
        inputs=[],
        outputs=[forecast_plot, comparison_plot, forecast_table, update_time]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
