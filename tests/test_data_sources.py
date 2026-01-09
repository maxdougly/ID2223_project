"""
Test script to verify all data sources are working.
Run this before executing the main pipelines.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from datetime import datetime, timedelta
from functions.util import get_historical_weather, get_electricity_prices


def test_openmeteo():
    print("\n" + "="*60)
    print("Testing OpenMeteo API (Weather Data)")
    print("="*60)

    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

        print(f"Fetching weather data from {start_date} to {end_date}...")
        df = get_historical_weather(start_date, end_date, 59.33, 18.07)

        if df.empty:
            print("FAILED: No data returned from OpenMeteo API")
            return False

        print(f"SUCCESS: Retrieved {len(df)} days of weather data")
        print(f"\nSample data:")
        print(df.head(3))
        return True

    except Exception as e:
        print(f"FAILED: {str(e)}")
        return False


def test_electricity_api():
    print("\n" + "="*60)
    print("Testing elprisetjustnu.se API (Electricity Prices)")
    print("="*60)

    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

        print(f"Fetching electricity prices from {start_date} to {end_date}...")
        df = get_electricity_prices(start_date, end_date, region='SE3')

        if df.empty:
            print("FAILED: No data returned from electricity API")
            return False

        print(f"SUCCESS: Retrieved {len(df)} days of electricity price data")
        print(f"\nSample data:")
        print(df.head(3))

        print(f"\nPrice statistics:")
        print(f"  Mean: {df['price_sek_kwh_mean'].mean():.3f} SEK/kWh")
        print(f"  Min:  {df['price_sek_kwh_mean'].min():.3f} SEK/kWh")
        print(f"  Max:  {df['price_sek_kwh_mean'].max():.3f} SEK/kWh")
        return True

    except Exception as e:
        print(f"FAILED: {str(e)}")
        return False


def test_storage():
    print("\n" + "="*60)
    print("Testing Storage Abstraction")
    print("="*60)

    try:
        from functions.storage_factory import get_storage

        print("Testing local storage...")
        storage_local = get_storage(mode='local')
        fs = storage_local.get_feature_store()
        print("SUCCESS: Local storage working")

        import os
        if os.getenv('HOPSWORKS_API_KEY'):
            print("\nTesting Hopsworks storage...")
            storage_prod = get_storage(mode='production')
            fs_prod = storage_prod.get_feature_store()
            print("SUCCESS: Hopsworks storage working")
        else:
            print("\nSKIPPED: Hopsworks test (no API key set)")
            print("   Set HOPSWORKS_API_KEY to test production mode")

        return True

    except Exception as e:
        print(f"FAILED: {str(e)}")
        return False


def main():
    print("\n" + "="*60)
    print("ELECTRICITY PRICE PREDICTION - DATA SOURCE TESTS")
    print("="*60)

    results = {
        'OpenMeteo (Weather)': test_openmeteo(),
        'elprisetjustnu.se (Prices)': test_electricity_api(),
        'Storage Layer': test_storage()
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        padding = "." * (40 - len(name))
        print(f"{name}{padding} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\nAll tests passed! You're ready to run the pipelines.")
    else:
        print("\nSome tests failed. Fix the issues above before running pipelines.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
