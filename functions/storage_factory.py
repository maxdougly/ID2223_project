"""
Unified interface for local Parquet storage and Hopsworks Feature Store.

Usage:
    storage = get_storage(mode='local')  # or 'production'
    fs = storage.get_feature_store()
    fg = fs.get_or_create_feature_group('electricity_price', version=1)
    fg.insert(df)
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal

StorageMode = Literal['local', 'production']


class StorageFactory:

    @staticmethod
    def get_storage(mode: StorageMode = 'local'):
        if mode == 'local':
            from functions.local_storage import get_local_project
            return get_local_project()
        elif mode == 'production':
            return HopsworksStorage()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'production'")


class HopsworksStorage:

    def __init__(self):
        try:
            import hopsworks
            self._hopsworks = hopsworks
            self._project = None
            self._fs = None
        except ImportError:
            raise ImportError(
                "Hopsworks not installed. Install with: pip install hopsworks\n"
                "Or use --mode local for local testing"
            )

    def get_feature_store(self):
        if self._project is None:
            api_key = os.getenv('HOPSWORKS_API_KEY')
            if not api_key:
                raise ValueError(
                    "HOPSWORKS_API_KEY not found in environment.\n"
                    "Set it with: export HOPSWORKS_API_KEY='your-key'\n"
                    "Or use --mode local for local testing"
                )

            print("Connecting to Hopsworks...")
            self._project = self._hopsworks.login(project="electricity_price")
            self._fs = self._project.get_feature_store()
            print(f"Connected to Hopsworks project: {self._project.name}")

        return self._fs

    def get_model_registry(self):
        if self._project is None:
            self.get_feature_store()
        return self._project.get_model_registry()


def get_storage(mode: StorageMode = 'local'):
    return StorageFactory.get_storage(mode)


def detect_mode() -> StorageMode:
    """Auto-detect mode: 'production' if HOPSWORKS_API_KEY is set, else 'local'"""
    if os.getenv('HOPSWORKS_API_KEY'):
        return 'production'
    return 'local'
