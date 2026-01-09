"""Local storage mimicking Hopsworks Feature Store using Parquet files."""

import pandas as pd
import os
from datetime import datetime


class LocalFeatureGroup:
    """Mimics Hopsworks Feature Group using local Parquet files."""

    def __init__(self, name, version=1, base_path="data/processed"):
        self.name = name
        self.version = version
        self.base_path = base_path
        self.file_path = os.path.join(base_path, f"{name}_v{version}.parquet")

    def insert(self, df, overwrite=False, write_options=None):
        os.makedirs(self.base_path, exist_ok=True)

        if overwrite or not os.path.exists(self.file_path):
            df.to_parquet(self.file_path, index=False)
            print(f"Saved {len(df)} rows to {self.file_path}")
        else:
            # Append mode: read existing, concat, save
            existing_df = pd.read_parquet(self.file_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on primary key (date + city)
            if 'date' in combined_df.columns and 'city' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['date', 'city'], keep='last')
            combined_df.to_parquet(self.file_path, index=False)
            print(f"Appended {len(df)} rows to {self.file_path} (total: {len(combined_df)} rows)")

    def read(self):
        if not os.path.exists(self.file_path):
            print(f"No data file found at {self.file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(self.file_path)
        print(f"Loaded {len(df)} rows from {self.file_path}")
        return df

    def select_all(self):
        return self


class LocalFeatureView:
    """Mimics Hopsworks Feature View using local data."""

    def __init__(self, name, feature_group, labels=None):
        self.name = name
        self.feature_group = feature_group
        self.labels = labels or []

    def get_batch_data(self):
        return self.feature_group.read()

    def train_test_split(self, test_size=0.2, description=''):
        """Time-based train/test split."""
        df = self.get_batch_data()

        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

        # Sort by date for time-based split
        df = df.sort_values('date').reset_index(drop=True)

        # Split point
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in self.labels + ['date', 'city']]

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[self.labels[0]] if self.labels else pd.Series()
        y_test = test_df[self.labels[0]] if self.labels else pd.Series()

        # Add date and city back for reference
        X_train['date'] = train_df['date'].values
        X_train['city'] = train_df['city'].values
        X_test['date'] = test_df['date'].values
        X_test['city'] = test_df['city'].values

        print(f"Train/test split: {len(X_train)} train, {len(X_test)} test")

        return X_train, X_test, y_train, y_test


class LocalFeatureStore:
    """Mimics Hopsworks Feature Store using local file system."""

    def __init__(self, base_path="data/processed"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def get_or_create_feature_group(self, name, version=1, description='',
                                     primary_key=None, event_time=None):
        return LocalFeatureGroup(name, version, self.base_path)

    def get_feature_group(self, name, version=1):
        fg = LocalFeatureGroup(name, version, self.base_path)
        if not os.path.exists(fg.file_path):
            raise Exception(f"Feature group {name} v{version} not found")
        return fg

    def get_or_create_feature_view(self, name, version=1, query=None, labels=None):
        if query is None:
            raise ValueError("query (feature_group) is required")

        return LocalFeatureView(name, query, labels)


class LocalModelRegistry:
    """Mimics Hopsworks Model Registry using local file system."""

    def __init__(self, base_path="data/models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def python(self):
        return self

    def create_model(self, name, metrics=None, description=''):
        return LocalModel(name, self.base_path, metrics, description)

    def get_model(self, name, version=1):
        return LocalModel(name, self.base_path, version=version)


class LocalModel:
    """Mimics Hopsworks Model object."""

    def __init__(self, name, base_path, metrics=None, description='', version=1):
        self.name = name
        self.base_path = base_path
        self.metrics = metrics
        self.description = description
        self.version = version
        self.model_dir = os.path.join(base_path, f"{name}_v{version}")

    def save(self, local_path):
        import shutil

        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        shutil.copytree(local_path, self.model_dir)

        # Save metadata
        import json
        metadata = {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'metrics': self.metrics,
            'saved_at': datetime.now().isoformat()
        }

        with open(os.path.join(self.model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {self.model_dir}")

    def download(self):
        if not os.path.exists(self.model_dir):
            raise Exception(f"Model {self.name} v{self.version} not found")

        print(f"Model loaded from {self.model_dir}")
        return self.model_dir


class LocalProject:
    """Mimics Hopsworks Project object."""

    def __init__(self, name="local_project"):
        self.name = name
        self._fs = LocalFeatureStore()
        self._mr = LocalModelRegistry()

    def get_feature_store(self):
        return self._fs

    def get_model_registry(self):
        return self._mr


def get_local_project():
    """Get local project for testing without Hopsworks."""
    print("Using LOCAL storage (no Hopsworks)")
    return LocalProject("local_electricity_prediction")
