import pandas as pd
import numpy as np
import json
from typing import Dict, Optional


class FeatureBuffer:
    """
    Maintains a rolling buffer of sensor readings to compute
    lag, rolling, and diff features matching the training dataset.
    """

    def __init__(self, maxlen, features_json_path):
        """
        Initialize the feature buffer.

        Args:
            maxlen: Maximum number of rows to keep in buffer
            features_json_path: Path to features.json defining feature list
        """
        self.maxlen = maxlen
        self.buffer = pd.DataFrame()

        # Load feature list from JSON
        with open(features_json_path, "r") as f:
            feature_config = json.load(f)
            self.feature_names = feature_config["features"]
            self.target_col = feature_config["target"]

        # Define pollutants for diff features
        self.pollutants = [
            "PT08.S1(CO)",
            "PT08.S2(NMHC)",
            "PT08.S3(NOx)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "NOx(GT)",
            "NO2(GT)",
        ]

        # Define base sensor columns needed in buffer
        self.base_sensors = [
            "PT08.S1(CO)",
            "NMHC(GT)",
            "C6H6(GT)",
            "PT08.S2(NMHC)",
            "NOx(GT)",
            "PT08.S3(NOx)",
            "NO2(GT)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "T",
            "RH",
            "AH",
        ]

    def push(self, record: Dict) -> None:
        """
        Add a new record to the buffer.

        Args:
            record: Dictionary containing timestamp, CO(GT), and sensor readings
        """
        # Create row with timestamp and all sensor values
        row_data = {
            "timestamp": record["timestamp"],
            self.target_col: record.get(self.target_col),
        }

        # Add all base sensor readings
        for col in self.base_sensors:
            row_data[col] = record.get(col)

        new_row = pd.DataFrame([row_data])
        self.buffer = pd.concat([self.buffer, new_row], ignore_index=True)

        # Keep only last maxlen rows
        if len(self.buffer) > self.maxlen:
            self.buffer = self.buffer.iloc[-self.maxlen :]

    def has_minimum_rows(self, min_rows: int = 168) -> bool:
        """
        Check if buffer has minimum rows required for feature computation.

        Args:
            min_rows: Minimum number of rows required

        Returns:
            True if buffer has sufficient rows
        """
        return len(self.buffer) >= min_rows

    def _compute_lag_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute lag features for CO(GT)."""
        features = {}
        lag_periods = [1, 2, 3, 6, 12, 24, 48, 72]

        for lag in lag_periods:
            if len(df) >= lag:
                val = df[self.target_col].iloc[-lag - 1]  # -1 because current is at -1
                features[f"{self.target_col}_lag_{lag}"] = (
                    float(val) if pd.notna(val) else None
                )
            else:
                features[f"{self.target_col}_lag_{lag}"] = None

        return features

    def _compute_rolling_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute rolling statistics for CO(GT)."""
        features = {}
        windows = [3, 6, 12, 24, 48, 168]

        for w in windows:
            if len(df) > w:  # Need more than w rows to exclude current
                # Exclude current row (last row) from rolling window
                window_data = df[self.target_col].iloc[-(w + 1) : -1]

                features[f"{self.target_col}_rolling_mean_{w}"] = (
                    float(window_data.mean()) if len(window_data) > 0 else None
                )
                features[f"{self.target_col}_rolling_std_{w}"] = (
                    float(window_data.std()) if len(window_data) > 0 else None
                )
                features[f"{self.target_col}_rolling_min_{w}"] = (
                    float(window_data.min()) if len(window_data) > 0 else None
                )
                features[f"{self.target_col}_rolling_max_{w}"] = (
                    float(window_data.max()) if len(window_data) > 0 else None
                )
            else:
                features[f"{self.target_col}_rolling_mean_{w}"] = None
                features[f"{self.target_col}_rolling_std_{w}"] = None
                features[f"{self.target_col}_rolling_min_{w}"] = None
                features[f"{self.target_col}_rolling_max_{w}"] = None

        return features

    def _compute_diff_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute rate of change (diff) features for pollutants."""
        features = {}
        diff_periods = [1, 3, 24]

        for col in self.pollutants:
            if col not in df.columns:
                continue

            for period in diff_periods:
                feature_name = f"{col}_diff_{period}h"

                if len(df) >= period + 1:
                    current = df[col].iloc[-1]
                    past = df[col].iloc[-(period + 1)]

                    if pd.notna(current) and pd.notna(past):
                        features[feature_name] = float(current - past)
                    else:
                        features[feature_name] = None
                else:
                    features[feature_name] = None

        return features

    def _compute_interaction_features(self, current_row: Dict) -> Dict[str, float]:
        """Compute pollutant interaction features (products and ratios)."""
        features = {}

        for i, p1 in enumerate(self.pollutants):
            val1 = current_row.get(p1)
            if val1 is None or pd.isna(val1):
                val1 = None

            for p2 in self.pollutants[i + 1 :]:
                val2 = current_row.get(p2)
                if val2 is None or pd.isna(val2):
                    val2 = None

                # Multiplication
                if val1 is not None and val2 is not None:
                    features[f"{p1}_x_{p2}"] = float(val1 * val2)
                else:
                    features[f"{p1}_x_{p2}"] = None

                # Ratio
                if val1 is not None and val2 is not None:
                    features[f"{p1}_ratio_{p2}"] = float(val1 / (val2 + 1e-8))
                else:
                    features[f"{p1}_ratio_{p2}"] = None

        return features

    def _compute_environmental_features(self, current_row: Dict) -> Dict[str, float]:
        """Compute environmental interaction features."""
        features = {}

        T = current_row.get("T")
        RH = current_row.get("RH")
        AH = current_row.get("AH")

        # temp_humidity
        if T is not None and AH is not None:
            features["temp_humidity"] = float(T * AH)
        else:
            features["temp_humidity"] = None

        # temp_sq
        if T is not None:
            features["temp_sq"] = float(T**2)
        else:
            features["temp_sq"] = None

        # humidity_sq
        if AH is not None:
            features["humidity_sq"] = float(AH**2)
        else:
            features["humidity_sq"] = None

        # temp_rh
        if T is not None and RH is not None:
            features["temp_rh"] = float(T * RH)
        else:
            features["temp_rh"] = None

        return features

    def _compute_time_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Compute time-based features from timestamp."""
        features = {}

        # Basic time features
        features["hour"] = int(timestamp.hour)
        features["day_of_week"] = int(timestamp.dayofweek)
        features["month"] = int(timestamp.month)
        features["day_of_month"] = int(timestamp.day)
        features["week_of_year"] = int(timestamp.isocalendar()[1])

        # Cyclical encodings
        features["hour_sin"] = float(np.sin(2 * np.pi * timestamp.hour / 24))
        features["hour_cos"] = float(np.cos(2 * np.pi * timestamp.hour / 24))
        features["dow_sin"] = float(np.sin(2 * np.pi * timestamp.dayofweek / 7))
        features["dow_cos"] = float(np.cos(2 * np.pi * timestamp.dayofweek / 7))
        features["month_sin"] = float(np.sin(2 * np.pi * (timestamp.month - 1) / 12))
        features["month_cos"] = float(np.cos(2 * np.pi * (timestamp.month - 1) / 12))

        # Categorical time features
        features["is_weekend"] = int(timestamp.dayofweek in [5, 6])
        features["is_rush_hour"] = int(timestamp.hour in [7, 8, 9, 17, 18, 19])
        features["is_night"] = int(timestamp.hour in [22, 23, 0, 1, 2, 3, 4, 5])
        features["is_winter"] = int(timestamp.month in [12, 1, 2])
        features["is_summer"] = int(timestamp.month in [6, 7, 8])

        return features

    def compute_features(self) -> Optional[Dict[str, float]]:
        """
        Compute all features for the most recent record in buffer.

        Returns:
            Dictionary of features matching features.json, or None if insufficient data
        """
        if not self.has_minimum_rows(min_rows=168):
            return None

        current_row = self.buffer.iloc[-1].to_dict()
        timestamp = pd.to_datetime(current_row["timestamp"])

        # Initialize feature dict with base sensors
        features = {}
        for col in self.base_sensors:
            val = current_row.get(col)
            features[col] = float(val) if val is not None and pd.notna(val) else None

        # Compute all feature types
        features.update(self._compute_lag_features(self.buffer))
        features.update(self._compute_rolling_features(self.buffer))
        features.update(self._compute_diff_features(self.buffer))
        features.update(self._compute_interaction_features(current_row))
        features.update(self._compute_environmental_features(current_row))
        features.update(self._compute_time_features(timestamp))

        # Ensure all features from features.json are present
        final_features = {}
        for feature_name in self.feature_names:
            final_features[feature_name] = features.get(feature_name, None)

        return final_features

    def get_buffer_length(self) -> int:
        """Get current buffer length."""
        return len(self.buffer)

    def get_current_record(self) -> Optional[Dict]:
        """Get the most recent record from buffer."""
        if len(self.buffer) == 0:
            return None
        return self.buffer.iloc[-1].to_dict()
