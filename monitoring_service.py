import pandas as pd
import numpy as np
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json
import traceback
import logging
from evidently import Dataset, DataDefinition, Regression
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Monitoring service for streaming ML predictions.

    Assumes consumer has completed 168-row warmup before first call.
    Generates reports automatically:
    - Daily: every 24 predictions
    - Weekly: every 168 predictions
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "CO(GT)",
        prediction_column: str = "prediction",
        reports_dir: str = "./reports",
        max_buffer_size: int = 1000,
    ):
        """
        Initialize monitoring service.

        Args:
            reference_data: Validation dataset with features + target + predictions
            feature_columns: List of feature column names used in model
            target_column: Name of target column
            prediction_column: Name for prediction column
            reports_dir: Directory to store generated reports
            max_buffer_size: Maximum number of predictions to keep in buffer
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.reports_dir = Path(reports_dir)

        # Identify categorical features (time-based binary/discrete features)
        categorical_features = [
            "is_weekend",
            "is_rush_hour",
            "is_night",
            "is_winter",
            "is_summer",
            "hour",
            "day_of_week",
            "month",
            "day_of_month",
            "week_of_year",
        ]

        # Only use continuous features for numerical drift detection
        numerical_features = [
            f for f in feature_columns if f not in categorical_features
        ]

        # Create Evidently Dataset for reference data with regression mapping
        self.data_definition = DataDefinition(
            regression=[Regression(target=target_column, prediction=prediction_column)],
            numerical_columns=numerical_features,
            categorical_columns=categorical_features,
        )

        self.reference_dataset = Dataset.from_pandas(
            reference_data, data_definition=self.data_definition
        )

        # Create reports directory structure
        self._create_reports_dir()

        # Monitoring buffer: stores predictions for drift detection
        self.monitoring_buffer = deque(maxlen=max_buffer_size)

        # Statistics tracking
        self.stats = {
            "total_predictions": 0,
            "drift_detected_count": {"daily": 0, "weekly": 0},
            "reports_generated": {"daily": 0, "weekly": 0},
        }

        logger.info("MonitoringService initialized")
        logger.info(f"  Reference data: {len(reference_data)} rows")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Reports directory: {self.reports_dir}")

    def add_prediction(
        self,
        features: Dict,
        prediction: float,
        actual: float,
        timestamp: datetime,
    ) -> Dict:
        """
        Add a new prediction to monitoring system.
        Automatically generates reports at appropriate intervals.

        Args:
            features: Dictionary of engineered feature values
            prediction: Model prediction value
            actual: Ground truth value
            timestamp: Prediction timestamp

        Returns:
            Dictionary with status and generated report paths
        """
        # Validate features - check for None values
        has_none = False
        none_features = []

        for feature_name in self.feature_columns:
            feature_value = features.get(feature_name)
            if feature_value is None or (
                isinstance(feature_value, float) and np.isnan(feature_value)
            ):
                has_none = True
                none_features.append(feature_name)

        if has_none:
            logger.warning(
                f"Skipped prediction due to None values in features: {none_features[:5]}..."
            )
            return {
                "status": "skipped",
                "reason": "none_values_in_features",
                "total_predictions": self.stats["total_predictions"],
            }

        # Build record
        record = {
            **features,
            self.prediction_column: prediction,
            self.target_column: actual,
        }

        # Add to monitoring buffer
        self.monitoring_buffer.append(record)
        self.stats["total_predictions"] += 1

        self._check_and_generate_reports(timestamp)

        return {
            "status": "success",
            "total_predictions": self.stats["total_predictions"],
        }

    def _create_reports_dir(self):
        """Create reports directory structure if not exists."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "daily").mkdir(exist_ok=True)
        (self.reports_dir / "weekly").mkdir(exist_ok=True)

    def _check_and_generate_reports(self, timestamp: datetime):
        """
        Check if conditions are met for report generation.

        Returns:
            List of generated report file paths
        """
        n_predictions = self.stats["total_predictions"]

        # Daily report: every 24 predictions
        if n_predictions % 24 == 0 and n_predictions >= 24:
            self._generate_daily_report(timestamp)

        # Weekly report: every 168 predictions
        if n_predictions % 168 == 0 and n_predictions >= 168:
            self._generate_weekly_report(timestamp)

    def _generate_daily_report(self, timestamp: datetime):
        """
        Generate daily report with drift and quality metrics.
        Uses last 24 predictions.
        """
        try:
            current_df = self._get_recent_data(window_size=24)

            if len(current_df) < 24:
                logger.warning(
                    f"Insufficient data for daily report: {len(current_df)}/24 rows"
                )
                return None

            # Create Evidently Dataset for current data
            current_dataset = Dataset.from_pandas(
                current_df, data_definition=self.data_definition
            )

            # Check if ground truth available
            has_target = self.target_column in current_df.columns

            # Build report with appropriate presets
            if has_target:
                report = Report(
                    [DataDriftPreset(), DataSummaryPreset(), RegressionPreset()]
                )
            else:
                report = Report([DataDriftPreset(), DataSummaryPreset()])

            # Run report
            result = report.run(
                reference_data=self.reference_dataset, current_data=current_dataset
            )

            # Save HTML
            report_path = (
                self.reports_dir
                / "daily"
                / f"daily_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
            )
            result.save_html(str(report_path))

            self.stats["reports_generated"]["daily"] += 1

            # Check for drift
            if self._check_drift(result):
                self.stats["drift_detected_count"]["daily"] += 1
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")

            traceback.print_exc()
            return None

    def _generate_weekly_report(self, timestamp: datetime):
        """
        Generate weekly performance report.
        Uses last 168 predictions. Requires ground truth.
        """
        try:
            current_df = self._get_recent_data(window_size=168)

            if len(current_df) < 168:
                logger.warning(
                    f"Insufficient data for weekly report: {len(current_df)}/168 rows"
                )
                return None

            # Weekly report requires ground truth for performance metrics
            has_target = self.target_column in current_df.columns

            if not has_target:
                logger.warning("Weekly report skipped: requires ground truth values")
                return None

            # Create Evidently Dataset for current data
            current_dataset = Dataset.from_pandas(
                current_df, data_definition=self.data_definition
            )

            # Comprehensive performance analysis
            report = Report([RegressionPreset(), DataDriftPreset()])

            # Run report
            result = report.run(
                reference_data=self.reference_dataset, current_data=current_dataset
            )

            # Save HTML
            report_path = (
                self.reports_dir
                / "weekly"
                / f"weekly_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
            )
            result.save_html(str(report_path))

            self.stats["reports_generated"]["weekly"] += 1

            # Check for drift
            if self._check_drift(result):
                self.stats["drift_detected_count"]["weekly"] += 1

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            traceback.print_exc()
            return None

    def _get_recent_data(self, window_size: int) -> pd.DataFrame:
        """
        Extract recent predictions from buffer as DataFrame.
        Ensures all data is numeric and properly formatted for Evidently.

        Args:
            window_size: Number of recent predictions to retrieve

        Returns:
            DataFrame with features, predictions, and targets
        """
        if len(self.monitoring_buffer) < window_size:
            recent_records = list(self.monitoring_buffer)
        else:
            recent_records = list(self.monitoring_buffer)[-window_size:]

        if not recent_records:
            return pd.DataFrame()

        df = pd.DataFrame(recent_records)

        # Ensure required columns present
        required_cols = self.feature_columns + [self.prediction_column]
        if self.target_column in df.columns:
            required_cols.append(self.target_column)

        # Keep only model-relevant columns
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]

        # Convert all feature columns to float explicitly
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert prediction and target to float
        if self.prediction_column in df.columns:
            df[self.prediction_column] = pd.to_numeric(
                df[self.prediction_column], errors="coerce"
            )

        if self.target_column in df.columns:
            df[self.target_column] = pd.to_numeric(
                df[self.target_column], errors="coerce"
            )

        # Drop any rows that still have NaN after conversion
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        if dropped > 0:
            logger.warning(
                f"Dropped {dropped} rows with NaN values after type conversion"
            )

        return df

    def _check_drift(self, result) -> bool:
        """
        Extract drift detection result from Evidently Snapshot.

        Args:
            result: Snapshot object returned from report.run()

        Returns:
            True if dataset-level drift detected
        """
        try:
            # Get metric results dictionary from Snapshot
            metric_results = result.metric_results

            # Iterate through all metric results
            for metric_id, metric_result in metric_results.items():
                # metric_result is a MetricResult object with a 'result' attribute
                if hasattr(metric_result, "result"):
                    result_data = metric_result.result

                    # Check if this result has dataset drift information
                    if hasattr(result_data, "dataset_drift"):
                        drift_detected = result_data.dataset_drift

                        if drift_detected:
                            n_drifted = getattr(
                                result_data, "number_of_drifted_columns", 0
                            )
                            logger.warning(
                                f"DRIFT DETECTED: {n_drifted} features drifted"
                            )
                            return True

            return False

        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Get current monitoring statistics.

        Returns:
            Dictionary with prediction counts, drift alerts, and report counts
        """
        return {
            "total_predictions": self.stats["total_predictions"],
            "buffer_size": len(self.monitoring_buffer),
            "drift_detected_count": self.stats["drift_detected_count"],
            "reports_generated": self.stats["reports_generated"].copy(),
        }

    def export_summary(self) -> str:
        """
        Export monitoring summary to JSON file.

        Returns:
            Path to saved summary file
        """
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": self.get_statistics(),
        }

        summary_path = self.reports_dir / "monitoring_summary.json"

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary exported: {summary_path}")
        return str(summary_path)
