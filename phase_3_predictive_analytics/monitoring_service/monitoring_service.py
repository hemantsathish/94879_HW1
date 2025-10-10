import pandas as pd
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import json
import pickle

from evidently import Dataset, DataDefinition, Regression
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset


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
        buffer_backup_path: str = "./monitoring_buffer.pkl",
        max_buffer_size: int = 1000,
    ):
        """
        Initialize monitoring service.

        Args:
            reference_data: Test dataset from training (with features + target)
            feature_columns: List of feature column names used in model
            target_column: Name of target column
            prediction_column: Name for prediction column
            reports_dir: Directory to store generated reports
            buffer_backup_path: Path to persist buffer for crash recovery
            max_buffer_size: Maximum number of predictions to keep in buffer
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.reports_dir = Path(reports_dir)
        self.buffer_backup_path = buffer_backup_path

        # Create Evidently Dataset for reference data with regression mapping
        self.data_definition = DataDefinition(
            regression=[Regression(target=target_column, prediction=prediction_column)],
            numerical_columns=feature_columns,
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

        print("MonitoringService initialized")
        print(f"  Reference data: {len(reference_data)} rows")
        print(f"  Features: {len(feature_columns)}")
        print(f"  Reports directory: {self.reports_dir}")

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
        # Build record
        record = {
            **features,
            self.prediction_column: prediction,
            "timestamp": timestamp,
            self.target_column: actual,
        }

        # Add to monitoring buffer
        self.monitoring_buffer.append(record)
        self.stats["total_predictions"] += 1

        # Check and generate reports
        report_paths = self._check_and_generate_reports()

        # Backup buffer periodically (every 24 predictions)
        if self.stats["total_predictions"] % 24 == 0:
            self._save_buffer()

        return {
            "status": "success",
            "total_predictions": self.stats["total_predictions"],
            "report_paths": report_paths,
        }

    def _create_reports_dir(self):
        """Create reports directory structure if not exists."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "daily").mkdir(exist_ok=True)
        (self.reports_dir / "weekly").mkdir(exist_ok=True)

    def _check_and_generate_reports(self) -> List[str]:
        """
        Check if conditions are met for report generation.

        Returns:
            List of generated report file paths
        """
        report_paths = []
        n_predictions = self.stats["total_predictions"]

        # Daily report: every 24 predictions
        if n_predictions % 24 == 0 and n_predictions >= 24:
            path = self._generate_daily_report()
            if path:
                report_paths.append(path)
                print(f"Daily report generated: {path}")

        # Weekly report: every 168 predictions
        if n_predictions % 168 == 0 and n_predictions >= 168:
            path = self._generate_weekly_report()
            if path:
                report_paths.append(path)
                print(f"Weekly report generated: {path}")

        return report_paths

    def _generate_daily_report(self) -> Optional[str]:
        """
        Generate daily report with drift and quality metrics.
        Uses last 24 predictions.
        """
        try:
            current_df = self._get_recent_data(window_size=24)

            if len(current_df) < 24:
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

            # Run report and get result object
            result = report.run(
                reference_data=self.reference_dataset, current_data=current_dataset
            )

            # Save HTML using result object
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / "daily" / f"daily_{timestamp}.html"
            result.save_html(str(report_path))

            self.stats["reports_generated"]["daily"] += 1

            # Check for drift using result object
            if self._check_drift(result):
                self.stats["drift_detected_count"]["daily"] += 1

            return str(report_path)

        except Exception as e:
            print(f"Error generating daily report: {e}")
            return None

    def _generate_weekly_report(self) -> Optional[str]:
        """
        Generate weekly performance report.
        Uses last 168 predictions. Requires ground truth.
        """
        try:
            current_df = self._get_recent_data(window_size=168)

            if len(current_df) < 168:
                return None

            # Weekly report requires ground truth for performance metrics
            has_target = self.target_column in current_df.columns

            if not has_target:
                print("⚠ Weekly report skipped: requires ground truth values")
                return None

            # Create Evidently Dataset for current data
            current_dataset = Dataset.from_pandas(
                current_df, data_definition=self.data_definition
            )

            # Comprehensive performance analysis
            report = Report([RegressionPreset(), DataDriftPreset()])

            # Run report and get result object
            result = report.run(
                reference_data=self.reference_dataset, current_data=current_dataset
            )

            # Save HTML using result object
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / "weekly" / f"weekly_{timestamp}.html"
            result.save_html(str(report_path))

            self.stats["reports_generated"]["weekly"] += 1

            if self._check_drift(result):
                self.stats["drift_detected_count"]["weekly"] += 1

            return str(report_path)

        except Exception as e:
            print(f"Error generating weekly report: {e}")
            return None

    def _get_recent_data(self, window_size: int) -> pd.DataFrame:
        """
        Extract recent predictions from buffer as DataFrame.

        Args:
            window_size: Number of recent predictions to retrieve

        Returns:
            DataFrame with features, predictions, and targets (if available)
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
        return df[available_cols]

    def _check_drift(self, result) -> bool:
        """
        Extract drift detection result from Evidently report result.

        Args:
            result: Report result object returned from report.run()

        Returns:
            True if dataset-level drift detected
        """
        try:
            result_dict = result.as_dict()

            # Navigate through metrics to find drift results
            metrics = result_dict.get("metrics", [])

            for metric in metrics:
                metric_name = metric.get("metric", "")

                # Check for dataset drift in new API structure
                if "drift" in metric_name.lower():
                    result_data = metric.get("result", {})

                    # Try different possible keys for drift detection
                    drift_detected = result_data.get(
                        "dataset_drift", False
                    ) or result_data.get("drift_detected", False)

                    if drift_detected:
                        n_drifted = result_data.get("number_of_drifted_columns", 0)
                        print(f"⚠ DRIFT DETECTED: {n_drifted} features drifted")
                        return True

            return False

        except Exception as e:
            print(f"Error checking drift: {e}")
            return False

    def _save_buffer(self):
        """Persist buffer state to disk for crash recovery."""
        try:
            buffer_state = {
                "monitoring_buffer": list(self.monitoring_buffer),
                "stats": self.stats,
            }

            with open(self.buffer_backup_path, "wb") as f:
                pickle.dump(buffer_state, f)

        except Exception as e:
            print(f"Warning: Buffer save failed: {e}")

    def load_buffer(self) -> bool:
        """
        Load buffer from disk after crash/restart.

        Returns:
            True if buffer loaded successfully
        """
        try:
            if Path(self.buffer_backup_path).exists():
                with open(self.buffer_backup_path, "rb") as f:
                    buffer_state = pickle.load(f)

                self.monitoring_buffer = deque(
                    buffer_state["monitoring_buffer"],
                    maxlen=self.monitoring_buffer.maxlen,
                )
                self.stats = buffer_state["stats"]

                print(f"Buffer restored: {len(self.monitoring_buffer)} records")
                return True

        except Exception as e:
            print(f"Warning: Buffer load failed: {e}")

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

        print(f"Summary exported: {summary_path}")
        return str(summary_path)


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    """
    Example: Initialize and use MonitoringService with Kafka consumer.
    """

    # Load reference data (test set from training)
    reference_data = pd.read_csv("test_data.csv")

    # Define feature columns (from training script)
    feature_columns = [
        col
        for col in reference_data.columns
        if col not in ["CO(GT)", "DateTime", "timestamp"]
    ]

    # Initialize monitoring service (once at consumer startup)
    monitor = MonitoringService(
        reference_data=reference_data,
        feature_columns=feature_columns,
        target_column="CO(GT)",
        prediction_column="prediction",
        reports_dir="./monitoring_reports",
    )

    # Optional: load previous buffer state
    monitor.load_buffer()

    # Simulate consumer processing stream
    # Consumer has already done 168-row warmup and feature engineering
    for idx, row in reference_data.iterrows():
        # Features already engineered by consumer
        features = row[feature_columns].to_dict()
        actual = row["CO(GT)"]

        # Model prediction (from inference service)
        prediction = 2.5  # Replace with actual model.predict(features)

        # Add to monitoring
        result = monitor.add_prediction(
            features=features,
            prediction=prediction,
            actual=actual,
            timestamp=row["DateTime"],
        )

        # Log report generation
        if result["report_paths"]:
            for path in result["report_paths"]:
                print(f"  📊 Report: {path}")

    # Export final summary
    monitor.export_summary()

    # Get statistics
    stats = monitor.get_statistics()
    print("\nMonitoring Complete:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Drift alerts: {stats['drift_detected_count']}")
    print(f"  Daily reports: {stats['reports_generated']['daily']}")
    print(f"  Weekly reports: {stats['reports_generated']['weekly']}")
