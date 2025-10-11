# Real-Time ML Model Monitoring with Kafka and Evidently

A production-grade machine learning pipeline for air quality prediction (CO levels) with real-time streaming inference, drift detection, and performance monitoring using Kafka, FastAPI, Evidently, Prometheus, and Grafana.

---

## 🎯 Overview

This project implements a real-time ML monitoring system that:

- **Streams** sensor data row-by-row through Kafka (each row = 1 hour of readings) to topic `air_quality.raw`
- **Predicts** CO(GT) levels using an XGBoost model loaded from **MLflow Model Registry** with 100+ engineered features
- **Publishes** predictions to Kafka topic `air_quality.pred`
- **Monitors** data drift, prediction drift, and model performance using Evidently (called from consumer)
- **Tracks** system metrics (latency, throughput, errors) via Prometheus and Grafana
- **Generates** daily (24-hour) and weekly (168-hour) drift reports automatically

The system simulates a production ML deployment where:
- A Kafka producer streams test data one row at a time to `air_quality.raw`
- A consumer buffers 168 rows for warmup (to build lag/rolling features)
- FastAPI serves predictions with feature engineering pipeline
- Consumer calls Evidently monitoring service to detect drift against training data baseline
- Predictions are published to `air_quality.pred` topic
- Prometheus/Grafana track operational metrics

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Test Dataset   │
│ (test_data_raw) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Kafka Producer  │─────▶│  Kafka Topic:    │
│ (producer.py)   │      │"air_quality.raw" │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Kafka Consumer  │
                         │ (consumer.py)   │
                         │  - 168-row      │
                         │    warmup       │
                         │  - Feature      │
                         │    engineering  │
                         │  - Calls        │
                         │    Evidently    │
                         └────────┬────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
       ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
       │   FastAPI      │ │  Evidently   │ │ Kafka Topic: │
       │  /predict      │ │  Monitoring  │ │"air_quality. │
       │                │ │   Service    │ │    pred"     │
       └────────┬───────┘ └──────┬───────┘ └──────────────┘
                │                 │
                ▼                 ▼
       ┌────────────────┐ ┌──────────────┐
       │ MLflow Model   │ │ HTML Reports │
       │   Registry     │ │(daily/weekly)│
       │  (XGBoost)     │ └──────────────┘
       └────────┬───────┘
                │
                ▼
       ┌────────────────┐
       │  Prometheus    │
       │    Metrics     │
       └────────┬───────┘
                │
                ▼
       ┌────────────────┐
       │    Grafana     │
       │   Dashboard    │
       └────────────────┘
```

**Key Components:**
- **Zookeeper**: Kafka coordination
- **Kafka**: Message broker for streaming
- **Kafka UI**: Web interface for Kafka monitoring (port 8080)
- **FastAPI**: ML inference API (port 8000)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Metrics visualization (port 3000)

---

## 📁 Project Structure

```
├── artifacts/                          # Model and configuration files (used in Docker)
│   ├── features.json                   # List of 100+ engineered features
│   ├── model.pkl                       # Best XGBoost model from MLflow runs
│   └── raw_columns.json                # Raw sensor columns for streaming
│
├── dataset/                            # Data files and creation scripts
│   ├── AirQualityUCI.csv              # Original raw dataset
│   ├── create_dataset.py              # Script to prepare streaming and reference data
│   ├── eval_data_engineered.csv       # Validation data with engineered features
│   ├── reference.csv                  # Reference dataset for Evidently (with predictions)
│   └── test_data_raw.csv              # Raw test data for Kafka streaming
│
├── reports/                            # Evidently monitoring reports
│   ├── daily/                         # 24-hour drift reports
│   ├── weekly/                        # 168-hour drift reports
│   │   ├── weekly_20050219_020000.html
│   │   ├── weekly_20050226_020000.html
│   │   └── ...
│   └── monitoring_summary.json        # Aggregated monitoring statistics
│
├── training/                           # MLflow experiment tracking
│   ├── run_data/                      # Data from all 19 MLflow runs
│   │   ├── metrics/                   # Performance metrics (RMSE, MAE, R², MAPE, etc.)
│   │   │   ├── test_rmse.csv
│   │   │   ├── test_r2.csv
│   │   │   ├── valid_mae.csv
│   │   │   └── ...
│   │   └── MLFlowRuns.csv             # Overview of all 19 experiment runs
│   ├── run_visualizations/            # Performance comparison plots
│   │   ├── test_rmse.png
│   │   ├── valid_mae.png
│   │   └── ...
│   └── model_training.ipynb           # Training notebook with Bayesian optimization
│
├── grafana/                            # Grafana configuration
│   └── datasources.yml                # Prometheus datasource config
│
├── api_service.py                      # FastAPI inference service
├── consumer.py                         # Kafka consumer with feature engineering
├── producer.py                         # Kafka producer for streaming test data
├── feature_buffer.py                   # Buffer for building lag/rolling features
├── monitoring_service.py               # Evidently drift monitoring service
├── generate_reference_dataset.py       # Creates reference dataset with predictions
├── docker-compose.yml                  # Multi-container Docker setup
├── Dockerfile                          # API service container definition
├── prometheus.yml                      # Prometheus scrape configuration
└── README.md                           # This file
```

### Key File Purposes

#### Artifacts (Used in Docker Container)
- **`features.json`**: Complete list of ~100+ engineered features (lag, rolling stats, interactions, time encodings)
- **`raw_columns.json`**: Raw sensor columns that should be streamed by producer

#### Dataset Files
- **`test_data_raw.csv`**: Raw sensor readings (no DateTime, no engineered features) - streamed by producer
- **`eval_data_engineered.csv`**: Validation data with all engineered features
- **`reference.csv`**: Generated from eval data with model predictions - baseline for Evidently drift detection

#### Training Files
- **MLflow tracked 19 experimental runs** testing different models:
  - RandomForest, ExtraTrees, GradientBoosting, **XGBoost** (winner), LightGBM, CatBoost
  - Ensemble methods (weighted averaging, stacking)
- **Metrics tracked**: RMSE, MAE, R², MAPE, SMAPE, MASE on test/validation sets
- **Visualizations**: Performance comparison charts across all runs

---

## ⚙️ Prerequisites

- **Python**: 3.11 or higher
- **Docker & Docker Compose**: Latest version
- **System Requirements**:
  - 8GB RAM minimum (for Kafka, Prometheus, Grafana)
  - 5GB disk space for Docker images and reports

---

## 🚀 Setup and Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `kafka-python` - Kafka producer/consumer
- `fastapi`, `uvicorn` - API service
- `xgboost`, `scikit-learn` - ML model
- `evidently` - Drift monitoring (v0.6.8+)
- `prometheus-client` - Metrics export
- `pandas`, `numpy` - Data processing

### Step 2: Create Datasets

Generate the test streaming data and reference dataset:

```bash
# Create test_data_raw.csv and eval_data_engineered.csv
python dataset/create_dataset.py
```

**Output:**
- `dataset/test_data_raw.csv` - Raw sensor data for streaming (no DateTime, no features)
- `dataset/eval_data_engineered.csv` - Validation data with engineered features

### Step 3: Generate Reference Dataset

Create the baseline dataset for Evidently drift detection:

```bash
python generate_reference_dataset.py
```

**What it does:**
- Loads `eval_data_engineered.csv` (validation data with features)
- Runs predictions using `artifacts/model.pkl`
- Saves `dataset/reference.csv` with features + target + predictions
- This becomes the "known good" baseline for comparison

### Step 4: Build and Start Docker Containers

Build the Docker images and start all services (Kafka, API, Prometheus, Grafana):

```bash
# Build the images
docker-compose build

# Start all services
docker-compose up -d
```

**Services started:**
- Zookeeper (port 2181)
- Kafka (port 9092)
- Kafka UI (port 8080) - `http://localhost:8080`
- FastAPI (port 8000) - `http://localhost:8000/docs`
- Prometheus (port 9090) - `http://localhost:9090`
- Grafana (port 3000) - `http://localhost:3000`

**Verify services:**
```bash
docker-compose ps
```

Wait for all services to show "healthy" status (~30-60 seconds).

---

## 🎬 Running the Pipeline

### Step 1: Start the Kafka Producer

Stream test data one row at a time to Kafka topic `air_quality.raw`:

```bash
python producer.py --csv dataset/test_data_raw.csv
```

**What happens:**
- Reads `test_data_raw.csv` row by row
- Sends each row (1 hour of sensor readings) to Kafka topic **`air_quality.raw`**
- Simulates real-time streaming with configurable delay
- Logs progress: "Sent row X/Y to air_quality.raw"

### Step 2: Start the Kafka Consumer

Process streaming data and generate predictions:

```bash
python consumer.py
```

**What happens:**
1. **Warmup Phase (first 168 rows)**:
   - Subscribes to Kafka topic **`air_quality.raw`**
   - Buffers incoming sensor readings
   - Builds lag features (1, 2, 3, 6, 12, 24, 48, 72 hours)
   - Builds rolling statistics (multiple windows)
   - Cannot make predictions yet (insufficient history)

2. **Inference Phase (after 168 rows)**:
   - Receives new row from Kafka topic `air_quality.raw`
   - Engineer 100+ features using buffer
   - Calls FastAPI `/predict` endpoint
   - Receives prediction from MLflow Model Registry model
   - **Calls Evidently monitoring service** to detect drift
   - Publishes prediction to Kafka topic **`air_quality.pred`**
   - Stores prediction + features for monitoring

3. **Report Generation**:
   - **Daily reports**: Every 24 predictions → `reports/daily/`
   - **Weekly reports**: Every 168 predictions → `reports/weekly/`

**Console output:**
```
[INFO] Waiting for warmup... (0/168 rows)
[INFO] Waiting for warmup... (50/168 rows)
[INFO] Warmup complete! Starting inference...
[INFO] Prediction 1: CO(GT) = 2.34
[INFO] Prediction 24: CO(GT) = 1.87 | Daily report generated
[INFO] Prediction 168: CO(GT) = 2.10 | Weekly report generated
```

### Step 3: System Runs Continuously

The producer and consumer will continue running:
- Producer streams all test data to `air_quality.raw`
- Consumer processes each row from `air_quality.raw`, makes predictions, calls Evidently for drift detection
- Predictions published to `air_quality.pred`
- Reports generated automatically at intervals
- Prometheus collects metrics every 15 seconds
- Grafana updates dashboards in real-time

**To stop:**
```bash
# Stop producer/consumer (Ctrl+C in their terminals)
# Stop Docker services
docker-compose down
```

---

## 📊 Monitoring and Observability

### Evidently Reports

**Daily Reports** (`reports/daily/`):
- Generated every 24 predictions (24 hours of data)
- Monitors short-term drift and performance
- Useful for detecting rapid changes

**Weekly Reports** (`reports/weekly/`):
- Generated every 168 predictions (7 days of data)
- Monitors long-term trends
- Format: `weekly_YYYYMMDD_HHMMSS.html`

**What's Monitored:**
1. **Data Drift**: Feature distribution changes vs. training data
   - Detects shifts in sensor readings (temperature, humidity, pollutants)
   - Uses statistical tests (Kolmogorov-Smirnov, Chi-squared)

2. **Prediction Drift**: Model output distribution changes
   - Monitors if predictions shift significantly
   - Early warning before performance degrades

3. **Model Performance** (with ground truth):
   - RMSE, MAE, R² on streaming predictions
   - Compares to baseline reference performance
   - Detects model degradation

**Open reports:**
- Navigate to `reports/weekly/` in file explorer
- Open any `.html` file in browser
- Interactive visualizations with drill-down capabilities

### Prometheus Metrics

Access Prometheus UI at `http://localhost:9090`

**Metrics Exported** (via `/metrics` endpoint):
- **`prediction_latency`** (Histogram): Time to generate prediction
- **`prediction_value`** (Gauge): Latest CO(GT) prediction
- **`missing_values_count`** (Gauge): Number of missing features
- **`inference_errors_total`** (Counter): Total prediction failures
- **`kafka_messages_consumed_total`** (Counter): Messages processed

### Grafana Dashboards

Access Grafana at `http://localhost:3000`
- **Username**: `admin`
- **Password**: `admin`

**Dashboard Panels** (5 visualizations):

1. **Prediction Latency** (Histogram)
   - Distribution of inference times
   - P50, P95, P99 percentiles
   - Detects performance degradation

2. **Throughput** (Time Series)
   - Predictions per second
   - Kafka message consumption rate
   - Monitors pipeline health

3. **Latest Prediction** (Gauge)
   - Current CO(GT) prediction value
   - Color-coded thresholds (green/yellow/red)

4. **Missing Values** (Time Series)
   - Count of missing features over time
   - Should remain at 0 for healthy pipeline
   - Spikes indicate data quality issues

5. **Error Rate** (Counter)
   - Total inference errors
   - Sudden increases indicate system problems

**Setting up the dashboard:**
1. Import datasource from `grafana/datasources.yml` (auto-provisioned)
2. Create dashboard with 5 panels using PromQL queries above
3. Save and monitor in real-time

---

## 🧠 Model Training Details

### Training Process

The model was trained using **MLflow** to track 19 experimental runs:

**Models Tested:**
- RandomForest
- ExtraTrees
- GradientBoosting
- **XGBoost** ✅ (selected)
- LightGBM
- CatBoost
- Ensemble methods (weighted, stacking)

**Hyperparameter Optimization:**
- **Bayesian optimization** via scikit-optimize
- **Early stopping** to prevent overfitting
- **Cross-validation** on validation set
- Target: Minimize RMSE, maximize R²

**Feature Engineering** (~100+ features):
- **Lag features**: 1, 2, 3, 6, 12, 24, 48, 72 hours
- **Rolling statistics**: Mean, std, min, max over multiple windows
- **Rate of change**: First differences, accelerations
- **Pollutant interactions**: Cross-products, ratios
- **Time encodings**: Cyclical hour, day, month (sin/cos)

**Best Model (XGBoost):**
- Selected based on test set performance
- **Loaded from MLflow Model Registry** in production (not from `model.pkl`)
- Used with `TransformedTargetRegressor` (log transformation)
- 19 experimental runs tracked with complete hyperparameter search

### MLflow Artifacts

All training artifacts stored in `training/`:

**`training/run_data/metrics/`**:
- `test_rmse.csv`, `test_mae.csv`, `test_r2.csv`
- `valid_rmse.csv`, `valid_mae.csv`, `valid_r2.csv`
- `test_mape.csv`, `test_smape.csv`, `test_mase.csv`
- Performance metrics for all 19 runs

**`training/run_data/MLFlowRuns.csv`**:
- Complete overview of all experiments
- Hyperparameters, metrics, timestamps
- Model selection comparison table

**`training/run_visualizations/`**:
- `test_rmse.png`, `test_mae.png`, `valid_r2.png`
- Bar charts comparing model performance
- Helps visualize which model performed best

**`training/model_training.ipynb`**:
- Complete training pipeline
- Data preprocessing, feature engineering
- Model training with Bayesian optimization
- Evaluation and model selection
- Model registration to MLflow Model Registry

---

## 🛠️ Troubleshooting

**Kafka connection errors:**
```bash
# Verify Kafka is running
docker-compose ps

# Check Kafka logs
docker-compose logs kafka
```

**API not responding:**
```bash
# Check API health
curl http://localhost:8000/health

# View API logs
docker-compose logs api
```

**Consumer not processing:**
- Ensure producer is running first and writing to `air_quality.raw`
- Check consumer is subscribed to correct topic (`air_quality.raw`)
- Verify 168-row warmup completed
- Check predictions are being published to `air_quality.pred`

**Reports not generating:**
- Check `reports/` directory permissions
- Ensure monitoring service has write access
- Verify prediction count reaches thresholds (24, 168)

**Grafana dashboard empty:**
- Verify Prometheus datasource connected
- Check API is exporting metrics: `curl http://localhost:8000/metrics`
- Ensure time range in Grafana includes recent data

---

## 📝 Notes

- **Warmup Requirement**: First 168 rows are for building lag/rolling features. Predictions start after warmup.
- **Kafka Topics**: Producer writes to `air_quality.raw`, consumer publishes predictions to `air_quality.pred`
- **Evidently Integration**: Monitoring service is called directly from the consumer after each prediction
- **Model Registry**: Production model is loaded from MLflow Model Registry, not from `model.pkl` artifact
- **Ground Truth**: This pipeline assumes access to actual CO(GT) values for performance monitoring (suitable for assignments/testing).
- **Production Deployment**: In real-world scenarios, ground truth may be delayed or unavailable. Adjust monitoring accordingly.
- **Resource Usage**: Kafka + Prometheus + Grafana can consume significant memory. Monitor Docker resource limits.

---

## 📧 Support

For issues or questions:
1. Check troubleshooting section above
2. Review Evidently documentation: https://docs.evidentlyai.com/
3. Review Kafka documentation: https://kafka.apache.org/documentation/

---

**Built with:** Python 3.11+ | Kafka | FastAPI | XGBoost | Evidently | Prometheus | Grafana | Docker
