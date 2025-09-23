# Phase 3: Predictive Analytics Model Development and Deployment

## Objective
Develop, validate, and deploy machine learning models for real-time air quality forecasting, demonstrating integration of streaming data with predictive analytics for operational decision-making.

# Requirements

- **Docker Desktop** (latest version)  
- **Python 3.10+** installed locally  
- **Virtual environment (recommended)** for Python dependencies  

### 1. Create and activate a virtual environment
From the project root:
```bash
python -m venv .venv
```

Activate it:
- On **Windows**:
  ```bash
  .\.venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Install required Python libraries
```bash
pip install -r requirements.txt
```
## Training
Train models (Naive baseline, Random Forest, SARIMA):
```
python phase_3/train.py \
  --csv phase_1_streaming_infrastructure/data/raw/AirQualityUCI.csv \
  --outdir artifacts \
  --trees 400
```
This will:
- Compute Naive baseline metrics.
- Train a Random Forest.
- Fit a SARIMA model.
- Save artifacts:
  -  `artifacts/model.pkl`: Random Forest
  - `artifacts/sarima.pkl`: SARIMA
  - `artifacts/features.json`: feature order + medians
  - `artifacts/metrics.json`: evaluation metrics

## Streaming Inference

### 1. Start Kafka (if not already running from Phase 1)
```bash
cd "..\phase_1_streaming_infrastructure"
docker compose up -d
```

### 2. Create topics (if not already created)

```bash
docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.clean --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.pred  --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
```

### 3. Run the inference service
```bash
python phase_3/infer_service.py \
  --bootstrap 127.0.0.1:9092 \
  --in-topic air_quality.clean \
  --out-topic air_quality.pred \
  --artifacts artifacts \
  --min_history 24 \
  --use_sarima
```


### You can track predictions on the Kafka UI