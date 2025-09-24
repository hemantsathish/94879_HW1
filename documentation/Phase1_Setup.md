# Phase 1: Streaming Infrastructure Setup and Architecture

## Objective
Deploy a robust Kafka streaming environment capable of ingesting, processing, and monitoring environmental sensor data with realistic fault tolerance and observability.

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


## Setup

### 1. Navigate to the infra folder
```bash
cd "..\phase_1_streaming_infrastructure"
```

---

### 2. Stop any old containers & wipe volumes
```bash
docker compose down -v
```

### 3. Start the stack
```bash
docker compose up -d
```

---

### 4. Check container status
```bash
docker compose ps
```

You should see **3 containers** with `STATUS = Up`:
- `phase_1_streaming_infrastructure-zookeeper-1`
- `phase_1_streaming_infrastructure-kafka-1`
- `phase_1_streaming_infrastructure-kafka-ui-1`


### 5. Create topics
Run these inside the **Kafka container** (DLQ is optional):
```bash
docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.raw  --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.clean --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.dlq   --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
```

Verify:
```bash
docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --list --bootstrap-server kafka:29092"
```

**Expected output:**
```
air_quality.raw
air_quality.clean
air_quality.dlq
```

---

### 6. Open Kafka UI
Go to: [http://localhost:8080](http://localhost:8080)  
You should see the **local cluster** and those **3 topics**.


### 7. Start your Python environment (In both windows created below)
From project root:
```bash
cd "..\kafka_air_quality_project"
.\.venv\Scripts\Activate
```

### 8. Create temporal train/test splits for modeling
Run the split script once to produce train and stream CSVs:
```bash
python ./phase_1_streaming_infrastructure/src/data_split.py --csv phase_1_streaming_infrastructure/data/raw/AirQualityUCI.csv --outdir phase_1_streaming_infrastructure/data/splits 
```

### 9. Run the Producer (Window A)
This replays the UCI Air Quality CSV into Kafka (`air_quality.raw`):
```bash
python .\phase_1_streaming_infrastructure\src\producer.py --csv ".\phase_1_streaming_infrastructure\data\splits\AirQualityUCI_stream.csv" --topic air_quality.raw --bootstrap 127.0.0.1:9092 --speedup 120
```

---

### 10. Run the Consumer (Window B)
This consumes from `air_quality.raw`, cleans data, forwards to `air_quality.clean`, and writes Parquet into silver:
```bash
python .\phase_1_streaming_infrastructure\src\consumer.py --in-topic air_quality.raw --out-topic air_quality.clean --bootstrap 127.0.0.1:9092 --out-dir ".\phase_1_streaming_infrastructure\data\silver" --batch-size 25
```

**Expected:**
- Logs like:
  ```
  Parquet write: 25 rows -> ...\silver\air_quality_clean_2004-03-10.parquet
  ```

- In **Kafka UI**, `air_quality.clean` topic fills with messages.  
- Files appear in `phase_1_streaming_infrastructure\data\silver`.  
- Metrics endpoint live at [http://localhost:8000/metrics](http://localhost:8000/metrics).


### 10. Stop
- Stop producer with **Ctrl+C** when you’re done streaming.  
- Stop consumer with **Ctrl+C** (it will flush the last batch).  


## Architectural Decisions

### Medallion Layering for File Differentiation
- **Bronze** (`air_quality.raw`): Holds unmodified sensor data with sentinel values intact.  
- **Silver** (`air_quality.clean`): Validated, standardized data written to parquet.  

### Kafka Configuration
- **Dual listeners:**
  - `127.0.0.1:9092` for host clients  
  - `kafka:29092` for inter-container traffic  
- **Replication & ISR** set to `1` since this is a single-node cluster.  
- **Auto-create topics disabled** to enforce explicit retention/partition design.  

### Producer Strategy
- `gzip` compression to reduce payload size.  
- `acks=all + retries=5` ensures durability despite single-node setup.  
- **Speedup replay** simulates real-time pacing of historical data.  

### Consumer Strategy
- **Manual offset commits** after parquet write: Ensures at-least-once delivery.  
- **Daily partitioned parquet**: Efficient time-series queries and replays.  
- **QA metrics** (`qa_missing_fields`, `qa_out_of_range_fields`, batch missing ratio) logged for monitoring.  

### Containerization with Docker
- **Docker Compose** was used to provision Zookeeper, Kafka broker, and Kafka-UI in isolated containers.  
- This avoided dependency conflicts with local environments and provided a **consistent, reproducible** setup across systems.
- Docker also simplifies startup, teardown, and resets while ensuring each service runs with the correct version and configuration, reducing operational overheads.


## 3. Troubleshooting Guide

This section lists **common issues** and quick fixes. When in doubt, check container logs:
```bash
docker compose logs -f kafka
docker compose logs -f zookeeper
docker compose logs -f kafka-ui
```

### Topics not created / missing
**Symptoms**
- `kafka-topics --list` does not show `air_quality.raw`, `air_quality.clean`, `air_quality.dlq`.
- Producer/consumer errors like `UNKNOWN_TOPIC_OR_PARTITION`.

**Causes & Fixes**
1) **Auto-create is disabled**  
   Create topics explicitly **inside the Kafka container** (note the internal bootstrap server `kafka:29092`):
   ```bash
   docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.raw  --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
   docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.clean --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
   docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --create --if-not-exists --topic air_quality.dlq   --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1"
   ```

2) **Using wrong bootstrap address**  
   - From **host**: use `127.0.0.1:9092`  
   - From **inside containers**: use `kafka:29092`

3) **Cluster not ready**  
   Wait a few seconds after `docker compose up -d`, then re-run the create/list commands:
   ```bash
   docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --list --bootstrap-server kafka:29092"
   ```


### Producer cannot connect / timeouts
**Symptoms**
- `Broker transport failure`, `Connection refused`, or `Timed out`.

**Checklist**
- **Advertised listeners match**:
  - `KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://127.0.0.1:9092,PLAINTEXT_INTERNAL://kafka:29092`
- **Port in use** (9092)? Check and stop any other local Kafka process.
- **Windows + WSL2 DNS issues**: try `127.0.0.1` instead of `localhost`.

**Sanity test (from host):**
```bash
kafka-topics --bootstrap-server 127.0.0.1:9092 --list
```

### Consumer runs but shows no data
**Causes & Fixes**

1) **Producer isn’t running or CSV path is wrong**  
   - Check Producer logs.
   - Ensure the CSV path is correct and quoted on Windows.

2) **Wrong topic**  
   - Verify the consumer is reading `air_quality.raw`.

**Quick peek at the topic:**
```bash
kafka-console-consumer --bootstrap-server 127.0.0.1:9092 --topic air_quality.raw --from-beginning --max-messages 5
```

### Kafka-UI not loading / empty cluster
**Checklist**
- Container is up:
  ```bash
  docker compose ps
  docker compose logs -f kafka-ui
  ```
- UI configured to `kafka:29092` (internal).  
- Browser cache: hard refresh or try another browser.

### Python environment issues
**Common errors & fixes**
- **`ModuleNotFoundError: confluent_kafka`**  
  Install dependencies in the **active venv**:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install -r requirements.txt
  ```
  On Windows, if build tools are missing, install the prebuilt wheel:
  ```bash
  pip install --only-binary=:all: confluent-kafka
  ```

- **Unicode/encoding errors reading CSV**  
  Set terminal to UTF-8 (Windows):
  ```bash
  chcp 65001
  ```
  Or add `encoding="utf-8"` to your `read_csv` if needed.

- **File paths on Windows**  
  Always quote paths with spaces:
  ```bash
  --csv ".\phase_1_streaming_infrastructure\data\raw\AirQualityUCI.csv"
  ```

---

### Data parsing issues (timestamps/decimals)
**Symptoms**
- Many records end up in `undated` parquet.
- High missing ratio warnings in logs.

**Fixes**
- The UCI file uses semicolon separator and comma decimals. Ensure:
  - Producer uses: `sep=";"`, `decimal=","`, and time normalization (`"18.00.00" : "18:00:00"`).
- Verify the date is **day-first** when parsing.
- Confirm time-zone handling (`UTC`) is applied.

---

### Messages not visible in `air_quality.clean`
**Causes**
- Consumer failed while cleaning, check consumer logs for errors.

**Actions**
- Watch consumer logs for batch messages and parquet writes.
- Check the UI topic view for `air_quality.clean`.


### Parquet write failures / permissions
**Symptoms**
- Errors writing to `data/silver`.

**Fixes**
- Ensure the output directory exists or let the consumer create it.
- Check file locks on Windows (close any viewer like Excel).
- Verify Python has write permissions to the project directory.

### Useful admin commands

- **Describe topic (partition count, leader, configs)**:
  ```bash
  docker compose exec -T kafka bash -lc "/usr/bin/kafka-topics --describe --topic air_quality.raw --bootstrap-server kafka:29092"
  ```

- **Consumer group lag**:
  ```bash
  docker compose exec -T kafka bash -lc "/usr/bin/kafka-consumer-groups --bootstrap-server kafka:29092 --describe --group air-quality-consumers"
  ```

- **Reset offsets to earliest**:
  ```bash
  docker compose exec -T kafka bash -lc "/usr/bin/kafka-consumer-groups --bootstrap-server kafka:29092 --group air-quality-consumers --topic air_quality.raw --reset-offsets --to-earliest --execute"
  ```

---

### DLQ (dead-letter queue) usage
If you route bad records to `air_quality.dlq`, verify with:
```bash
kafka-console-consumer --bootstrap-server 127.0.0.1:9092 --topic air_quality.dlq --from-beginning --max-messages 10
```
If the DLQ is empty while you expect errors, ensure your consumer **produces** to DLQ on failure paths.


### Debugging Tools
- **Kafka-UI**: [http://localhost:8080](http://localhost:8080) → inspect messages, offsets, partitions.  
- **Logs**: producer and consumer log to stdout with timestamps and levels.  
- **Parquet outputs**: check `./phase_1_streaming_infrastructure/data/silver` for cleaned daily files.  