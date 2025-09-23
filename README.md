# 94879 Assignment 1: Real-Time Air Quality Streaming & Analysis
Repository for the first assignment in 94879: Fundamentals of Operationalizing AI

## Overview
This project demonstrates the design and deployment of a **Kafka-based streaming infrastructure** for real-time air quality monitoring. Using the UCI Air Quality dataset, it simulates live sensor data ingestion, performs preprocessing and quality checks, and outputs cleaned, analysis-ready data for downstream predictive modeling.

The work is divided into phases:
- **Phase 1:** Streaming infrastructure setup (Producer → Kafka → Consumer → Parquet).
- **Phase 2:** Exploratory data analysis (Temporal patterns, correlations, anomaly detection).
- **Phase 3:** Predictive modeling (Random Forest, SARIMA, Kafka streaming).

## Project Components
- **Dockerized Kafka stack**: Zookeeper, Kafka broker, Kafka-UI.
- **Producer** that replays historical UCI Air Quality data in real time.
- **Consumer** that validates, cleans, and writes data to Parquet.
- **Quality metrics** for missing and out-of-range values.
- **EDA Notebook** with temporal, correlation, decomposition, and anomaly analyses.
- **Predictive modeling and streaming** for Phase 3.

## Requirements
- **Docker Desktop**  
- **Python 3.10+**  
- Recommended: `venv` for dependency management  

## Setup & Usage

Detailed setup and usage instructions for each phase can be found in the documentation folder included in the repository.

- **Phase 1:** Streaming infrastructure and pipeline setup
- **Phase 2:** Exploratory data analysis and statistical insights (Python notebook)
- **Phase 3:** Predictive modeling and evaluation

Please refer to the respective documentation files for step-by-step guidance.



