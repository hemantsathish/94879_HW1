# Phase 1: Comprehensive Preprocessing Methodology with Business Justification

## Objective
Define a consistent, transparent preprocessing strategy for environmental sensor data. The goal is to transform raw UCI Air Quality measurements into a **trusted, analysis-ready dataset** while retaining auditability of the original stream.


## 1. Basic Principles
- **Preserve the raw feed** for traceability.  
- **Deliver a clean (silver) layer** optimized for analytics and modeling.  
- **Detect and flag data quality issues** early to support operational monitoring.  
- **Balance strict cleaning with business utility**: remove values that mislead decision-making, but retain imperfect data where it still provides signal.


## 2. Missing Value Handling

### Sentinel Values
- **Issue:** The UCI dataset encodes missing values as `-200`.  
- **Policy:** Convert all `-200` values (numeric or string) to `NULL`.  
- **Business Justification:** Treating `-200` as valid would drastically understate pollutant concentrations, misleading forecasts. Null conversion ensures downstream models interpret these readings correctly as “no data.”

### Nulls / NaNs
- **Policy:** Any `NaN` or empty values are also standardized to `NULL`.  
- **Business Justification:** Uniform handling avoids inconsistent downstream behavior and simplifies imputation strategies if needed.


## 3. Range Validation

- **Policy:** Any measurement outside the bounds is set to `NULL` and flagged in `qa_out_of_range_fields`.  
- **Business Justification:** Values outside plausible sensor ranges likely indicate malfunction, not reality. Nulling prevents skewed trend analysis and misinformed operational alerts.


## 4. Timestamp Normalization

### Accepted Inputs
- `Date + Time`
- `ts`
- `event_time`

### Standardization
- All timestamps parsed into **UTC ISO-8601** format.  
- If parsing fails → record is preserved in an **undated parquet file**.

**Business Justification:**
- UTC standardization ensures comparability across sensors/sites.  
- Preserving undated records (instead of discarding) provides full auditability and supports root-cause analysis of upstream formatting errors.

## 5. QA Metrics and Monitoring

### Per Record
- `qa_missing_fields`: count of required fields missing.  
- `qa_out_of_range_fields`: count of fields outside plausible ranges.  

### Per Batch
- **Missing Ratio** = (missing fields ÷ total fields).  
- Warning threshold: >20%.  

**Business Justification:**
- These lightweight metrics provide early warning of systemic sensor drift, ingestion failures, or calibration errors.  
- Proactive detection minimizes downstream business risks (e.g., inaccurate air quality alerts).


## 6. Placement in Architecture

- **Producer (Bronze Layer):** Publishes raw values, including `-200` sentinels.  
- **Consumer (Silver Layer):** Applies missing value handling, range checks, timestamp normalization, and parquet writes.  

**Business Justification:**
- Bronze preserves an immutable audit trail for compliance.  
- Silver ensures analysts and ML pipelines always work with standardized, trustworthy data.


## 7. Data Integrity and Storage

- **Daily parquet partitioning** by UTC date for efficient querying and model training.  
- **Undated parquet file** stores records with invalid timestamps.  
- **Structured logging** surfaces QA metrics, supporting monitoring dashboards and alerting.  

**Business Justification:**
- Partitioning improves query performance and supports incremental backfills.  
- Retaining undated data ensures no information is lost and supports forensic investigations.  
- Logging enables integration with monitoring systems, aligning with enterprise data quality SLAs.


## 8. Strategic Benefits

- **Improved Forecast Accuracy**: Eliminates spurious values that could bias predictive models.  
- **Operational Reliability**: Provides visibility into sensor health and ingestion quality.  
- **Regulatory Readiness**: Maintains both raw and cleaned data for compliance and audit.  
- **Business Trust**: Stakeholders can rely on outputs without manual cleansing, reducing decision latency.  