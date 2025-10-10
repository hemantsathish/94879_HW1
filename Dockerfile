FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY artifacts/ /app/artifacts/
COPY api_service.py /app/

EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]


