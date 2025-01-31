#!/bin/bash
set -x  # Enable debugging

echo "Starting pip install..."
pip install mlflow
pip list | grep mlflow

echo "Starting MLflow server..."
python -m mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

sleep 5

echo "Starting main application..."
python main.py

