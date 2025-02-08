#!/bin/sh
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 \
              --port 5000 &
sleep 10
exec python main.py