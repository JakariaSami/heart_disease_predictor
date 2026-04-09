#!/bin/bash
if [ ! -f "/app/models/best_model.pkl" ]; then
    echo "Model not found. Training..."
    cd /app && python src/train.py
    if [ $? -ne 0 ]; then
        echo "Training failed. Exiting."
        exit 1
    fi
    echo "Training complete."
fi

uvicorn api.main:app --host 0.0.0.0 --port 8000