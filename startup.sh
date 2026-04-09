#!/bin/bash
# Train model if it doesn't exist
if [ ! -f "/app/models/best_model.pkl" ]; then
    echo "Model not found. Training..."
    cd /app && python src/train.py
    echo "Training complete."
fi

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000