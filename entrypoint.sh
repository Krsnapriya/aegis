#!/bin/bash
export PYTHONPATH=/app:$PYTHONPATH

# 1. Start FastAPI Inference Server in background
echo "🚀 Starting Plutchik Inference API on port 8000..."
python3 inference_server.py &

# 2. Wait for the API to be healthy before starting Streamlit
echo "⏳ Waiting for Inference API to initialize (loading model)..."
MAX_RETRIES=30
COUNT=0
while [ $COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://127.0.0.1:8000/health | grep -q "healthy"; then
        echo "✅ Inference API is UP and Healthy!"
        break
    fi
    echo "..."
    sleep 2
    COUNT=$((COUNT+1))
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo "⚠️ Warning: Inference API failed to start in time. Starting Dashboard anyway..."
fi

# 3. Start Streamlit Dashboard in background
echo "🎭 Starting Plutchik ERC Dashboard on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# 4. Start Nginx in foreground to keep container alive and proxy everything
echo "🌐 Starting Nginx Proxy on port 7860..."
nginx -g "daemon off;"
