#!/bin/bash
export PYTHONPATH=/app:$PYTHONPATH

# 1. Start FastAPI Inference Server in background
echo "🚀 Starting Plutchik Inference API on port 8000..."
python3 plutchik_erc_dashboard/inference_server.py &

# 2. Start Streamlit Dashboard in background
echo "🎭 Starting Plutchik ERC Dashboard on port 8501..."
streamlit run plutchik_erc_dashboard/app.py --server.port 8501 --server.address 0.0.0.0 &

# 3. Start Nginx in foreground to keep container alive and proxy everything
echo "🌐 Starting Nginx Proxy on port 7860..."
nginx -g "daemon off;"
