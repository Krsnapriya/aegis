#!/bin/bash

# Setup script for Plutchik ERC Dashboard

echo "🎭 Setting up Plutchik Emotion Recognition Dashboard..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create model directory
mkdir -p my_plutchik_model

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Train model: python train.py --train"
echo "3. Run Streamlit: streamlit run app.py"
echo ""
