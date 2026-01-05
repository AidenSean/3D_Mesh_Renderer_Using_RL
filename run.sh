#!/bin/bash
echo "Starting Pixel RL 3D Builder..."

# Ensure dependencies
# pip install -r requirements.txt || echo "Warning: Check install_log.txt"

# Download Model if needed
echo "Verifying GenAI Model state..."
python src/download_model.py

# Run Main
echo "Running Application..."
python main.py "$@"
