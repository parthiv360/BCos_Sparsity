#!/bin/bash

SCRIPT_NAME="inspect_model.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting Inspect Code"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" "$SCRIPT_NAME"

echo "=========================================="
echo "Inspection Completed"
echo "=========================================="
