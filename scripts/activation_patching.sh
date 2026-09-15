#!/bin/bash

set -e

SCRIPT_NAME="activation_patching.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="activation_patching"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting Activation Patching"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --checkpoint "bcos_gpt2/b_1.25/checkpoint-156250" 

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --checkpoint "bcos_gpt2/b_1.5/checkpoint-156250" 

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --checkpoint "bcos_gpt2/b_2.0/checkpoint-156250" 

echo "=========================================="
echo "Activation Patching Completed"
echo "=========================================="
