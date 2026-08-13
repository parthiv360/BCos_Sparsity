#!/bin/bash

SCRIPT_NAME="bcos_lm/test_model.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="bcos_lm.test_model"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting Evaluation"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" -m "$MODULE_NAME" \
    --model_dir "bcos_gpt2_epoch" \
    --output_dir "bcos_gpt2_epoch_results" \

echo "=========================================="
echo "Evaluation Completed"
echo "=========================================="