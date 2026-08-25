#!/bin/bash

SCRIPT_NAME="bcos_lm/count_openwebtext.py"
CONDA_ENV_NAME="agentic-eval"

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="bcos_lm.count_openwebtext"

# Navigate to the project directory
cd "$PROJECT_DIR" || {
    echo "Failed to change directory to $PROJECT_DIR"
    exit 1
}

echo "=========================================="
echo "Starting Token Counting"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" -m "$MODULE_NAME" \
   --checkpoint "bcos_gpt2/checkpoint-156000" \
   --dataset_name "Skylion007/openwebtext"

echo "=========================================="
echo "Token Counting Completed"
echo "=========================================="