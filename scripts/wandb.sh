#!/bin/bash

SCRIPT_NAME="bcos_lm/wandb_init.py"
CONDA_ENV_NAME="agentic-eval"

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"
MODULE_NAME="bcos_lm.wandb_init"

# Navigate to the project directory
cd "$PROJECT_DIR" || {
    echo "Failed to change directory to $PROJECT_DIR"
    exit 1
}

echo "=========================================="
echo "Starting W&B Import"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" -m "$MODULE_NAME" \
   --checkpoint "bcos_gpt2_epoch/checkpoint-156250" \
   --project "bcos_gpt2"

echo "=========================================="
echo "W&B Import Completed"
echo "=========================================="