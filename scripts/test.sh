#!/bin/bash

SCRIPT_NAME="setup_model.py"
CONDA_ENV_NAME="agentic-eval"

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"

# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$CONDA_ENV_NAME"

conda activate "/home/pasa00007/.conda/envs/agentic-eval"

cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting GPT-2 Model Setup"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

# Execute the setup script
python "$SCRIPT_NAME"

echo "=========================================="
echo "GPT-2 Model Setup Completed"
echo "=========================================="

# Deactivate the Conda environment
conda deactivate