#!/bin/bash

SCRIPT_NAME="setup_model.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"

# Initialize Conda
source "/scratch/compuling/pasa00007/miniconda3/etc/profile.d/conda.sh"

# Activate the Conda base environment
conda activate "$CONDA_ENV_NAME"

# Navigate to the project directory
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