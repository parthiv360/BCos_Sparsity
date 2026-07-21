#!/bin/bash

SCRIPT_NAME="bcos_lm/train_bcos_gpt2.py"
CONDA_ENV_NAME="base"  # Use the base environment

PROJECT_DIR="/home/pasa00007/Hiwi/BCos_Sparsity/"
CONDA_PYTHON="/home/pasa00007/.conda/envs/agentic-eval/bin/python"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting Training"
echo "Script: $SCRIPT_NAME"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

"$CONDA_PYTHON" "$SCRIPT_NAME" \
    --model_name_or_path gpt2 \
    --dataset_name="webtext" \
    --output_dir "bcos_gpt2" \
    --batch_size=16 \
    --gradient_accumulation_steps=1 \
    --max_seq_length=512 \
    --learning_rate=5e-04 \
    --num_train_epochs=1 \
    --warmup_steps_or_ratio=0.01 \
    --num_train_examples=500000 \
    --num_eval_examples=10000 \
    --seed=42 \
    --b 1.1 \
    --bcos

echo "=========================================="
echo "Training Completed"
echo "=========================================="