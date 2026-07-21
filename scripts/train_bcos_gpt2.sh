#!/usr/bin/env bash

python -m decoder_experiments.train_bcos_gpt2 \
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
    --bcos \