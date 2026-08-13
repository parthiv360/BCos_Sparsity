import torch
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np

DATA_DIR = "data"
        
ioi_datasets = ["ioi_dataset"]

datasets = { "ioi": ioi_datasets}

# load dataset
def load_dataset(dataset_type, dataset_name):
    dataset = {}
    data_path = f"{DATA_DIR}/{dataset_type}_with_targets/{dataset_name}.csv"
    # load the pandas dataframe
    df = pd.read_csv(data_path)
    one_sentence_prefixes = df['one_prefix_prefix'].tolist()
    one_word_targets = df['one_prefix_word_good'].tolist()
    one_word_foils = df['one_prefix_word_bad'].tolist()
    if 'target_phrase' in df.columns:
        evidences = df['target_phrase'].tolist()
    else:
        evidences = df['target'].tolist()
    indexes = list(range(len(one_sentence_prefixes)))
    dataset['prefix'] = one_sentence_prefixes
    dataset['target'] = one_word_targets
    dataset['foil'] = one_word_foils
    dataset['evidence'] = evidences
    dataset['index'] = indexes
    return dataset

#evaluate one checkpoint
def evaluate_checkpoint(model_dir, dataset_type, dataset_name, device):
    config = AutoConfig.from_pretrained(model_dir)
    config._attn_implementation = "eager"
    if "gpt" in model_dir:
        model = GPT2LMHeadModel.load_from_pretrained(model_dir, config=config)
    else:
        raise ValueError(f"Model {model_dir} not supported.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    dataset = load_dataset(dataset_type, dataset_name)
    print(f"Loaded {dataset_name} dataset with " f"{len(dataset['prefix'])} examples.")

    correct = 0
    total = 0
    prob_diff_results = []
    target_probs = []
    foil_probs = []
    target_logits = []
    foil_logits = []

    for i in tqdm(dataset['index']):
        prefix = dataset['prefix'][i]
        target = ' ' + dataset['target'][i]
        foil = ' ' + dataset['foil'][i]
        inputs = tokenizer(prefix, return_tensors="pt").to(device)
        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)['input_ids'][0, 0]
        foil_ids = tokenizer(foil, return_tensors="pt", add_special_tokens=False)['input_ids'][0, 0]
        if target_ids == foil_ids:
            continue
        # Get the logits for the target and foil
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            next_token_logits = logits[0,-1]
            target_logit = next_token_logits[target_ids].item()
            foil_logit = next_token_logits[foil_ids].item()
            probabilities = torch.softmax(logits, dim=-1)
            target_prob = probabilities[0, -1, target_ids].item()
            foil_prob = probabilities[0, -1, foil_ids].item()

            if target_logit> foil_logit:
                correct += 1
            total += 1
            prob_diff = target_prob - foil_prob
        prob_diff_results.append(prob_diff)
        target_probs.append(target_prob)
        foil_probs.append(foil_prob)
        target_logits.append(target_logit)
        foil_logits.append(foil_logit)
    
    if total == 0:
        raise RuntimeError("No valid examples found in the dataset.")
                
    accuracy = correct / total
    results = {
        "checkpoint": model_dir,
        "accuracy": float(accuracy),
        "total": int(total),
        "correct": int(correct),
        "mean_prob_diff": float(np.mean(prob_diff_results)),
        "mean_target_probs": float(np.mean(target_probs)),
        "mean_foil_probs": float(np.mean(foil_probs)),
        "mean_target_logits": float(np.mean(target_logits)),
        "mean_foil_logits": float(np.mean(foil_logits))
    }

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model and tokenizer
    model_dir = args.model_dir
    model_name_or_path = "gpt2"
    
    checkpoint_root = model_dir
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_root} does not exist.")

    checkpoints = []

    for name in os.listdir(checkpoint_root):
        checkpoint_path = os.path.join(checkpoint_root, name)
        if os.path.isdir(checkpoint_path) and name.startswith("checkpoint-"):
            checkpoints.append(checkpoint_path)
    
    checkpoints.sort()

    if len(checkpoints) == 0:
        raise RuntimeError(f"No checkpoints found in {checkpoint_root}")

    results_dir = os.path.join(args.output_dir,"ioi")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    all_checkpoint_results = []

    for checkpoint in checkpoints:

        result = evaluate_checkpoint(checkpoint, "ioi", "ioi_dataset", device)
        all_checkpoint_results.append(result)
        checkpoint_name = os.path.basename(checkpoint)
        output_file = os.path.join(results_dir,f"{checkpoint_name}.json")

        with open(output_file,"w") as f:
            json.dump(result,f,indent=4)

        print(f"\nSaved result to: "f"{output_file}")
    
    summary_file = os.path.join(results_dir,"summary.json")
    with open(summary_file,"w") as f:
        json.dump(all_checkpoint_results,f,indent=4)
    print(f"\nSaved summary to: {summary_file}")
    
    