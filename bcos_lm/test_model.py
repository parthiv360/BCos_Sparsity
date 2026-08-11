import torch
from transformers import AutoConfig, AutoTokenizer
from .gpt2 import GPT2LMHeadModel
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import os

file_path = "dataset/ioi_dataset.json"

# load dataset
def load_dataset(file_path):
    dataset = {}
    with open(file_path, "r") as f:
        dataset = json.load(f)
        print(f"Dataset loaded from {file_path}")
    
    return dataset

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    # Load the model and tokenizer
    model_dir = args.model_dir
    model_name_or_path = "gpt2"
    config = AutoConfig.from_pretrained(model_dir)
    if "gpt" in model_dir:
        model = GPT2LMHeadModel.load_from_pretrained(model_dir, config=config)
    else:
        raise ValueError(f"Model {model_dir} not supported.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load the dataset
    for data_type in datasets.keys():

        all_prob_diff_results = {}
        for dataset_name in datasets[data_type]:
            dataset = load_dataset(data_type, dataset_name)
            all_prob_diff_results[dataset_name] = {}
            print(f"Loaded {dataset_name} dataset with {len(dataset['prefix'])} examples.")
            prob_diff_results = []
            target_probs = []
            foil_probs = []
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
                    probabilities = torch.softmax(logits, dim=-1)
                    target_prob = probabilities[0, -1, target_ids].item()
                    foil_prob = probabilities[0, -1, foil_ids].item()
                    prob_diff = target_prob - foil_prob
                prob_diff_results.append(prob_diff)
                target_probs.append(target_prob)
                foil_probs.append(foil_prob)
            all_prob_diff_results[dataset_name]["mean_prob_diff"] = np.mean(prob_diff_results)
            all_prob_diff_results[dataset_name]["mean_target_probs"] = np.mean(target_probs)
            all_prob_diff_results[dataset_name]["mean_foil_probs"] = np.mean(foil_probs)
            print(f"Mean probability difference for {dataset_name}: {all_prob_diff_results[dataset_name]['mean_prob_diff']}")
            print(f"Mean target probability for {dataset_name}: {all_prob_diff_results[dataset_name]['mean_target_probs']}")
            print(f"Mean foil probability for {dataset_name}: {all_prob_diff_results[dataset_name]['mean_foil_probs']}")
        
        all_prob_diff_results["overall"] = {}
        all_prob_diff_results["overall"]["prob_diff_mean"] = np.mean([all_prob_diff_results[dataset_name]["mean_prob_diff"] for dataset_name in datasets[data_type]])
        all_prob_diff_results["overall"]["target_probs_mean"] = np.mean([all_prob_diff_results[dataset_name]["mean_target_probs"] for dataset_name in datasets[data_type]])
        all_prob_diff_results["overall"]["foil_probs_mean"] = np.mean([all_prob_diff_results[dataset_name]["mean_foil_probs"] for dataset_name in datasets[data_type]])
        
        # Save the explanations to a file
        if not os.path.exists(f"{results_dir}/{data_type}"):
            os.makedirs(f"{results_dir}/{data_type}")
        output_file = f"{results_dir}/{data_type}/probability_differences.json"
        with open(output_file, 'w') as f:
            json.dump(all_prob_diff_results, f, indent=4)
        print(f"Saved explanations to {output_file}")
        print(f"Finished processing {data_type} dataset.")