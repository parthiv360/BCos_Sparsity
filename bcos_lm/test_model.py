import torch
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np

DATA_DIR = "BCos_Sparsity/data"
        
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
    config.attn_implementation = "eager"
    if "gpt" in model_dir:
        model = GPT2LMHeadModel.load_from_pretrained(model_dir, config=config)
    else:
        raise ValueError(f"Model {model_dir} not supported.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results_dir = f"decoder_experiments/{args.output_dir}/ioi"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load the dataset
    for data_type in datasets.keys():

        all_results = {}
        for dataset_name in datasets[data_type]:
            dataset = load_dataset(data_type, dataset_name)
            all_results[dataset_name] = {}
            print(f"Loaded {dataset_name} dataset with {len(dataset['prefix'])} examples.")
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
            all_results[dataset_name]["accuracy"] = accuracy
            all_results[dataset_name]["total"] = total
            all_results[dataset_name]["correct"] = correct
            all_results[dataset_name]["mean_prob_diff"] = np.mean(prob_diff_results)
            all_results[dataset_name]["mean_target_probs"] = np.mean(target_probs)
            all_results[dataset_name]["mean_foil_probs"] = np.mean(foil_probs)
            
            print("\n" + "=" * 60)
            print(f"IOI RESULTS: {dataset_name}")
            print("=" * 60)

            print(f"Accuracy for {dataset_name}: {all_results[dataset_name]['accuracy'] * 100:.2f}%")
            print(f"Mean probability difference for {dataset_name}: {all_results[dataset_name]['mean_prob_diff']}")
            print(f"Mean target probability for {dataset_name}: {all_results[dataset_name]['mean_target_probs']}")
            print(f"Mean foil probability for {dataset_name}: {all_results[dataset_name]['mean_foil_probs']}")
        
        all_results["overall"] = {}
        all_results["overall"]["accuracy_mean"] = np.mean([all_results[dataset_name]["accuracy"] for dataset_name in datasets[data_type]])
        all_results["overall"]["prob_diff_mean"] = np.mean([all_results[dataset_name]["mean_prob_diff"] for dataset_name in datasets[data_type]])
        all_results["overall"]["target_probs_mean"] = np.mean([all_results[dataset_name]["mean_target_probs"] for dataset_name in datasets[data_type]])
        all_results["overall"]["foil_probs_mean"] = np.mean([all_results[dataset_name]["mean_foil_probs"] for dataset_name in datasets[data_type]])
        
        # Save the explanations to a file
        if not os.path.exists(f"{results_dir}/{data_type}"):
            os.makedirs(f"{results_dir}/{data_type}")
        output_file = f"{results_dir}/{data_type}/probability_differences.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved explanations to {output_file}")
        print(f"Finished processing {data_type} dataset.")