from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json, os
import argparse

def count_tokens(dataset_name, checkpoint_path):
    SIZES = {
    "1.5M": 1_500_000,
    "5M": 5_000_000,
    }

    OUTPUT_DIR = f"token_counts/{dataset_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    dataset = load_dataset(dataset_name, trust_remote_code=True)["train"]

    for name, size in SIZES.items():
        subset = dataset.select(range(size))
        total_tokens = 0
        total_words = 0
        for example in tqdm(subset, total=size):
            text = example["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)
            total_words += len(text.split())
        output_file = os.path.join(OUTPUT_DIR, f"{name}_stats.json")
        with open(output_file, "w") as f:
            json.dump({"total_tokens": total_tokens, "total_words": total_words}, f)
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", 
                        default="Skylion007/openwebtext", 
                        type=str, 
                        help="Name of the dataset to load.")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Path to the model checkpoint.")
    args = parser.parse_args()
    count_tokens(args.dataset_name, args.checkpoint)