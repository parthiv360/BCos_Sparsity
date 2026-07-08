import os
import torch
import json
from setup_model import GPT2Setup

class IOIEvaluator:
    """
    IOI Evaluator class for evaluating the model's predictions on IOI tasks.
    """

    def __init__(self, gpt2_setup: GPT2Setup):
        """
        Initializes the GPT2 gpt2_setup.
        """
        self.gpt2_setup = gpt2_setup

    def evaluate_prompt(self, prompt, correct, incorrect):
        """
        Evaluates a single prompt and extracts raw logits.

        Returns:
            tuple: (is_correct (bool), logit_difference (float))
        """
        inputs = self.gpt2_setup.tokenizer(prompt, return_tensors="pt").to(self.gpt2_setup.device)
        correct_input = self.gpt2_setup.tokenizer(" " + correct, return_tensors="pt").input_ids.to(self.gpt2_setup.device)
        incorrect_input = self.gpt2_setup.tokenizer(" " + incorrect, return_tensors="pt").input_ids.to(self.gpt2_setup.device)

        # Get logits
        with torch.no_grad():
            outputs = self.gpt2_setup.model(**inputs)
            # Slice to get the final token's logit distribution
            logits = outputs.logits[:, -1, :] 

        # Extract raw logits instead of softmax probabilities
        correct_logit = logits[0, correct_input[0, 0]].item()
        incorrect_logit = logits[0, incorrect_input[0, 0]].item()

        # Calculate metrics
        is_correct = correct_logit > incorrect_logit
        logit_diff = correct_logit - incorrect_logit

        return is_correct, logit_diff

    def evaluate_dataset(self, dataset_path):
        """
        Evaluates the entire dataset.

        Returns:
            tuple: (accuracy (float), average_logit_difference (float))
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        correct_count = 0
        total_logit_diff = 0.0
        
        for item in dataset:
            prompt = item["prompt"]
            correct = item["correct"]
            incorrect = item["incorrect"]

            is_correct, logit_diff = self.evaluate_prompt(prompt, correct, incorrect)
            
            if is_correct:
                correct_count += 1
            total_logit_diff += logit_diff

        accuracy = correct_count / len(dataset)
        avg_logit_diff = total_logit_diff / len(dataset)
        
        return accuracy, avg_logit_diff


def main():
    dataset_path = "dataset/ioi_dataset.json" 
    gpt2_setup = GPT2Setup()
    evaluator = IOIEvaluator(gpt2_setup)
    
    print(f"Evaluating dataset: {dataset_path}")
    accuracy, avg_logit_diff = evaluator.evaluate_dataset(dataset_path)
    
    print("-" * 40)
    print(f"Accuracy               : {accuracy * 100:.2f}%")
    print(f"Avg Logit Difference   : {avg_logit_diff:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()