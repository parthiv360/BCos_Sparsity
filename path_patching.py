import argparse
import torch
import numpy as np
from hooks import Hooks
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel
from utils import get_logit_diff
class PathPatching:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(checkpoint_path)
        config._attn_implementation = "eager"

        self.model = GPT2LMHeadModel.load_from_pretrained(checkpoint_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()  

        self.hooks = Hooks(self.model)
        print(f"Model loaded from {checkpoint_path}.")

    def evaluate_path(self, sender_layer: int, receiver_layer: int, clean_prompt: str, corrupted_prompt: str, target_correct: str, target_incorrect: str ):
        print("Calculating Baseline...")
        clean_baseline = get_logit_diff(self.model,self.tokenizer,clean_prompt,target_correct,target_incorrect,self.device)
        corrupted_baseline = get_logit_diff(self.model, self.tokenizer, corrupted_prompt, target_correct,target_incorrect,self.device)

        input_new = self.tokenizer(clean_prompt, return_tensors ='pt').to(self.device)
        input_orig = self.tokenizer(corrupted_prompt, return_tensors ='pt').to(self.device)

        sender_module = self.model.transformer.h[sender_layer]
        receiver_module = self.model.transformer.h[receiver_layer]

        # Step 1:
        self.hooks.register_save_hook(sender_module, hook_name="sender_new")
        with torch.no_grad():
            self.model(**input_new)

        a_new_sender = self.hooks.activations["sender_new"].clone()
        self.hooks.remove_hooks()

        #Step 2:
        self.hooks.register_save_hook(sender_module, hook_name="sender_orig")
        with torch.no_grad():
            self.model(**input_orig)

        a_orig_sender = self.hooks.activations["sender_orig"].clone()
        self.hooks.remove_hooks()

        #Step 3:
        self.hooks.register_path_patch_hook(
            receiver_module=receiver_module,
            a_orig=a_orig_sender,
            a_new=a_new_sender,
            hook_name=f"patch_{sender_layer}_to_{receiver_layer}"
        )

        patched_diff = get_logit_diff(self.model, self.tokenizer, corrupted_prompt, target_correct, target_incorrect, self.device)
        self.hooks.remove_hooks()

        recovery = (patched_diff - corrupted_baseline) / (clean_baseline - corrupted_baseline + 1e-8)
        
        return {
            "clean_baseline": clean_baseline,
            "corrupted_baseline": corrupted_baseline,
            "patched_diff": patched_diff,
            "recovery": recovery
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        )
    args = parser.parse_args()

    clean_prompt = "When John and Mary went to the store, John gave a bottle of milk to"
    corrupted_prompt = "When John and Mary went to the store, Mary gave a bottle of milk to"
    target_correct = "Mary"
    target_incorrect = "John"

    path_patching = PathPatching(args.checkpoint)

    results = path_patching.evaluate_path(
        2,9,clean_prompt,corrupted_prompt,target_correct,target_incorrect
    )

    print(f"Path Patching result:")
    print("="*50)
    print(f"Clean Baseline:      {results['clean_baseline']:.4f}")
    print(f"Corrupted Baseline:  {results['corrupted_baseline']:.4f}")
    print(f"Patched Logit Diff:  {results['patched_diff']:.4f}")
    print(f"Recovery:            {results['recovery'] * 100:.2f}%")
