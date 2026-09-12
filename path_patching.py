import argparse
import torch
import numpy as np
from hooks import Hooks
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel

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

        with torch.no_grad():
            output = self.model(**input_orig)
            logits = output.logits
        self.hooks.remove_hooks()

        correct_token = self.tokenizer(" " + target_correct, return_tensors='pt').input_ids.to(self.device)
        incorrect_token = self.tokenizer(" " + target_incorrect, return_tensors='pt').input_ids.to(self.device)
        
        last_token_logits = logits[0, -1, :]
        logit_diff = last_token_logits[correct_token[0, 0]] - last_token_logits[incorrect_token[0, 0]]
        
        return logit_diff.item()

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

    eval_path = path_patching.evaluate_path(
        2,9,clean_prompt,corrupted_prompt,target_correct,target_incorrect
    )

    print(f"Path Patching result: {eval_path}")
