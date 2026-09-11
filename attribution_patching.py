import argparse
import torch
import numpy as np
from hooks import Hooks
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel
from utils import get_logit_diff_with_grad

class AttributionPatching:
    def __init__(self,checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(checkpoint_path)
        config._attn_implementation = "eager"

        self.model = GPT2LMHeadModel.load_from_pretrained(checkpoint_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        self.model.to(self.device)
        self.model.train() # Should be in train mode for attribution patching to work properly.

        for param in self.model.parameters():
            param.requires_grad = False

        self.hooks = Hooks(self.model)
        print(f"Model loaded from {checkpoint_path}.")

    def layer_attribution_patching(self, clean_prompt, corrupted_prompt, target_correct, target_incorrect):
        """
        Performs attribution patching on all layers.
        """

        n_layers = len(self.model.transformer.h)

        # Clean Pass
        self.hooks.register_all_blocks()
        with torch.no_grad():
            self.model(**self.tokenizer(clean_prompt, return_tensors='pt').to(self.device))

        clean_cache = {k: v.clone().detach() for k, v in self.hooks.activations.items()}
        self.hooks.remove_hooks()

        # Corrupted Pass
        for i, block in enumerate(self.model.transformer.h):
            self.hooks.register_save_hook_with_grad(block, hook_name=f"block_{i}")

        with torch.enable_grad():
            outputs = self.model(**self.tokenizer(corrupted_prompt, return_tensors='pt').to(self.device))
            logit_diff = get_logit_diff_with_grad(outputs.logits, self.tokenizer, target_correct, target_incorrect, self.device)
            logit_diff.backward()

        # Taylor Expression Calculation
        attributions =[]
        for i in range(n_layers):
            layer_name = f"block_{i}"
            a_clean = clean_cache[layer_name]
            a_corrupt = self.hooks.activations[layer_name]
            grad_a_corrupt = a_corrupt.grad

            attr = torch.sum((a_clean-a_corrupt.detach()) * grad_a_corrupt).item()
            attributions.append(attr)

        self.hooks.remove_hooks()
        return attributions

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

    attribute_patching = AttributionPatching(checkpoint_path=args.checkpoint)

    layer_attr = attribute_patching.layer_attribution_patching(
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        target_correct=target_correct,
        target_incorrect=target_incorrect
    )

    print("\nLayer Attributions (Approximated Recovery):")
    for i, attr in enumerate(layer_attr):
        print(f"Layer {i}: {attr:.4f}")