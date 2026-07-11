import torch
from setup_model import GPT2Setup
from hooks import Hooks
from utils import get_logit_diff

class ActivationPatching:
    def __init__(self):

        self.gpt2 = GPT2Setup()
        self.model = self.gpt2.model
        self.tokenizer = self.gpt2.tokenizer
        self.device = self.gpt2.device
        self.hooks = Hooks(self.model)

    def test_hooks(self):
        """
        Test function to demonstrate the functionality of hooks in the GPT-2 model.
        """

        self.hooks.register_all_blocks()

        prompt = "John and Mary went to the market. John gave a bottle of milk to"

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        assert len(self.hooks.activations) == len(self.model.transformer.h), "Activations not captured for all blocks."

        self.hooks.print_activation_cache()

    def print_specific_activation(self, block_index: int):
        """
        Prints the activation for a specific block's attention layer.
        """

        activation_key = f"block_{block_index}"
        if activation_key in self.hooks.activations:
            print(f"Activation for {activation_key}:")
            print(self.hooks.activations[activation_key])
        else:
            print(f"No activation found for {activation_key}.")
    
    def activation_patching(self,layer, hook_name, clean_prompt, corrupted_prompt, target_correct, target_incorrect, target_layer):
        """
        Perform activation patching by comparing the logit differences between a clean and corrupted prompt.
        """

        print(f"Establishing baseline logit difference...")
        correct_diff = get_logit_diff(model=self.model, 
                                      tokenizer=self.tokenizer, 
                                      prompt=clean_prompt, 
                                      correct=target_correct, 
                                      incorrect=target_incorrect, 
                                      device=self.device)
        incorrect_diff = get_logit_diff(model=self.model,
                                        tokenizer=self.tokenizer, 
                                      prompt=corrupted_prompt, 
                                      correct=target_correct, 
                                      incorrect=target_incorrect, 
                                      device=self.device)
        
        print(f"Baseline logit difference for clean prompt: {correct_diff}")
        print(f"Baseline logit difference for corrupted prompt: {incorrect_diff}")

        self.hooks.register_save_hook(layer, hook_name)
        inputs = self.tokenizer(clean_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        
        if hook_name not in self.hooks.activations:
            raise RuntimeError(f"No activation captured for {hook_name}")
        
        activation = self.hooks.activations[hook_name]
        print(f"Activation captured for {hook_name}: {activation.shape}")
        self.hooks.remove_hooks()

        self.hooks.register_patch_hook(target_layer, activation, hook_name="patch_"+hook_name)
        patched_diff = get_logit_diff(model=self.model,
                                      tokenizer=self.tokenizer, 
                                      prompt=corrupted_prompt, 
                                      correct=target_correct, 
                                      incorrect=target_incorrect, 
                                      device=self.device)
        self.hooks.remove_hooks()
        recovery = (patched_diff - incorrect_diff) / (correct_diff - incorrect_diff + 1e-8)  
        print(f"Logit difference after patching on {corrupted_prompt}: {patched_diff}")
        print(f"Recovery after patching: {recovery:.2f}")

        return recovery

def main():
    # activation_patching = ActivationPatching()
    # activation_patching.test_hooks()
    # block_index = 9
    # activation_patching.print_specific_activation(block_index)
    # activation_patching.hooks.remove_hooks()

    clean_prompt = "When John and Mary went to the store, John gave a bottle of milk to"
    corrupted_prompt = "When John and Mary went to the store, Mary gave a bottle of milk to"

    target_correct = "Mary"
    target_incorrect = "John"

    activation_patcher = ActivationPatching()
    recoveries = []
    for i in range(12):
        recovery = activation_patcher.activation_patching(layer=activation_patcher.model.transformer.h[i].attn,
                                           hook_name=f"block_{i}.attn",
                                           clean_prompt=clean_prompt,
                                           corrupted_prompt=corrupted_prompt,
                                           target_correct=target_correct,
                                           target_incorrect=target_incorrect,
                                           target_layer=activation_patcher.model.transformer.h[i].attn)
        recoveries.append(recovery)
    for i, recovery in enumerate(recoveries):
        print(f"Recovery for block {i}: {recovery:.2f}")

if __name__ == "__main__":
    main()