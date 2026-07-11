import torch
from setup_model import GPT2Setup
from hooks import Hooks

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


def main():
    activation_patching = ActivationPatching()
    activation_patching.test_hooks()
    block_index = 9
    activation_patching.print_specific_activation(block_index)
    activation_patching.hooks.remove_hooks()

if __name__ == "__main__":
    main()