import torch
from setup_model import GPT2Setup
from hooks import Hooks

def test_hooks():
    """
    Test function to demonstrate the functionality of hooks in the GPT-2 model.
    """
    gpt2 = GPT2Setup()
    model = gpt2.model
    tokenizer = gpt2.tokenizer
    device = gpt2.device    

    hooks = Hooks(model)

    hooks.register_all_attentions()

    prompt = "John and Mary went to the market.John gave a bottle of milk to"

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    
    assert len(hooks.activations) == len(model.transformer.h), "Activations not captured for all blocks."

    hooks.print_activation_cache()

    hooks.remove_hooks()

def main():
    test_hooks()

if __name__ == "__main__":
    main()
