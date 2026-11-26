import argparse

from bcos_lm.gpt2 import GPT2LMHeadModel
import torch
from setup_model import GPT2Setup
from hooks import Hooks
from utils import get_logit_diff
from transformers import AutoConfig, AutoTokenizer
import numpy as np

class ActivationPatching:
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

    def head_activation_patching(self, layer, head_idx, clean_prompt, corrupted_prompt, target_correct, target_incorrect):
        """
        Perform head activation patching by comparing the logit differences between a clean and corrupted prompt.
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

        attn = self.model.transformer.h[layer].attn
        self.hooks.save_head_activation(layer)
        inputs = self.tokenizer(clean_prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            self.model(**inputs)
        clean_activation = self.hooks.get_head_activation(layer)
        self.hooks.clear_head_activation(layer)
        print(f"Head activation captured for layer {layer}, head {head_idx}: {clean_activation.shape}")

        self.hooks.patch_head_activation(layer, head_idx, clean_activation)
        patched_diff = get_logit_diff(model=self.model,
                                      tokenizer=self.tokenizer,
                                        prompt=corrupted_prompt,
                                        correct=target_correct,
                                        incorrect=target_incorrect,
                                        device=self.device)
        self.hooks.clear_patch_activation(layer)
        recovery = (patched_diff - incorrect_diff) / (correct_diff - incorrect_diff + 1e-8)  
        print(f"Logit difference after patching on {corrupted_prompt}: {patched_diff}")
        print(f"Recovery after patching: {recovery:.2f}")
        return recovery
        
def create_prompt_pairs(prompts, answers):
    assert len(prompts) == len(answers)
    assert len(prompts) % 2 == 0

    prompt_pairs = []

    for i in range(0, len(prompts), 2):

        clean_prompt = prompts[i]
        corrupted_prompt = prompts[i + 1]

        correct_clean, incorrect_clean = answers[i]
        correct_corrupt, incorrect_corrupt = answers[i + 1]

        assert correct_clean == incorrect_corrupt
        assert incorrect_clean == correct_corrupt

        prompt_pairs.append({
            "clean": clean_prompt,
            "corrupted": corrupted_prompt,
            "correct": correct_clean.strip(),
            "incorrect": incorrect_clean.strip(),
        })

    return prompt_pairs


def main():
    # activation_patching = ActivationPatching()
    # activation_patching.test_hooks()
    # block_index = 9
    # activation_patching.print_specific_activation(block_index)
    # activation_patching.hooks.remove_hooks()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )

    args = parser.parse_args()

    prompts = [
        "When John and Mary went to the shops, John gave the bag to",
        "When John and Mary went to the shops, Mary gave the bag to",
        "When Tom and James went to the park, James gave the ball to",
        "When Tom and James went to the park, Tom gave the ball to",
        "When Dan and Sid went to the shops, Sid gave an apple to",
        "When Dan and Sid went to the shops, Dan gave an apple to",
        "After Martin and Amy went to the park, Amy gave a drink to",
        "After Martin and Amy went to the park, Martin gave a drink to",
        ]
    
    answers = [
        (" Mary", " John"),
        (" John", " Mary"),
        (" Tom", " James"),
        (" James", " Tom"),
        (" Dan", " Sid"),
        (" Sid", " Dan"),
        (" Martin", " Amy"),
        (" Amy", " Martin"),
    ]

    prompt_pairs = create_prompt_pairs(prompts, answers)

    activation_patcher = ActivationPatching(args.checkpoint)
    # recoveries = []
    # for i in range(12):
    #     recovery = activation_patcher.activation_patching(layer=activation_patcher.model.transformer.h[i].attn,
    #                                        hook_name=f"block_{i}.attn",
    #                                        clean_prompt=clean_prompt,
    #                                        corrupted_prompt=corrupted_prompt,
    #                                        target_correct=target_correct,
    #                                        target_incorrect=target_incorrect,
    #                                        target_layer=activation_patcher.model.transformer.h[i].attn)
    #     recoveries.append(recovery)
    # for i, recovery in enumerate(recoveries):
    #     print(f"Recovery for block {i}: {recovery:.2f}")
    all_recoveries = []
    for i,pair in enumerate(prompt_pairs):
        print(f"\nProcessing prompt pair {i+1}/{len(prompt_pairs)}")

        recoveries = []
        for layer in range(12):
            layer_recoveries = []
            for head in range(12):
                recovery = activation_patcher.head_activation_patching(layer=layer,
                                                                    head_idx=head,
                                                                    clean_prompt=pair["clean"],
                                                                    corrupted_prompt=pair["corrupted"],
                                                                    target_correct=pair["correct"],
                                                                    target_incorrect=pair["incorrect"])
                layer_recoveries.append(recovery)
            recoveries.append(layer_recoveries)
        all_recoveries.append(recoveries)

    mean_recoveries = np.mean(np.stack(all_recoveries), axis=0)

    print("\nRecovery Matrix:")
    print(np.array2string(
        np.array(mean_recoveries),
    formatter={"float_kind": lambda x: f"{x:.2f}"}
    ))




if __name__ == "__main__":
    main()