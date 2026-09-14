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

    def evaluate_head_patch(self, 
                            sender_layer:int,
                            sender_head:int,
                            receiver_layer:int,
                            receiver_head:int,
                            clean_prompt: str,
                            corrupted_prompt: str,
                            target_correct:str,
                            target_incorrect:str):
        print("Calculating Baseline...")
        clean_baseline = get_logit_diff(self.model,self.tokenizer,clean_prompt,target_correct,target_incorrect,self.device)
        corrupted_baseline = get_logit_diff(self.model, self.tokenizer, corrupted_prompt, target_correct,target_incorrect,self.device)

        input_new = self.tokenizer(clean_prompt, return_tensors ='pt').to(self.device)
        input_orig = self.tokenizer(corrupted_prompt, return_tensors ='pt').to(self.device)

        num_heads = self.model.config.num_attention_heads
        d_model = self.model.config.hidden_size
        head_dim = d_model // num_heads

        sender_c_proj = self.model.transformer.h[sender_layer].attn.c_proj
        receiver_c_attn = self.model.transformer.h[receiver_layer].attn.c_attn

        # Step1
        self.hooks.register_save_c_proj_hook(sender_c_proj, hook_name="sender_clean")
        with torch.no_grad():
            self.model(**input_new)
        c_proj_clean = self.hooks.activations["sender_clean"].clone()
        self.hooks.remove_hooks()

        # Step2
        self.hooks.register_save_c_proj_hook(sender_c_proj,hook_name="sender_corrupt")
        with torch.no_grad():
            self.model(**input_orig)
        c_proj_corrupt = self.hooks.activations["sender_corrupt"].clone()
        self.hooks.remove_hooks()

        # Step3
        batch, seq, _ = c_proj_clean.shape
        diff = (c_proj_clean - c_proj_corrupt).view(batch,seq,num_heads,head_dim)

        head_diff = torch.zeros_like(diff)
        head_diff[:,:,sender_head,:] = diff[:,:,sender_head,:]
        head_diff_flat = head_diff.view(batch,seq,d_model)
        delta_res_stream = torch.matmul(head_diff_flat, sender_c_proj.weight)

        # Step4
        self.hooks.register_patch_qkv_hook(
            target_module=receiver_c_attn,
            delta_res_stream=delta_res_stream,
            receiver_head=receiver_head,
            num_head=num_heads,
            head_dim=head_dim
        )

        patched_diff = get_logit_diff(self.model, self.tokenizer,corrupted_prompt,target_correct,target_incorrect,self.device)
        self.hooks.remove_hooks()

        recovery = (patched_diff-corrupted_baseline)/(clean_baseline-corrupted_baseline)

        return {
            "clean_baseline":clean_baseline,
            "corrupted_baseline":corrupted_baseline,
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

    # results = path_patching.evaluate_path(
    #     8,9,clean_prompt,corrupted_prompt,target_correct,target_incorrect
    # )

    # results = path_patching.evaluate_head_patch(
    #     7,9,9,9,clean_prompt,corrupted_prompt,target_correct,target_incorrect
    # )

    # print(f"Path Patching result:")
    # print("="*50)
    # print(f"Clean Baseline:      {results['clean_baseline']:.4f}")
    # print(f"Corrupted Baseline:  {results['corrupted_baseline']:.4f}")
    # print(f"Patched Logit Diff:  {results['patched_diff']:.4f}")
    # print(f"Recovery:            {results['recovery'] * 100:.2f}%")

    receiver_layer = 8
    receiver_head =6
    num_heads = 12

    recovery_mat = np.zeros((receiver_layer,num_heads))
    imp_edges = []

    print(f"Starting Backwards for Circuit Identification")
    print(f"Target Receiver: Layer {receiver_layer}, Head {receiver_head}")
    print(f"==========================================\n")

    for s_layer in range(receiver_layer):
        for s_head in range(num_heads):
            results = path_patching.evaluate_head_patch(
                sender_layer=s_layer,
                sender_head=s_head,
                receiver_layer=receiver_layer,
                receiver_head=receiver_head,
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                target_correct=target_correct,
                target_incorrect=target_incorrect)

            recovery = results['recovery']
            recovery_mat[s_layer,s_head] = recovery

            if recovery>0.05:
                print(f"Imp Edge: L{s_layer}H{s_head} --> L{receiver_layer}H{receiver_head}")
                imp_edges.append((s_layer,s_head))

    print("\nPath Patching Recovery Matrix:")
    print(np.array2string(
        recovery_mat,
        formatter={"float_kind": lambda x: f"{x:.2f}"}
    ))

    print(f"\nAll important senders to L9H9: {imp_edges}")