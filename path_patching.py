import argparse
from pathlib import Path
import torch
import numpy as np
from hooks import Hooks
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel
from utils import get_logit_diff
from collections import deque

import networkx as nx
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

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


def visualize_circuit(circuit_graph, output_filename="ioi_circuit.png"):
    """
    Function to create graph visualization for the ioi circuit.
    """
    G = nx.DiGraph()

    for receiver, senders in circuit_graph.items():
        r_name = f"L{receiver[0]}H{receiver[1]}"
        G.add_node(r_name, layer=receiver[0])
        
        for s_layer, s_head, score in senders:
            s_name = f"L{s_layer}H{s_head}"
            G.add_node(s_name, layer=s_layer)
            G.add_edge(s_name, r_name, weight=score, label=f"{score*100:.0f}%")

    if len(G.nodes) == 0:
        print("Graph is empty. Nothing to visualize.")
        return

    pos = {}
    layer_y_counts = {}
    
    for node, data in G.nodes(data=True):
        layer = data['layer']
        layer_y_counts[layer] = layer_y_counts.get(layer, 0) + 1

    current_y = {layer: 0 for layer in layer_y_counts}
    
    for node, data in G.nodes(data=True):
        layer = data['layer']
        total_in_layer = layer_y_counts[layer]
        y = current_y[layer] - (total_in_layer - 1) / 2.0
        pos[node] = (layer, y)
        current_y[layer] += 1

    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]

    plt.figure(figsize=(12, 6))
    plt.title("Mechanistic Circuit for Indirect Object Identification (IOI)", fontsize=16)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, width=edge_weights, edge_color="gray", connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    plt.axis("off") 
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    pdf_filename = output_filename.replace(".png", ".pdf")
    plt.savefig(pdf_filename, bbox_inches='tight')
    
    print(f"[*] Visualizations successfully saved to {output_filename} and {pdf_filename}")
    plt.close()

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

    threshold = 0.05
    num_heads = path_patching.model.config.num_attention_heads
    initial_receiver = (9, 9)
    receivers = deque([initial_receiver])
    discovered = {initial_receiver}
    visited_rec = set()
    circuit_graph = {}

    print(f"\nStarting Circuit Discovery ...")
    print("="*50)

    # BFS
    while receivers:
        cur_rec = receivers.popleft()
        r_layer, r_head = cur_rec

        if cur_rec in visited_rec or r_layer == 0:
            continue

        visited_rec.add(cur_rec)
        circuit_graph[cur_rec]= []

        for s_layer in range(r_layer):
            for s_head in range(num_heads):
                result = path_patching.evaluate_head_patch(
                    sender_layer=s_layer,
                    sender_head=s_head,
                    receiver_layer=r_layer,
                    receiver_head=r_head,
                    clean_prompt=clean_prompt,
                    corrupted_prompt=corrupted_prompt,
                    target_correct=target_correct,
                    target_incorrect=target_incorrect
                )

                recovery = result['recovery']

                if recovery > threshold:
                    circuit_graph[cur_rec].append((s_layer,s_head,recovery))

                    sender = (s_layer, s_head)
                    if sender not in discovered:
                        discovered.add(sender)
                        receivers.append(sender)

    print("\n")
    print("="*50)
    print(f"Circuit Identification Completed! Final Circuit:")
    print("="*50)
    for receiver, senders in circuit_graph.items():
        if senders: 
            sender_strings = [f"L{s[0]}H{s[1]} ({s[2]*100:.1f}%)" for s in senders]
            print(f"Receiver L{receiver[0]}H{receiver[1]} gets input from: {', '.join(sender_strings)}")

    output_filename = f"{Path(args.checkpoint).name}.png"
    print("\nGenerating Graph Visualization...")
    visualize_circuit(circuit_graph, output_filename=output_filename)