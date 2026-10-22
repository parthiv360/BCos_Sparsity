import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from hooks import Hooks
from transformers import AutoConfig, AutoTokenizer
from bcos_lm.gpt2 import GPT2LMHeadModel
from utils import get_logit_diff

import networkx as nx
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class CircuitPruner:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(checkpoint_path)
        config._attn_implementation = "eager"

        self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path, config=config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model.eval()

        self.hooks = Hooks(self.model)

        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.d_model = self.model.config.hidden_size
        self.head_dim = self.d_model // self.num_heads

        print(f"Model loaded from {checkpoint_path}.")

    def precompute_delta(self, clean_prompt:str, corrupted_prompt:str):
        """
        Precompute and cache the delta contribution of each sender head.
        """

        input_clean = self.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
        input_corrupt = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.device)

        c_proj_clean = {}
        c_proj_corrupted = {}

        # 1. Clean Activation
        for layer in range(self.num_layers):
            c_proj_module = self.model.transformer.h[layer].attn.c_proj
            self.hooks.register_save_c_proj_hook(c_proj_module, hook_name=f"clean_{layer}")

        with torch.no_grad():
            self.model(**input_clean)

        saved_clean = { layer: self.hooks.activations[f"clean_{layer}"].clone() for layer in range(self.num_layers) }
        self.hooks.remove_hooks()

        # 2. Corrupted Activation
        for layer in range(self.num_layers):
            c_proj_module = self.model.transformer.h[layer].attn.c_proj
            self.hooks.register_save_c_proj_hook(c_proj_module, hook_name=f"corrupted_{layer}")

        with torch.no_grad():
            self.model(**input_corrupt)

        saved_corrupted = { layer: self.hooks.activations[f"corrupted_{layer}"].clone() for layer in range(self.num_layers) }
        self.hooks.remove_hooks()

        # 3. Compute Delta
        deltas = {}

        for layer in range(self.num_layers):
            clean_activation = saved_clean[layer]
            corrupted_activation = saved_corrupted[layer]

            W = self.model.transformer.h[layer].attn.c_proj.weight

            for head in range(self.num_heads):
                idx = head*self.head_dim
                end_idx = (head+1)*self.head_dim
                delta = clean_activation[:, :, idx:end_idx] - corrupted_activation[:, :, idx:end_idx]

                head_W = W[idx:end_idx, :]
                delta_res = torch.matmul(delta, head_W)

                deltas[(layer, head)] = delta_res
        return deltas

    def evaluate_circuit(self, corrupted_prompt:str, target_correct: str, target_incorrect:str, active_edges:list, precomputed_deltas:dict ):
        "Evaluate a newly pruned sparse circuit."

        receiver = {}
        for sender_layer, sender_head, receiver_layer, receiver_head in active_edges:
            if (receiver_layer, receiver_head) not in receiver:
                receiver[(receiver_layer, receiver_head)] = []
            receiver[(receiver_layer, receiver_head)].append((sender_layer, sender_head)) #if multiple senders to same receiver

        for (receiver_layer, receiver_head), senders in receiver.items():
            total_delta = None
            for sender in senders:
                if total_delta is None:
                    total_delta = precomputed_deltas[sender].clone()
                else:
                    total_delta += precomputed_deltas[sender]

            receiver_c_attn = self.model.transformer.h[receiver_layer].attn.c_attn

            self.hooks.register_patch_qkv_hook(
                target_module=receiver_c_attn,
                delta_res_stream=total_delta,
                receiver_head=receiver_head,
                num_head=self.num_heads,
                head_dim=self.head_dim,
            )

        patched_diff = get_logit_diff(self.model,self.tokenizer,corrupted_prompt,target_correct,target_incorrect,device=self.device)
        self.hooks.remove_hooks()
        return patched_diff

    def prune(self, clean_prompt:str, corrupted_prompt:str, target_correct:str, target_incorrect:str, threshold:float):
        """
        Prune the circuit based on the precomputed delta contributions.
        """

        print("Calculating Baselines ...")
        clean_baseline = get_logit_diff(self.model, self.tokenizer, clean_prompt, target_correct, target_incorrect, device=self.device)
        corrupted_baseline = get_logit_diff(self.model, self.tokenizer, corrupted_prompt, target_correct, target_incorrect, device=self.device)

        print(f"Clean Baseline: {clean_baseline:.4f}, Corrupted Baseline: {corrupted_baseline:.4f}")

        print("Precomputing Delta Contributions ...")
        precomputed_deltas = self.precompute_delta(clean_prompt, corrupted_prompt)

        edge_scores = {}

        # name_mover_heads = [(9,7),(9,9),(10,0)] #obtained from activation patching experiment

        all_edges = [(sender_layer, sender_head, receiver_layer, receiver_head)
                     for receiver_layer in range(self.num_layers)
                     for receiver_head in range(self.num_heads)
                     for sender_layer in range(receiver_layer)
                     for sender_head in range(self.num_heads)]

        print("Evaluating complete circuit ...")
        full_circuit_diff = self.evaluate_circuit(corrupted_prompt, target_correct, target_incorrect, all_edges, precomputed_deltas)
        full_circuit_recovery = (full_circuit_diff - corrupted_baseline) / (clean_baseline - corrupted_baseline + 1e-8)
        print(f"Full Circuit Recovery Score: {full_circuit_recovery:.4f}")

        target_diff = corrupted_baseline + threshold * (full_circuit_diff - corrupted_baseline)

        edge_norms = {}
        for edge in all_edges:
            sender_layer, sender_head, receiver_layer, receiver_head = edge
            edge_weight = precomputed_deltas[(sender_layer, sender_head)]
            edge_norms[edge]= precomputed_deltas[(sender_layer, sender_head)].norm().item()

        candidate_to_prune = sorted(all_edges, key=lambda e: edge_norms[e])

        active_edges_set = set(all_edges)

        for edge in tqdm(candidate_to_prune, desc="Evaluating Edges"):
            active_edges_set.remove(edge)
            current_diff = self.evaluate_circuit(corrupted_prompt, target_correct, target_incorrect, list(active_edges_set), precomputed_deltas)

            current_gain = current_diff - corrupted_baseline
            full_gain = full_circuit_diff - corrupted_baseline

            current_recovery = current_gain / (full_gain + 1e-8)
            if current_recovery < threshold:
                active_edges_set.add(edge)

        final_circuit = list(active_edges_set)

        # Cross Verify the scores
        final_diff = self.evaluate_circuit(corrupted_prompt, target_correct, target_incorrect, final_circuit, precomputed_deltas)
        recovery = (final_diff - corrupted_baseline) / (clean_baseline - corrupted_baseline + 1e-8)
        print(f"Final Recovery Score: {recovery:.4f} (Target: {threshold})")
        print(f"Pruning Completed. Total edges retained: {len(final_circuit)}.")

        circuit = {}
        for sender_layer, sender_head, receiver_layer, receiver_head in final_circuit:
            if (receiver_layer, receiver_head) not in circuit:
                circuit[(receiver_layer, receiver_head)] = []
            circuit[(receiver_layer, receiver_head)].append((sender_layer, sender_head, edge_weight[(sender_layer, sender_head, receiver_layer, receiver_head)]))
        return circuit

def visualize_circuit(circuit:dict, output="ioi_pruned_circuit.png"):
    """
    Visualize the pruned circuit.
    """

    G = nx.DiGraph()

    for receiver,senders in circuit.items():
        r_name = f"L{receiver[0]}H{receiver[1]}"
        G.add_node(r_name, layer=receiver[0])

        for sender_layer, sender_head, score in senders:
            s_name = f"L{sender_layer}H{sender_head}"
            G.add_node(s_name, layer=sender_layer)
            G.add_edge(s_name, r_name, weight=abs(score),label=f"{score*100:.0f}%")

    if len(G.nodes) == 0:
        print("No edges to visualize.")
        return

    plt.figure(figsize=(20,12))
    plt.title("Pruned IoI Circuit", fontsize=20, pad=20)
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    for node in pos:
        pos[node][1]*= 2.5
    edge_weights = [max(1.0, G[u][v]['weight'] *10) for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=2800, node_color="#87CEEB", edgecolors="black", linewidths=2.0)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, width=edge_weights, edge_color="dimgray", connectionstyle="arc3,rad=0.25", alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    edge_labels = {(u,v): data['label'] for u,v,data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,font_weight="bold", label_pos=0.3, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Circuit visualization saved to {output}.")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Threshold for pruning.")

    args = parser.parse_args()

    clean_prompt = "When John and Mary went to the store, John gave a bottle of milk to"
    corrupted_prompt = "When John and Mary went to the store, Mary gave a bottle of milk to"
    target_correct = "Mary"
    target_incorrect = "John"

    pruner = CircuitPruner(args.checkpoint_path)

    final_circuit = pruner.prune(clean_prompt, corrupted_prompt, target_correct, target_incorrect, args.threshold)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path("pruned_circuit")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = str(output_dir / f"{checkpoint_path.parent.name}_{checkpoint_path.name}_pruned.png")

    visualize_circuit(final_circuit, output=output_filename)