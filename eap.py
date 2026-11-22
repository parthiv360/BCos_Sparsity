import argparse

from bcos_lm.gpt2 import GPT2LMHeadModel
import torch
from typing import Dict, List, Tuple
from transformers import AutoConfig, AutoTokenizer
from Collections import defaultdict

class EdgeAttributionPatching:
    def __init__(self,model,tokenizer,device=None):
        self.model = model
        self.tokenizer = tokenizer

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.d_model = model.config.hidden_size
        self.head_dim = self.d_model // self.num_heads

        self.sender_inputs = {}
        self.receiver_outputs = {}

        self.handles = []

    def tokenize(self,prompt):
        return self.tokenizer(prompt,return_tensors="pt").to(self.device)

    def get_target_id(self,target):
        ids = self.tokenizer.encode(" " + target.strip(), add_special_tokens=False)
        if(len(ids) != 1):
            raise ValueError(f"Target '{target}' should correspond to a single token, but got {len(ids)} tokens.")
        return ids[0]

    def logit_diff(self, logits, correct_id, incorrect_id):
        # Shape of Logits: (batch_size, sequence_length, vocab_size)
        # Shape of Next Token Logits: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]
        return (next_token_logits[:, correct_id] - next_token_logits[:, incorrect_id]).mean()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
    def capture_c_proj_inputs(self,prompt):
        """
        Capture the inputs to the c_proj layers of all attention heads in all layers.
        Save this information in self.sender_inputs, which is a dictionary mapping layer 
        to the input tensor of shape (batch_size, sequence_length, d_model).
        """
        self.sender_inputs.clear()

        def make_hook(layer):
            def hook(module, inputs):
                self.sender_inputs[layer] = inputs[0].detach().clone()
            return hook

        for layer in range(self.num_layers):
            module = self.model.transformer.h[layer].attn.c_proj
            handle = module.register_forward_pre_hook(make_hook(layer))
            self.handles.append(handle)

        try:
            tokens = self.tokenize(prompt)
            with torch.no_grad():
                self.model(**tokens, use_cache=False)
        finally:
            self.remove_hooks()

        return {
            layer: activation.clone()
            for layer, activation in self.sender_inputs.items()
        }

    def capture_receiver_gradients(self, prompt,correct_id, incorrect_id):
        """
        Capture the gradients wrt each layer's attention output.
        Stores it in self.receiver_outputs with shape (batch_size, sequence_length, d_model).
        """
        self.receiver_outputs.clear()
        self.model.zero_grad(set_to_none=True)

        def make_hook(layer):
            def hook(module, inputs, output):

                if not output.requires_grad:
                    output.requires_grad_(True)
                output.retain_grad() # To read the gradient later. 

                self.receiver_outputs[layer] = output
            return hook

        for layer in range(self.num_layers):
            module = self.model.transformer.h[layer].attn.c_attn
            handle = module.register_forward_hook(make_hook(layer))
            self.handles.append(handle)

        try:
            tokens= self.tokenize(prompt)
            outputs = self.model(**tokens, use_cache=False)
            logits = outputs.logits
            loss = self.logit_diff(logits, correct_id, incorrect_id)
            loss.backward()

            gradients = {}

            for layer, activation in self.receiver_outputs.items():
                if activation.grad is None:
                    raise RuntimeError(f"Gradient for layer {layer} is None. ")
                gradients[layer] = activation.grad.detach().clone()
        finally:
            self.remove_hooks()

        return loss.item(),gradients

    def compute_sender_deltas(self, clean_inputs, corrupted_inputs):
        """
        COmpute the derivative of the c_proj layer's output wrt corrupted prompt
        Store for each layer and head.
        """
        sender_deltas = {}
        for layer in range(self.num_layers):
            clean = clean_inputs[layer]
            corrupted = corrupted_inputs[layer]

            # Shape of clean and corrupted should be the same: (batch_size, sequence_length, d_model)
            assert clean.shape == corrupted.shape, (f"Shape mismatch for layer {layer}: {clean.shape} vs {corrupted.shape}")

            c_proj = self.model.transformer.h[layer].attn.c_proj

            for head in range(self.num_heads):
                start_idx = head * self.head_dim
                end_idx = (head + 1) * self.head_dim

                per_head = torch.zeros_like(corrupted)
                per_head[:,:, start_idx:end_idx] = clean[:, :, start_idx:end_idx] - corrupted[:, :, start_idx:end_idx]

                with torch.enable_grad():
                    _, delta = torch.autograd.functional.jvp(c_proj,
                                                             corrupted,
                                                             per_head,
                                                             create_graph=False,
                                                             strict=False)
                sender_deltas[(layer, head)] = delta.detach() # output shape: (batch_size, sequence_length, d_model)

        return sender_deltas

    def compute_edge_scores(self, sender_deltas, receiver_gradients):
        """
        Compute the edge scores for each (sender_layer, sender_head, receiver_layer, receiver_head)
        The edge score is computed as the dot product of the sender delta and the receiver gradient.
        """
        edge_scores = {}
        signed_edge_scores = {}

        for receiver_layer in range(self.num_layers):
            receiver_module = self.model.transformer.h[receiver_layer].attn.c_attn
            weight = receiver_module.weight

            if weight.shape != (self.d_model, 3*self.d_model):
                raise ValueError(f"Unexpected weight shape for layer {receiver_layer}: {weight.shape}")

            grad = receiver_gradients[receiver_layer]
            batch, seq_len, _ = grad.shape

            grad = grad.reshape(batch,seq_len,3,self.num_heads,self.head_dim) # 3 for Q,K,V

            for receiver_head in range(self.num_heads):
                receiver_grad = grad[:,:, :, receiver_head, :]

                for sender_layer in range(receiver_layer):
                    for sender_head in range(self.num_heads):
                        edge = (sender_layer, sender_head, receiver_layer, receiver_head)
                        delta_res = sender_deltas[(sender_layer, sender_head)] # shape is (batch, seq_len, d_model)
                        delta_qkv = torch.matmul(delta_res, weight)
                        delta_qkv = delta_qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
                        delta_receiver = delta_qkv[:, :, :, receiver_head, :]
                        attribution = (delta_receiver.float() * receiver_grad.float()).sum()
                        signed_edge_scores[edge] = attribution.item()
                        edge_scores[edge] = abs(attribution.item())

        return edge_scores, signed_edge_scores

    def get_stats(self, clean_prompt, corrupted_prompt, correct, incorrect):
        self.model.eval()
        correct_id = self.get_target_id(correct)
        incorrect_id = self.get_target_id(incorrect)

        clean_tokens = self.tokenize(clean_prompt)
        corrupted_tokens = self.tokenize(corrupted_prompt)

        assert clean_tokens['input_ids'].shape == corrupted_tokens['input_ids'].shape, "Clean and corrupted prompts must have the same shape."

        with torch.no_grad():
            clean_output = self.model(**clean_tokens, use_cache=False)
            clean_logit_diff = self.logit_diff(clean_output.logits, correct_id, incorrect_id).item()

        clean_inputs = self.capture_c_proj_inputs(clean_prompt)
        corrupted_inputs = self.capture_c_proj_inputs(corrupted_prompt)
        sender_deltas = self.compute_sender_deltas(clean_inputs, corrupted_inputs)
        corrupted_logit_diff, receiver_gradients = self.capture_receiver_gradients(corrupted_prompt, correct_id, incorrect_id)

        gap = clean_logit_diff - corrupted_logit_diff
        print(f"Clean logit difference: {clean_logit_diff:.6f}")
        print(f"Corrupted logit difference: {corrupted_logit_diff:.6f}")
        print(f"Gap: {gap:.6f}")


    def run(self, clean_prompt, corrupted_prompt, correct, incorrect):
        print("TOkenizing prompts...")
        correct_id = self.get_target_id(correct)
        incorrect_id = self.get_target_id(incorrect)
        clean_tokens = self.tokenize(clean_prompt)
        corrupted_tokens = self.tokenize(corrupted_prompt)

        if clean_tokens['input_ids'].shape != corrupted_tokens['input_ids'].shape:
            raise ValueError(f"Must have same size for clean and corrupted prompts. Got {clean_tokens['input_ids'].shape} and {corrupted_tokens['input_ids'].shape}")

        print("Capturing clean sender activations...")
        clean_inputs = self.capture_c_proj_inputs(clean_prompt)

        print("Capturing corrupted sender activations...")
        corrupted_inputs = self.capture_c_proj_inputs(corrupted_prompt)

        print("Capturing Sender residual deltas...")
        sender_deltas = self.compute_sender_deltas(clean_inputs, corrupted_inputs)

        print("Capturing receiver gradients...")
        corrupted_baseline, receiver_gradients = self.capture_receiver_gradients(corrupted_prompt, correct_id, incorrect_id)

        print("Computing edge scores...")
        edge_scores, signed_edge_scores = self.compute_edge_scores(sender_deltas, receiver_gradients)

        ranked_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        print("Total edges:", len(ranked_edges))

        return {
            "edge_scores": edge_scores,
            "signed_edge_scores": signed_edge_scores,
            "ranked_edges": ranked_edges,
            "corrupted_baseline": corrupted_baseline
        }

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Attribution Patching")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.checkpoint)
    config._attn_implementation = "eager"

    model = GPT2LMHeadModel.from_pretrained(args.checkpoint, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    model.eval()

    eap = EdgeAttributionPatching(model, tokenizer, device=device)
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

    all_signed_scores = defaultdict(list)
    all_absolute_scores = defaultdict(list)

    prompt_results = []

    for i,pair in enumerate(prompt_pairs):
        print(f"\nProcessing prompt pair {i+1}/{len(prompt_pairs)}")

        clean_prompt = pair["clean"]
        corrupted_prompt = pair["corrupted"]
        target_correct = pair["correct"]
        target_incorrect = pair["incorrect"]

        eap.get_stats(clean_prompt, corrupted_prompt, target_correct, target_incorrect)
        results = eap.run(clean_prompt, corrupted_prompt, target_correct, target_incorrect)

        signed_edge_scores = results["signed_edge_scores"]
        edge_scores = results["edge_scores"]

        for edge, score in signed_edge_scores.items():
            all_signed_scores[edge].append(score)
            all_absolute_scores[edge].append(edge_scores[edge])

        prompt_results.append(results)

    mean_signed_scores = {}
    mean_absolute_scores = {}

    for edge in all_signed_scores:
        mean_signed_scores[edge] = sum(all_signed_scores[edge]) / len(all_signed_scores[edge])
        mean_absolute_scores[edge] = sum(all_absolute_scores[edge]) / len(all_absolute_scores[edge])

    ranked_edges = sorted(mean_absolute_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nTotal edges:", len(ranked_edges))
    print("\n Top 20 edges by mean absolute score:")

    for edge, score in ranked_edges[:20]:
        sl, sh, rl, rh = edge
        print(
            f"L{sl}H{sh} -> L{rl}H{rh}: "
            f"Mean Abs = {score:.6f}, "
            f"Mean Signed = {mean_signed_scores[edge]:.6f}"
        )

        
    

