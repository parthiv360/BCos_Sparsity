from setup_model import GPT2Setup
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelInspector:
    def __init__(self):
        self.gpt2_setup = GPT2Setup()
        self.model = self.gpt2_setup.model
        self.tokenizer = self.gpt2_setup.tokenizer
        self.device = self.gpt2_setup.device
    
    def inspect_architecture(self):
        """
        Prints the core transformer components of the GPT-2 model.(only 1 layer)
        """
        print("--- Core Transformer Components ---")
        for name, module in self.model.named_modules():
            if any(target in name for target in ['wte', 'wpe', 'h.0', 'ln_f', 'lm_head']):
                depth = name.count('.')
                indent = "  " * depth
                print(f"{indent}[{name}] -> {type(module).__name__}")

    def inspect_attention_heads(self,prompt):
        """
        Visualizes the attention heads of the GPT-2 model for a given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        for layer_idx, layer_attention in enumerate(attentions):
            print(f"Layer {layer_idx}: Attention shape {layer_attention.shape}")
        
    
    def visualize_attention(self, prompt, layer=0, head=0):
        """
        Visualizes the attention map for a specific layer and head given a prompt.
        """

        print(f"\n--- Visualizing Attention (Layer {layer}, Head {head}) ---")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Inputs shape: (batch_size, seq_len)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        #  Layer attention shape: (batch_size, num_heads, seq_len, seq_len)
        layer_attention = attentions[layer][0, head].cpu().numpy()
        token_ids = inputs.input_ids[0].cpu().numpy()
        tokens = [self.tokenizer.decode([t]) for t in token_ids]

        os.makedirs("attention_maps", exist_ok=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            layer_attention, 
            xticklabels=tokens, 
            yticklabels=tokens, 
            cmap="viridis",
            square=True,
            cbar_kws={"shrink": .8}
        )
        plt.title(f"Attention Map - Layer {layer}, Head {head}")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(f"attention_maps/attention_layer{layer}_head{head}.png")
        plt.close()
        print(f"Attention map saved to attention_maps/attention_layer{layer}_head{head}.png")

def main():
    prompt = "When John and Mary went to the store, John gave a bottle of milk to"
    inspector = ModelInspector()
    inspector.inspect_architecture()
    inspector.inspect_attention_heads(prompt)
    inspector.analyze_weight_distribution()
    inspector.visualize_attention(prompt,layer=9,head=9)

if __name__ == "__main__":
    main()