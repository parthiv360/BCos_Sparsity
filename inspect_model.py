from bcos_lm.gpt2 import GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os, argparse

from transformers import AutoConfig, AutoTokenizer

class ModelInspector:
    def __init__(self, checkpoint_path):
       self.checkpoint_path = checkpoint_path

       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       config = AutoConfig.from_pretrained(checkpoint_path)
       config._attn_implementation = "eager"

       self.model = GPT2LMHeadModel.load_from_pretrained(checkpoint_path, config=config)
       self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
       self.model.to(self.device)
       self.model.eval()

       print(f"Loaded model from {checkpoint_path} on device {self.device}")

    
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

        checkpoint_name = os.path.basename(self.checkpoint_path)
        os.makedirs(f"attention_maps/{checkpoint_name}", exist_ok=True)
        plt.savefig(f"attention_maps/{checkpoint_name}/attention_layer{layer}_head{head}.png")
        plt.close()
        print(f"Attention map saved to attention_maps/{checkpoint_name}/attention_layer{layer}_head{head}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=9,
        help="Transformer layer to visualize"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=9,
        help="Attention head to visualize"
    )

    args = parser.parse_args()

    prompt = "When John and Mary went to the store, John gave a bottle of milk to"

    inspector = ModelInspector(args.checkpoint)
    inspector.inspect_architecture()
    inspector.inspect_attention_heads(prompt)
    inspector.visualize_attention(prompt, layer=args.layer, head=args.head)

if __name__ == "__main__":
    main()