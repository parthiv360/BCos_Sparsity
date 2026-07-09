import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Setup:
    """
    GPT-2 Setup class for initializing the model and tokenizer, and testing its predictions.
    """

    def __init__(self):
        """
        Initializes the GPT-2 model, tokenizer, and device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2",attn_implementation="eager" )
        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)

    def print_model_info(self):
        """
        Prints basic GPT-2 architecture information.
        """
        config = self.model.config

        print("\nGPT-2 Configuration")
        print("-" * 40)
        print(f"Number of layers      : {config.n_layer}")
        print(f"Attention heads/layer : {config.n_head}")
        print(f"Hidden size           : {config.n_embd}")
        print(f"Vocabulary size       : {config.vocab_size}")
        print(f"Context length        : {config.n_positions}")
        print(f"Model type            : {config.model_type}")

    def test_ioi_prediction(self, prompt: str):
        """
        Tests the model's ability to predict the next token given a prompt.

        Args:
            prompt (str): The input prompt for the model.
        """
        print(f"\nTesting prompt: '{prompt}'")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        top_k = 3
        top_probabilities = torch.softmax(last_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(top_probabilities, top_k)
        
        for i in range(top_k):
            token_str = self.tokenizer.decode(top_indices[i])
            print(f"Rank {i+1}: {token_str!r:10} | Probability: {top_probs[i].item():.4f}")
            
def main():
    """
    Main function to demonstrate the setup of the GPT-2 model and tokenizer.
    """
    gpt2_setup = GPT2Setup()
    print(f"Model and tokenizer are set up on device: {gpt2_setup.device}")
    gpt2_setup.print_model_info()

    sample_prompt = "When John and Mary went to the store, John gave a bottle of milk to"
    gpt2_setup.test_ioi_prediction(sample_prompt)

if __name__ == "__main__":
    main()