import torch

def get_logit_diff(model, tokenizer, prompt, correct, incorrect,device):
        """
        Computes the logit difference between the correct and incorrect tokens for a given prompt.
        """

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        correct_token = tokenizer(" " + correct, return_tensors='pt').input_ids.to(device)
        incorrect_token = tokenizer(" " + incorrect, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
        
        return logits[0, correct_token[0, 0]].item() - logits[0, incorrect_token[0, 0]].item()