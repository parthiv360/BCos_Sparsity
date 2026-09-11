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

def get_logit_diff_with_grad(logits, tokenizer, correct, incorrect, device):
    """
    Computes the logit difference between the correct and incorrect tokens for a given prompt.
    This function is used for attribution patching.
    """

    correct_token = tokenizer(" " + correct, return_tensors='pt').input_ids.to(device)
    incorrect_token = tokenizer(" " + incorrect, return_tensors='pt').input_ids.to(device)

    last_token_logits = logits[0, -1, :]

    return last_token_logits[correct_token[0, 0]] - last_token_logits[incorrect_token[0, 0]]