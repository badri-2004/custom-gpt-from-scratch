import torch
import numpy as np
from config import device, temperature, top_k, start_char

def generate(vocab,model,encode,decode, max_length=200):
    input_tensor = torch.tensor([encode(start_char)]).to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_tensor)
            logits = logits[:, -1, :]

            logits /= temperature

            if top_k is not None and top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, -float('Inf'))
                logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = next_token.clamp(min=0, max=len(vocab) - 1)

            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    generated_text = decode(input_tensor.squeeze().cpu().numpy())
    return generated_text
