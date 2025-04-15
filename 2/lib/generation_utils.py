import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature):
    return logits / temperature


def get_top_p_tokens(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.clone()
    logits[indices_to_remove] = float('-inf')

    return logits


def normalize_logits(logits):
    return F.softmax(logits, dim=-1)
