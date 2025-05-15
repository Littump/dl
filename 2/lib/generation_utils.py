import torch.nn.functional as F


def apply_temperature(logits, temperature):
    return logits / temperature


def normalize_logits(logits):
    return F.softmax(logits, dim=-1)
