import torch

def cross_correct_logits(logits_A, noise_B, alpha=0.7):
    # subtract other model's noise; block gradient through that noise
    return logits_A - alpha * noise_B.detach()

def refined_pseudo_from(logits, dim=1):
    probs = torch.softmax(logits, dim=dim)
    return torch.argmax(probs, dim=dim, keepdim=True).long()