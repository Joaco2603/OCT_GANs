"""Load model checkpoints."""
import torch

from ..envs.constansts_envs import device


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Load model checkpoint"""
    print(f"=> Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr