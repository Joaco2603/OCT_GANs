"""Save model checkpoints."""
import torch


def save_checkpoint(model, optimizer, filename):
    """Save model checkpoint"""
    print(f"=> Saving checkpoint: {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
