"""Data loader for ProGAN training."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from math import log2

from ..envs.constansts_envs import DATA_DIR, BATCH_SIZES, CHANNELS_IMG, NUM_WORKERS

# ==================== DATA LOADER ====================

torch.backends.cudnn.benchmark = True


def get_loader(image_size):
    """Get data loader for specified image size"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
        ),
    ])
    
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset