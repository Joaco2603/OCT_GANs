"""Generate example images using trained ProGAN generator."""
import os
import random
import numpy as np
import torch
from scipy.stats import truncnorm
from torchvision.utils import save_image

from model.config.envs.constansts_envs import device, Z_DIM, OUTPUT_DIR


def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_examples(gen, steps, truncation=0.7, n=100):
    """Generate example images using the trained generator"""
    gen.eval()
    alpha = 1.0
    output_path = OUTPUT_DIR / "saved_examples"
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating {n} examples...")
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(
                truncnorm.rvs(-truncation, truncation, size=(1, Z_DIM, 1, 1)),
                device=device,
                dtype=torch.float32
            )
            img = gen(noise, alpha, steps)
            save_image(img * 0.5 + 0.5, output_path / f"img_{i}.png")
    print(f"Images saved to: {output_path}")
    gen.train()