"""ProGAN Model Package - Main training entry point."""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from math import log2
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import truncnorm
from torchvision.utils import save_image

# Import configurations
from .config.envs.constansts_envs import (
    device, ngpu, DATA_DIR, WEIGHTS_DIR, LOGS_DIR, OUTPUT_DIR,
    START_TRAIN_AT_IMG_SIZE, BATCH_SIZES, PROGRESSIVE_EPOCHS,
    Z_DIM, IN_CHANNELS, CHANNELS_IMG, LEARNING_RATE,
    SAVE_MODEL, LOAD_MODEL, CHECKPOINT_GEN, CHECKPOINT_CRITIC,
    CHECKPOINT_GEN_SAVE, CHECKPOINT_CRITIC_SAVE,
    ENABLE_GPU_MONITORING, GRADIENT_ACCUMULATION_STEPS
)

# Import model components
from .generator import Generator
from .discriminator import Discriminator

# Import utilities
from .config.checkpoints.save_checkpoint import save_checkpoint
from .config.checkpoints.load_checkpoint import load_checkpoint
from .config.logs.gpu_logs import setup_emergency_save, print_gpu_stats, get_gpu_stats
from .config.train.data_loader import get_loader
from .config.train.trainig_loop import train_fn


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


# ==================== MAIN TRAINING ====================

def main():
    """Main training loop"""
    print("=" * 50)
    print("ProGAN Training - Local Version")
    print("=" * 50)
    print(f"Device: {device}")
    
    # Verificar y mostrar información de GPU
    if torch.cuda.is_available():
        print(f"🎮 GPU Detectada: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"⚙️  CUDA Version: {torch.version.cuda}")
        print_gpu_stats()
    else:
        print("⚠️  GPU no detectada - usando CPU")
        print("\n🔧 Para usar tu RTX 3070, instala PyTorch con CUDA:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    print(f"\n📁 Data directory: {DATA_DIR}")
    print(f"🖼️  Starting image size: {START_TRAIN_AT_IMG_SIZE}")
    print(f"📦 Batch sizes optimizados para RTX 3070: {BATCH_SIZES}")
    print(f"🔄 Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    print("=" * 50)
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\n⚠️  WARNING: Data directory not found!")
        print(f"Expected path: {DATA_DIR}")
        print("\nPlease ensure you have the OCT DRUSEN dataset in the correct location.")
        print("The directory structure should be:")
        print("  ProGAN/")
        print("    └── data/")
        print("        └── OCT2017/")
        print("            └── train/")
        print("                └── DRUSEN/")
        print("                    └── [your images here]")
        return
    
    # Initialize models
    gen = Generator(ngpu, Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)
    critic = Discriminator(ngpu, Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        gen = nn.DataParallel(gen, list(range(ngpu)))
        critic = nn.DataParallel(critic, list(range(ngpu)))

    # Initialize optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.amp.GradScaler('cuda')
    scaler_gen = torch.amp.GradScaler('cuda')

    # ⚠️ Configurar guardado de emergencia (Ctrl+C)
    setup_emergency_save(gen, critic, opt_gen, opt_critic)

    # Tensorboard writer
    writer = SummaryWriter(str(LOGS_DIR))

    # Load checkpoint if requested
    if LOAD_MODEL:
        if Path(CHECKPOINT_GEN).exists():
            load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
            load_checkpoint(CHECKPOINT_CRITIC, critic, opt_critic, LEARNING_RATE)
        else:
            print(f"⚠️  Checkpoint not found: {CHECKPOINT_GEN}")
            print("Starting training from scratch...")

    # Set to training mode
    gen.train()
    critic.train()

    tensorboard_step = 0
    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    
    print(f"\nStarting training from step {step} (image size: {START_TRAIN_AT_IMG_SIZE})")
    
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4 * 2 ** step)
        current_img_size = 4 * 2 ** step
        
        print(f"\n{'='*50}")
        print(f"Training at resolution: {current_img_size}x{current_img_size}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {BATCH_SIZES[step]}")
        print(f"{'='*50}")

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Mostrar stats de GPU al inicio de cada época
            if ENABLE_GPU_MONITORING and torch.cuda.is_available():
                print_gpu_stats()
            
            tensorboard_step, alpha = train_fn(
                critic, gen, loader, dataset, step, alpha,
                opt_critic, opt_gen, tensorboard_step, writer,
                scaler_gen, scaler_critic,
            )

            if SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN_SAVE)
                save_checkpoint(critic, opt_critic, filename=CHECKPOINT_CRITIC_SAVE)
            
            # Limpiar caché de CUDA para evitar acumulación de memoria
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        step += 1

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    writer.close()


# ==================== GENERATION MODE ====================

def generation_mode():
    """Load trained model and generate images"""
    print("=" * 50)
    print("ProGAN Generation Mode")
    print("=" * 50)
    
    if not Path(CHECKPOINT_GEN).exists():
        print(f"⚠️  Generator checkpoint not found: {CHECKPOINT_GEN}")
        return
    
    # Load generator
    gen = Generator(ngpu, Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    
    # Generate images at 256x256 resolution
    steps = int(log2(256 / 4))
    generate_examples(gen, steps, truncation=0.99, n=20)
    
    print("Generation completed!")


if __name__ == "__main__":
    import sys
    
    seed_everything(42)
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generation_mode()
    else:
        main()