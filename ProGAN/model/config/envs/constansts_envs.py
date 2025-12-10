"""Configuration constants and environment variables for ProGAN training."""
import os
import random
import numpy as np
import torch
from pathlib import Path

# ==================== CONFIG ====================
# Paths - Modify these according to your setup
# BASE_DIR apunta a ProGAN/ (subimos 4 niveles desde este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "OCT2017" / "train"
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "generated_images"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model Config
START_TRAIN_AT_IMG_SIZE = 512  # Set to 4 to start from scratch, 512 to continue
DATASET = 'oct'
CHECKPOINT_GEN = str(WEIGHTS_DIR / "generator_DRUSEN_v11.pth")
CHECKPOINT_CRITIC = str(WEIGHTS_DIR / "critic_DRUSEN_v11.pth")
CHECKPOINT_GEN_SAVE = str(WEIGHTS_DIR / "generator_DRUSEN_local.pth")
CHECKPOINT_CRITIC_SAVE = str(WEIGHTS_DIR / "critic_DRUSEN_local.pth")

# Device config
ngpu = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU Safety settings for RTX 3070
ENABLE_GPU_MONITORING = True
MAX_BATCH_SIZE_LIMIT = 16  # Limitar batch size máximo para evitar sobrecalentamiento

# Training config
SAVE_MODEL = True
LOAD_MODEL = False  # Set to True if you have pre-trained weights
LEARNING_RATE = 1e-3
# Batch sizes optimizados para RTX 3070 (8GB VRAM)
# Valores más conservadores para evitar sobrecalentamiento
BATCH_SIZES = [16, 16, 12, 8, 6, 4, 3, 2, 1]  # Reducidos para tu RTX 3070
CHANNELS_IMG = 3
Z_DIM = 256  # Latent dimension
IN_CHANNELS = 256
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES)  # Reducido de 30 a 20 épocas por resolución
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(device)
NUM_WORKERS = 2  # Ajustado para mejor rendimiento

# Gradient accumulation para simular batches más grandes sin usar tanta VRAM
GRADIENT_ACCUMULATION_STEPS = 2  # Acumular gradientes cada 2 pasos

# Guardado automático durante entrenamiento
AUTO_SAVE_EVERY_N_BATCHES = 500  # Guardar cada 500 batches (ajusta según necesidad)

# ==================== UTILS ====================

# Variables globales para guardado de emergencia
emergency_save_models = {}

