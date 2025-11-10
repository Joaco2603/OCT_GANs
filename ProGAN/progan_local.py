"""
ProGAN for OCT DRUSEN Image Generation - Local Version
Adapted from the original Colab notebook for local execution
"""

import random
import numpy as np
import os
import shutil
from pathlib import Path
import cv2
from math import log2
from scipy.stats import truncnorm
import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from tqdm import tqdm

# ==================== CONFIG ====================
# Paths - Modify these according to your setup
BASE_DIR = Path(__file__).parent
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

def setup_emergency_save(gen, critic, opt_gen, opt_critic):
    """Configurar guardado de emergencia en caso de Ctrl+C"""
    global emergency_save_models
    emergency_save_models = {
        'gen': gen,
        'critic': critic,
        'opt_gen': opt_gen,
        'opt_critic': opt_critic
    }
    
    def signal_handler(sig, frame):
        print("\n\n" + "=" * 60)
        print("🛑 ¡Ctrl+C detectado! Guardando pesos antes de salir...")
        print("=" * 60)
        
        try:
            # Guardar con timestamp para no sobrescribir
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_gen = str(WEIGHTS_DIR / f"EMERGENCY_generator_{timestamp}.pth")
            emergency_critic = str(WEIGHTS_DIR / f"EMERGENCY_critic_{timestamp}.pth")
            
            save_checkpoint(emergency_save_models['gen'], 
                          emergency_save_models['opt_gen'], 
                          emergency_gen)
            save_checkpoint(emergency_save_models['critic'], 
                          emergency_save_models['opt_critic'], 
                          emergency_critic)
            
            print(f"\n✅ Pesos guardados exitosamente:")
            print(f"   📁 {emergency_gen}")
            print(f"   📁 {emergency_critic}")
            print("\n💡 Puedes reanudar el entrenamiento cargando estos pesos")
            print("   Cambia CHECKPOINT_GEN y CHECKPOINT_CRITIC en el código")
            print("=" * 60)
        except Exception as e:
            print(f"\n❌ Error al guardar: {e}")
            print("⚠️  Los pesos NO se pudieron guardar")
        
        print("\n👋 Saliendo...")
        sys.exit(0)
    
    # Registrar handler para Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    print("✅ Sistema de guardado de emergencia activado (Ctrl+C guardará pesos)")


def get_gpu_stats():
    """Obtener estadísticas de la GPU si está disponible"""
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Intentar obtener temperatura (requiere nvidia-smi)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,power.draw', 
                     '--format=csv,noheader,nounits', '--id=0'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    temp, util, power = result.stdout.strip().split(',')
                    return {
                        'memory_allocated': f"{gpu_memory_allocated:.2f}GB",
                        'memory_total': f"{gpu_memory_total:.2f}GB",
                        'memory_percent': f"{(gpu_memory_allocated/gpu_memory_total)*100:.1f}%",
                        'temperature': f"{temp}°C",
                        'utilization': f"{util}%",
                        'power': f"{power}W"
                    }
            except:
                pass
            
            return {
                'memory_allocated': f"{gpu_memory_allocated:.2f}GB",
                'memory_total': f"{gpu_memory_total:.2f}GB",
                'memory_percent': f"{(gpu_memory_allocated/gpu_memory_total)*100:.1f}%"
            }
        except:
            return None
    return None


def print_gpu_stats():
    """Imprimir estadísticas de GPU de forma legible"""
    stats = get_gpu_stats()
    if stats:
        print("\n" + "=" * 50)
        print("📊 Estadísticas de GPU:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("=" * 50)


def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    """Plot losses and images to tensorboard"""
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True, nrow=4)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True, nrow=4)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fixed fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """Calculate gradient penalty for WGAN-GP"""
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename):
    """Save model checkpoint"""
    print(f"=> Saving checkpoint: {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Load model checkpoint"""
    print(f"=> Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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


# ==================== MODEL LAYERS ====================

class WSConv2d(nn.Module):
    """
    Weight Scaled Conv2d (Equalized Learning Rate)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """Pixel normalization"""
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """Convolutional block with optional pixel normalization"""
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


# ==================== GENERATOR ====================

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class Generator(nn.Module):
    """Progressive GAN Generator"""
    def __init__(self, ngpu, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        # Initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        """Fade in new layers during progressive training"""
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


# ==================== DISCRIMINATOR (CRITIC) ====================

class Discriminator(nn.Module):
    """Progressive GAN Discriminator/Critic"""
    def __init__(self, ngpu, z_dim, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # Create progressive blocks
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # For 4x4 resolution
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final block
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """Fade in new layers during progressive training"""
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """Minibatch standard deviation"""
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps

        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


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


# ==================== TRAINING LOOP ====================

def train_fn(
    critic, gen, loader, dataset, step, alpha,
    opt_critic, opt_gen, tensorboard_step, writer,
    scaler_gen, scaler_critic
):
    """Training function for one epoch"""
    loop = tqdm(loader, leave=True)
    
    # Inicializar contadores para gradient accumulation
    accumulated_loss_critic = 0
    accumulated_loss_gen = 0
    accumulation_counter = 0
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )
            # Normalizar por gradient accumulation
            loss_critic = loss_critic / GRADIENT_ACCUMULATION_STEPS

        scaler_critic.scale(loss_critic).backward()
        accumulated_loss_critic += loss_critic.item()
        
        # Actualizar cada GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler_critic.step(opt_critic)
            scaler_critic.update()
            opt_critic.zero_grad()

        # Train Generator
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)
            loss_gen = loss_gen / GRADIENT_ACCUMULATION_STEPS

        scaler_gen.scale(loss_gen).backward()
        accumulated_loss_gen += loss_gen.item()
        
        # Actualizar cada GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler_gen.step(opt_gen)
            scaler_gen.update()
            opt_gen.zero_grad()

        # Update alpha
        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        # 💾 GUARDADO AUTOMÁTICO cada N batches
        if batch_idx > 0 and batch_idx % AUTO_SAVE_EVERY_N_BATCHES == 0:
            print(f"\n💾 Auto-guardando checkpoint en batch {batch_idx}...")
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN_SAVE)
            save_checkpoint(critic, opt_critic, filename=CHECKPOINT_CRITIC_SAVE)
            print(f"✅ Checkpoint guardado")

        # Log to tensorboard y mostrar stats de GPU
        if batch_idx % 50 == 0:
            with torch.no_grad():
                fixed_fakes = gen(FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                accumulated_loss_critic * GRADIENT_ACCUMULATION_STEPS,
                accumulated_loss_gen * GRADIENT_ACCUMULATION_STEPS,
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1
            
            # Mostrar estadísticas de GPU cada 50 batches
            if ENABLE_GPU_MONITORING and batch_idx % 100 == 0:
                print_gpu_stats()
            
            # Reset accumulated losses
            accumulated_loss_critic = 0
            accumulated_loss_gen = 0

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=(accumulated_loss_critic * GRADIENT_ACCUMULATION_STEPS),
            alpha=f"{alpha:.3f}",
        )

    return tensorboard_step, alpha


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
