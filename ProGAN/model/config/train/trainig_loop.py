"""Training loop for ProGAN."""
import torch
from tqdm import tqdm

from ..envs.constansts_envs import (
    device, Z_DIM, LAMBDA_GP, GRADIENT_ACCUMULATION_STEPS,
    PROGRESSIVE_EPOCHS, FIXED_NOISE, AUTO_SAVE_EVERY_N_BATCHES,
    CHECKPOINT_GEN_SAVE, CHECKPOINT_CRITIC_SAVE, ENABLE_GPU_MONITORING
)
from ..checkpoints.save_checkpoint import save_checkpoint
from ..logs.gpu_logs import print_gpu_stats
from .gradient_penalty import gradient_penalty


def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    """Log training progress to TensorBoard."""
    writer.add_scalar("Loss/Critic", loss_critic, tensorboard_step)
    writer.add_scalar("Loss/Generator", loss_gen, tensorboard_step)
    writer.add_images("Real", real[:8], tensorboard_step)
    writer.add_images("Fake", fake[:8], tensorboard_step)


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

