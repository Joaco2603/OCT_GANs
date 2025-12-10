# Architecture Guide - OCT_GANs Repository

## System Overview

OCT_GANs is a **Progressive GAN implementation** specifically designed for generating synthetic OCT (Optical Coherence Tomography) retinal images with DRUSEN pathology. The system progressively trains from 4×4 to 512×512 resolution.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OCT_GANs System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │  Data Layer  │─────▶│ Training Core│                   │
│  └──────────────┘      └──────────────┘                   │
│         │                      │                           │
│         │                      ▼                           │
│         │              ┌──────────────┐                   │
│         │              │   ProGAN     │                   │
│         │              │   Model      │                   │
│         │              └──────────────┘                   │
│         │                      │                           │
│         │                      ▼                           │
│         │              ┌──────────────┐                   │
│         └─────────────▶│  Monitoring  │                   │
│                        │  & Logging   │                   │
│                        └──────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Layer (`ProGAN/data/`)

**Purpose**: Load and preprocess OCT images for training

**Structure**:
```
data/OCT2017/train/
  └── DRUSEN/
      ├── DRUSEN-001.jpeg
      ├── DRUSEN-002.jpeg
      └── ...
```

**Key Classes**:
- `torch.utils.data.DataLoader`: Batched data loading
- `torchvision.datasets.ImageFolder`: Directory-based dataset

**Transforms Pipeline**:
```python
transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1] range
])
```

**Data Flow**:
```
Image Files → ImageFolder → Transform → Batches → GPU
   (JPEG)      (Dataset)    (Tensor)   (Variable)  (Training)
```

### 2. ProGAN Model (`ProGAN/model/progan_local.py`)

#### Generator Architecture

**Purpose**: Generate realistic OCT images from random noise

```python
Input: Z ~ N(0,1), shape (batch, 256, 1, 1)
   ↓
InitialBlock: 256 → 256 (4×4)
   ↓
ProgressiveBlocks: 
   4×4 → 8×8 (256 → 256)
   8×8 → 16×16 (256 → 256)
   16×16 → 32×32 (256 → 256)
   32×32 → 64×64 (256 → 128)
   64×64 → 128×128 (128 → 64)
   128×128 → 256×256 (64 → 32)
   256×256 → 512×512 (32 → 16)
   ↓
ToRGB: channels → 3 (RGB image)
   ↓
Output: Fake OCT image (batch, 3, 512, 512)
```

**Key Techniques**:
- **PixelNorm**: Normalize feature vectors (prevents exploding values)
- **Equalized Learning Rate**: `w = w / sqrt(fan_in)` at runtime
- **Progressive Growing**: Smoothly fade in new layers using `alpha` parameter

#### Discriminator (Critic) Architecture

**Purpose**: Distinguish real vs. generated OCT images

```python
Input: OCT image (batch, 3, resolution, resolution)
   ↓
FromRGB: 3 → channels
   ↓
ProgressiveBlocks (mirror of Generator):
   512×512 → 256×256 (16 → 32)
   256×256 → 128×128 (32 → 64)
   128×128 → 64×64 (64 → 128)
   64×64 → 32×32 (128 → 256)
   32×32 → 16×16 (256 → 256)
   16×16 → 8×8 (256 → 256)
   8×8 → 4×4 (256 → 256)
   ↓
MinibatchStdDev: Add diversity signal
   ↓
FinalBlock: 256+1 → 1 (scalar score)
   ↓
Output: Realness score (batch, 1, 1, 1)
```

**Key Techniques**:
- **Minibatch Standard Deviation**: Encourages diversity
- **Gradient Penalty (WGAN-GP)**: Stabilizes training (λ=10)

### 3. Training System

#### Progressive Training Schedule

```python
Phase 1: 4×4   (20 epochs, batch_size=16)
Phase 2: 8×8   (20 epochs, batch_size=16)
Phase 3: 16×16 (20 epochs, batch_size=12)
Phase 4: 32×32 (20 epochs, batch_size=8)
Phase 5: 64×64 (20 epochs, batch_size=6)
Phase 6: 128×128 (20 epochs, batch_size=4)
Phase 7: 256×256 (20 epochs, batch_size=3)
Phase 8: 512×512 (20 epochs, batch_size=2)
```

**Alpha Fading**:
```python
alpha = current_step / (total_steps / 2)
alpha = min(alpha, 1.0)  # Clamp to [0, 1]

# Used in model:
output = alpha * new_output + (1 - alpha) * old_output
```

#### Training Loop (Per Batch)

```
1. Train Critic (Discriminator):
   for _ in range(CRITIC_ITERATIONS):
       real_images → Critic → real_score
       noise → Generator → fake_images
       fake_images → Critic → fake_score
       
       loss_critic = -(mean(real_score) - mean(fake_score))
       gradient_penalty = compute_gp(real, fake, critic)
       loss_critic += LAMBDA_GP * gradient_penalty
       
       loss_critic.backward()
       optimizer_critic.step()

2. Train Generator:
   noise → Generator → fake_images
   fake_images → Critic → fake_score
   
   loss_gen = -mean(fake_score)  # Maximize critic score
   
   loss_gen.backward()
   optimizer_gen.step()
```

#### Memory Optimization

**Mixed Precision Training**:
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    # Forward pass in float16
    output = model(input)
    loss = criterion(output, target)

# Backward pass with scaled gradients
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Gradient Accumulation**:
```python
GRADIENT_ACCUMULATION_STEPS = 2

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / GRADIENT_ACCUMULATION_STEPS
    loss.backward()
    
    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Monitoring & Logging

#### TensorBoard Integration

```python
writer = SummaryWriter(log_dir="ProGAN/logs")

# Training metrics
writer.add_scalar("Loss/Generator", loss_gen, global_step)
writer.add_scalar("Loss/Critic", loss_critic, global_step)
writer.add_scalar("Gradient_Penalty", gp, global_step)

# Generated images
writer.add_images("Generated", fake_images, global_step)

# GPU metrics
writer.add_scalar("GPU/Memory_GB", memory_allocated, global_step)
writer.add_scalar("GPU/Temperature_C", temp, global_step)
```

#### GPU Monitoring

```python
def get_gpu_stats():
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    # Query nvidia-smi
    temp, util, power = query_nvidia_smi()
    
    return {
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
        'temperature_c': temp,
        'utilization_pct': util,
        'power_draw_w': power
    }
```

### 5. Checkpoint System

#### Save Format

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'resolution': current_resolution,
    'alpha': alpha
}
torch.save(checkpoint, path)
```

#### Emergency Save

```python
# Triggered by SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_checkpoint(gen, opt_gen, 
                   f"weights/EMERGENCY_generator_{timestamp}.pth")
    save_checkpoint(critic, opt_critic,
                   f"weights/EMERGENCY_critic_{timestamp}.pth")
    sys.exit(0)
```

## Design Patterns

### 1. Progressive Growth Pattern

**Problem**: Training high-resolution GANs is unstable and slow

**Solution**: Start from low resolution, progressively add layers

```python
class ProgressiveGenerator(nn.Module):
    def forward(self, z, alpha, steps):
        out = self.initial(z)  # 4×4
        
        for i in range(steps):
            out = self.blocks[i](out)  # Upsample + Conv
        
        # Fade between resolutions
        if alpha < 1 and steps > 0:
            old_out = self.to_rgb[steps-1](out_old)
            new_out = self.to_rgb[steps](out)
            return alpha * new_out + (1 - alpha) * old_out
        else:
            return self.to_rgb[steps](out)
```

### 2. Wasserstein GAN with Gradient Penalty

**Problem**: Mode collapse, training instability

**Solution**: Use Wasserstein distance with gradient penalty

```python
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
```

### 3. Configuration Management

**Centralized Config** (Top of `progan_local.py`):

```python
# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "OCT2017" / "train"
WEIGHTS_DIR = BASE_DIR / "weights"

# Training
START_TRAIN_AT_IMG_SIZE = 512
BATCH_SIZES = [16, 16, 12, 8, 6, 4, 3, 2, 1]
PROGRESSIVE_EPOCHS = [20] * 9
LEARNING_RATE = 1e-3
GRADIENT_ACCUMULATION_STEPS = 2

# Model
Z_DIM = 256
IN_CHANNELS = 256
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
```

## Data Flow Diagrams

### Training Flow

```
Start Training
     ↓
Load Dataset → ImageFolder(DATA_DIR)
     ↓
For each resolution (4→512):
     ↓
  ┌─────────────────────┐
  │ Progressive Phase   │
  │ (e.g., 128×128)     │
  └─────────────────────┘
     ↓
  Create DataLoader(batch_size=BATCH_SIZES[step])
     ↓
  For each epoch (20):
     ↓
    ┌──────────────────┐
    │ For each batch:  │
    │                  │
    │ 1. Train Critic  │
    │ 2. Train Gen     │
    │ 3. Log metrics   │
    │ 4. Save images   │
    └──────────────────┘
     ↓
  Increment alpha (fade in new layers)
     ↓
  Save checkpoint
     ↓
Next resolution
     ↓
Training Complete
```

### Generation Flow

```
Load Generator Checkpoint
     ↓
Set model to eval mode
     ↓
Sample noise: Z ~ N(0,1)  [batch, 256, 1, 1]
     ↓
Generator(Z, alpha=1, steps=max)
     ↓
Generated Image [batch, 3, 512, 512]
     ↓
Denormalize: [-1,1] → [0,255]
     ↓
Save as JPEG/PNG
```

## Scalability & Performance

### Memory Constraints (RTX 3070 - 8GB VRAM)

| Resolution | Batch Size | Memory Usage | Training Time/Epoch |
|-----------|-----------|--------------|---------------------|
| 4×4       | 16        | ~2 GB        | 2 min               |
| 8×8       | 16        | ~2.5 GB      | 3 min               |
| 16×16     | 12        | ~3 GB        | 5 min               |
| 32×32     | 8         | ~3.5 GB      | 8 min               |
| 64×64     | 6         | ~4 GB        | 12 min              |
| 128×128   | 4         | ~5 GB        | 20 min              |
| 256×256   | 3         | ~6 GB        | 35 min              |
| 512×512   | 2         | ~7.5 GB      | 60 min              |

### Optimization Strategies

1. **Gradient Accumulation**: Simulate batch_size=4 with batch_size=2 + 2 accumulation steps
2. **Mixed Precision**: Use FP16 for forward pass, FP32 for critical operations
3. **Efficient Data Loading**: `num_workers=2`, `pin_memory=True`
4. **Checkpoint Offloading**: Save to disk, not GPU memory

## Error Handling

### Common Failure Modes

1. **CUDA Out of Memory**
   - **Detection**: `torch.cuda.OutOfMemoryError`
   - **Recovery**: Reduce batch size, enable gradient checkpointing

2. **Mode Collapse**
   - **Detection**: All generated images look identical
   - **Recovery**: Increase CRITIC_ITERATIONS, adjust LAMBDA_GP

3. **Training Instability**
   - **Detection**: Loss oscillates wildly
   - **Recovery**: Reduce learning rate, check gradient penalty

### Safeguards

```python
# Emergency save on Ctrl+C
setup_emergency_save(gen, critic, opt_gen, opt_critic)

# Auto-save every N batches
if batch_idx % AUTO_SAVE_EVERY_N_BATCHES == 0:
    save_checkpoint(...)

# GPU temperature monitoring
if temp > 85:
    print("WARNING: GPU temperature high, consider reducing batch size")
```

## Testing Strategy

### Unit Tests (Planned)
- Model architecture validation
- Checkpoint save/load consistency
- Data loader correctness

### Integration Tests
- End-to-end training for 1 epoch
- Generation from loaded checkpoint
- TensorBoard logging

### Smoke Tests
```powershell
# Quick sanity check
python ProGAN/model/progan_local.py --epochs 1 --start-res 4
```

## Future Architecture Improvements

### Modularization
- Extract Generator to `ProGAN/model/generator.py`
- Extract Discriminator to `ProGAN/model/discriminator.py`
- Create `ProGAN/training/trainer.py` for training loop

### Features
- Multi-GPU support with `DistributedDataParallel`
- StyleGAN-inspired style mixing
- Conditional generation (specify pathology type)
- FID score evaluation

---

**Related Documents**:
- `copilot-instructions.md` - Main development guide
- `copilot-tests.md` - Testing procedures
- `copilot-documentation.md` - Documentation standards

**Last Updated**: November 12, 2025