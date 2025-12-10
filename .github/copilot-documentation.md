# Documentation Standards - OCT_GANs Repository

## Documentation Philosophy

Documentation in OCT_GANs serves three audiences:
1. **Developers** - Internal architecture and code patterns
2. **Researchers** - Model theory and experimental setup
3. **Users** - Installation, training, and generation workflows

All documentation should be:
- **Accurate** - Reflects current codebase state
- **Actionable** - Provides runnable examples
- **Discoverable** - Easy to find via README or file structure
- **Bilingual** - English (primary) and Spanish (secondary)

## Documentation Structure

```
OCT_GANs/
├── README.md                    # Entry point for all users
├── LICENSE                      # Legal terms
├── .github/
│   ├── copilot-instructions.md # AI assistant guide (this file's siblings)
│   ├── copilot-arquitecture.md # System design
│   ├── copilot-tests.md        # Testing strategy
│   ├── copilot-documentation.md # (this file)
│   ├── copilot-contributing.md # Contribution workflow
│   └── copilot-guardrails.md   # Security & compliance
└── ProGAN/
    ├── docs/
    │   ├── en/                 # English documentation
    │   │   ├── CONFIGURATION_GPU.md
    │   │   ├── INSTRUCTIONS_DATASET.md
    │   │   ├── TRAINING_GUIDE.md
    │   │   └── API_REFERENCE.md
    │   └── es/                 # Spanish documentation
    │       ├── CONFIGURACION_GPU.md
    │       ├── INSTRUCCIONES_DATASET.md
    │       ├── GUIA_ENTRENAMIENTO.md
    │       └── REFERENCIA_API.md
    └── model/
        └── progan_local.py     # Inline docstrings
```

## Documentation Types

### 1. README (Repository Root)

**Purpose**: First impression and quick start guide

**Required Sections**:
```markdown
# Project Title

[Badges: Build Status, License, Python Version]

## Quick Summary
- One-sentence project description
- Key features (3-5 bullet points)
- Visual examples (generated images)

## Quick Start
- Environment setup (5-10 commands)
- Minimal training example
- Generation example

## Project Structure
- Directory layout with descriptions

## Installation
- Detailed setup for Windows/Linux
- GPU requirements
- Dependency installation

## Usage
- Training workflow
- Generation workflow
- Monitoring with TensorBoard

## Configuration
- Key hyperparameters
- Hardware recommendations

## Documentation
- Links to detailed guides

## Contributing
- Link to CONTRIBUTING.md

## Citation
- BibTeX entry for academic use

## License
- License type and link

## Contact
- How to get help
```

**Style Guide**:
- Use present tense ("generates images", not "will generate")
- Provide copy-pasteable commands in code blocks
- Include expected output for verification steps
- Use emojis sparingly (✅ for success, ⚠️ for warnings)

### 2. Detailed Guides (`ProGAN/docs/en/`)

#### CONFIGURATION_GPU.md

**Purpose**: GPU setup and optimization

**Template**:
```markdown
# GPU Configuration Guide

## Prerequisites
- NVIDIA GPU with Compute Capability 7.0+ (RTX 2000 series or newer)
- 8GB+ VRAM (RTX 3070 recommended)
- CUDA 11.8 or 12.1
- cuDNN 8.x

## Installation Steps

### Windows
```powershell
# Step 1: Verify GPU
nvidia-smi

# Step 2: Install PyTorch with CUDA
cd ProGAN
.\scripts\setup_cuda.ps1
```

### Linux
```bash
# ...
```

## Verification
```powershell
python ProGAN/config/check_gpu.py
```

Expected output:
```
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 3070
...
```

## Performance Tuning
- Batch size recommendations by resolution
- Memory optimization techniques
- Temperature management

## Troubleshooting
- Common CUDA errors and fixes
- Memory overflow solutions
```

#### INSTRUCTIONS_DATASET.md

**Purpose**: Dataset acquisition and preparation

**Sections**:
1. **Dataset Overview** - OCT2017 source, size, classes
2. **Download Methods** - Kaggle API, manual download
3. **Directory Structure** - Expected layout
4. **Verification** - Check image count and format
5. **Custom Datasets** - How to use your own data

#### TRAINING_GUIDE.md (New)

**Purpose**: Comprehensive training walkthrough

**Sections**:
1. **Pre-Training Checklist**
2. **Configuration Options** - Hyperparameter explanations
3. **Training Process** - What happens at each resolution
4. **Monitoring** - TensorBoard metrics interpretation
5. **Checkpointing** - Save/load strategies
6. **Troubleshooting** - Common training issues

#### API_REFERENCE.md (New)

**Purpose**: Programmatic interface documentation

**Format**:
```markdown
# API Reference

## Core Classes

### Generator

```python
class Generator(nn.Module):
    """
    Progressive GAN Generator for OCT image synthesis.
    
    Generates images from random noise using progressive growing,
    starting at 4×4 and scaling to 512×512 resolution.
    
    Args:
        z_dim (int): Latent vector dimension. Default: 256
        in_channels (int): Initial channel count. Default: 256
        img_channels (int): Output image channels (RGB=3). Default: 3
    
    Example:
        >>> gen = Generator(z_dim=256, in_channels=256, img_channels=3)
        >>> z = torch.randn(16, 256, 1, 1)
        >>> fake_images = gen(z, alpha=1.0, steps=6)  # 256×256
        >>> fake_images.shape
        torch.Size([16, 3, 256, 256])
    
    Methods:
        forward(z, alpha, steps): Generate images
        fade_in(old, new, alpha): Blend resolutions
    """
```

### Training Functions

```python
def train_progressive_gan(
    start_resolution: int = 4,
    epochs_per_resolution: List[int] = [20] * 9,
    batch_sizes: List[int] = [16, 16, 12, 8, 6, 4, 3, 2, 1],
    learning_rate: float = 1e-3,
    checkpoint_gen: Optional[str] = None,
    checkpoint_critic: Optional[str] = None,
    save_interval: int = 500
) -> Tuple[Generator, Discriminator]:
    """
    Train ProGAN with progressive resolution growth.
    
    Args:
        start_resolution: Initial image size (4, 8, 16, ..., 512)
        epochs_per_resolution: Training epochs at each resolution
        batch_sizes: Batch size for each resolution step
        learning_rate: Adam optimizer learning rate
        checkpoint_gen: Path to pretrained generator (optional)
        checkpoint_critic: Path to pretrained critic (optional)
        save_interval: Save checkpoint every N batches
    
    Returns:
        Trained generator and discriminator models
    
    Raises:
        FileNotFoundError: If dataset not found
        RuntimeError: If CUDA out of memory
    
    Example:
        >>> gen, critic = train_progressive_gan(
        ...     start_resolution=128,
        ...     epochs_per_resolution=[10, 10, 10],
        ...     learning_rate=5e-4
        ... )
    """
```
```

### 3. Inline Documentation (Code)

#### Docstring Standards

Use Google-style docstrings:

```python
def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Calculate Wasserstein GAN gradient penalty for training stability.
    
    Computes the penalty term to enforce Lipschitz constraint on the critic.
    Interpolates between real and fake images, then measures gradient norm.
    
    Args:
        critic (Discriminator): Critic network to evaluate
        real (torch.Tensor): Real images batch [B, C, H, W]
        fake (torch.Tensor): Generated images batch [B, C, H, W]
        alpha (float): Progressive growing fade-in parameter [0, 1]
        train_step (int): Current training resolution step
        device (str): Device to run computation on. Default: "cpu"
    
    Returns:
        torch.Tensor: Gradient penalty scalar value
    
    Example:
        >>> real = torch.randn(16, 3, 128, 128)
        >>> fake = gen(noise)
        >>> gp = gradient_penalty(critic, real, fake, alpha=1.0, train_step=5)
        >>> loss_critic = wasserstein_loss + LAMBDA_GP * gp
    
    References:
        Gulrajani et al., "Improved Training of Wasserstein GANs" (2017)
        https://arxiv.org/abs/1704.00028
    """
    batch_size, c, h, w = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    
    # Calculate critic scores
    mixed_scores = critic(interpolated, alpha, train_step)
    
    # ... rest of implementation
```

#### Comment Guidelines

**Good Comments** (explain WHY):
```python
# Use truncated normal to avoid extreme outliers in latent space
z = truncnorm.rvs(-2, 2, size=(batch_size, Z_DIM, 1, 1))

# Emergency save before GPU thermal throttling occurs
if gpu_temp > 85:
    save_checkpoint(gen, opt_gen, "emergency.pth")
```

**Bad Comments** (explain WHAT):
```python
# Set learning rate to 0.001
learning_rate = 1e-3  # ❌ Obvious from code

# Loop through resolutions
for resolution in [4, 8, 16, 32]:  # ❌ Redundant
```

**When to Comment**:
- Complex algorithms (WGAN-GP, progressive growing)
- Hardware-specific optimizations (batch sizes, mixed precision)
- Non-obvious design decisions
- Temporary workarounds (add TODO)

**When NOT to Comment**:
- Self-explanatory variable names
- Standard PyTorch operations
- Simple arithmetic

### 4. Configuration Documentation

Document all configurable parameters:

```python
# At top of progan_local.py

# ==================== CONFIG ====================
# Training Configuration
# --------------------

# START_TRAIN_AT_IMG_SIZE: Initial resolution for training
#   - Set to 4 to train from scratch
#   - Set to higher values (e.g., 256) to resume from checkpoint
#   - Must be power of 2: [4, 8, 16, 32, 64, 128, 256, 512]
START_TRAIN_AT_IMG_SIZE = 512

# BATCH_SIZES: Batch size for each progressive resolution step
#   - Optimized for RTX 3070 (8GB VRAM)
#   - Index corresponds to resolution: [4×4, 8×8, 16×16, ...]
#   - Reduce if encountering CUDA out of memory errors
BATCH_SIZES = [16, 16, 12, 8, 6, 4, 3, 2, 1]

# PROGRESSIVE_EPOCHS: Training epochs at each resolution
#   - Higher values = better convergence but longer training
#   - 20 epochs per resolution ≈ 3 hours total on RTX 3070
PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES)

# GRADIENT_ACCUMULATION_STEPS: Accumulate gradients before optimizer step
#   - Simulates larger batch sizes without increasing VRAM usage
#   - Effective batch size = BATCH_SIZES[i] * GRADIENT_ACCUMULATION_STEPS
#   - Increase to 4 or 8 if training is unstable
GRADIENT_ACCUMULATION_STEPS = 2

# Model Architecture
# --------------------

# Z_DIM: Latent vector dimensionality
#   - Higher values = more expressive generator
#   - 256 is standard for ProGAN, 512 for StyleGAN
Z_DIM = 256

# IN_CHANNELS: Initial feature map channels
#   - Controls model capacity
#   - Must match checkpoint if loading pretrained weights
IN_CHANNELS = 256

# LAMBDA_GP: Gradient penalty weight for WGAN-GP
#   - Standard value is 10
#   - Increase if discriminator overpowers generator
#   - Decrease if training is too slow
LAMBDA_GP = 10
```

## Documentation Workflow

### When to Update Documentation

| Change Type                  | Update Location                        |
|------------------------------|----------------------------------------|
| New CLI argument             | README.md, TRAINING_GUIDE.md           |
| Hyperparameter modification  | Inline comment + API_REFERENCE.md      |
| GPU requirements change      | README.md, CONFIGURATION_GPU.md        |
| Dataset structure change     | INSTRUCTIONS_DATASET.md                |
| New utility function         | API_REFERENCE.md + docstring           |
| Bug fix                      | Inline comment, changelog (if major)   |

### Documentation Review Checklist

Before merging PR:

- [ ] README commands are copy-pasteable and work
- [ ] Code has docstrings for all public functions/classes
- [ ] Configuration changes are explained
- [ ] Examples are tested and produce expected output
- [ ] English and Spanish docs are synchronized (if applicable)
- [ ] API changes are reflected in API_REFERENCE.md
- [ ] Breaking changes are noted prominently

## Special Documentation Patterns

### Command Output Examples

Show expected output for verification:

````markdown
```powershell
python ProGAN/config/check_gpu.py
```

**Expected Output**:
```
========================================
GPU Configuration Check
========================================
✅ CUDA available: True
✅ Current Device: 0
✅ Device Name: NVIDIA GeForce RTX 3070
✅ Device Capability: 8.6
✅ Memory Allocated: 0.00 GB
✅ Memory Reserved: 0.00 GB
========================================
```
````

### Troubleshooting Sections

Use Q&A format:

```markdown
## Troubleshooting

### Q: Training crashes with "CUDA out of memory"

**A**: Reduce batch size in `BATCH_SIZES`:

```python
# Before (default for RTX 3070)
BATCH_SIZES = [16, 16, 12, 8, 6, 4, 3, 2, 1]

# After (conservative for 6GB GPUs)
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 1, 1, 1]
```

Alternatively, increase `GRADIENT_ACCUMULATION_STEPS` to compensate.

---

### Q: Generated images are blurry

**A**: Possible causes:
1. **Insufficient training** - Increase `PROGRESSIVE_EPOCHS`
2. **Learning rate too high** - Reduce `LEARNING_RATE` to 5e-4
3. **Mode collapse** - Check TensorBoard for generator/critic loss balance
```

### Version-Specific Notes

```markdown
## Requirements

| Component    | Version   | Notes                              |
|--------------|-----------|------------------------------------|
| Python       | 3.8-3.10  | 3.11 not tested                    |
| PyTorch      | 2.0.0+    | Earlier versions may work          |
| CUDA         | 11.8/12.1 | Match with PyTorch wheel           |
| torchvision  | 0.15.0+   | Must match PyTorch version         |
| NumPy        | ≥1.19.0   | No upper bound                     |

**Known Issues**:
- PyTorch 1.13 has gradient penalty bug (upgrade to 2.0+)
- Windows: Avoid spaces in installation path
```

## Bilingual Documentation

### Translation Strategy

1. **Always write English first** (primary language)
2. **Translate to Spanish** within 24 hours for user-facing docs
3. **Keep structure identical** (same headings, examples)
4. **Use technical terms consistently**:
   - Generator → Generador
   - Training → Entrenamiento
   - Checkpoint → Punto de control

### Translation Tools

Use DeepL or GPT-4 for technical accuracy:

```python
# Automated translation helper
def translate_doc(input_file, output_file, target_lang='es'):
    """Translate markdown file preserving code blocks"""
    # Implementation
```

## Maintenance

### Documentation Debt Tracking

When code changes faster than docs:

```markdown
<!-- TODO: Update this section after PR #123 merges -->
## Training Configuration

⚠️ **Documentation outdated as of 2025-11-10**  
New batch size algorithm not yet documented. See `progan_local.py` lines 200-250.
```

### Periodic Reviews

- **Monthly**: Check README accuracy
- **Quarterly**: Regenerate examples with latest weights
- **Before release**: Full documentation audit

---

**Related Documents**:
- `copilot-instructions.md` - How to use these docs
- `copilot-contributing.md` - Documentation PR process
- `copilot-tests.md` - Testing examples

**Last Updated**: November 12, 2025