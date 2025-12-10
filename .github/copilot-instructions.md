# AI Assistant Instructions for OCT_GANs Repository

## Overview
This document orients AI coding assistants (GitHub Copilot, Claude, GPT, etc.) to the OCT_GANs repository structure, patterns, and workflows. Follow these guidelines when making edits, suggestions, or answering questions about this project.

## Model Routing Intelligence 🧠

Before responding to any request, determine which AI assistant model is best suited:

### Task Classification & Model Routing

**ProGAN Architecture & Deep Learning Tasks** → **Claude Opus / GPT-4 (Advanced reasoning)**
- Complex model architecture changes (Generator/Discriminator modifications)
- Training loop optimization and gradient accumulation strategies
- Mixed-precision training and memory optimization
- Progressive training phase transitions
- Loss function design and balancing

**Code Refactoring & Structure** → **Claude Sonnet / GPT-4 Turbo (Balanced)**
- Reorganizing code modules and improving separation of concerns
- Creating utility functions and helper classes
- Implementing design patterns (Factory, Strategy, Observer)
- Code quality improvements and best practices

**Quick Fixes & Documentation** → **GPT-3.5 / Claude Haiku (Fast response)**
- Fixing syntax errors and typos
- Adding inline comments and docstrings
- Updating README sections
- Simple variable renaming

**Data Processing & Visualization** → **Specialized models with vision capabilities**
- OCT image preprocessing pipelines
- Data augmentation strategies
- TensorBoard integration and logging
- Image quality metrics and validation

**Performance Optimization** → **Claude Opus / GPT-4 (Deep analysis)**
- GPU memory profiling and optimization
- Batch size tuning for RTX 3070
- CUDA kernel optimization suggestions
- Training speed improvements

### Routing Decision Tree
```
User Request
    │
    ├─ Architecture/Theory → Claude Opus / GPT-4
    ├─ Code Structure → Claude Sonnet / GPT-4 Turbo
    ├─ Quick Fix → GPT-3.5 / Claude Haiku
    ├─ Data/Vision → Vision-capable model
    └─ Performance → Claude Opus + profiling tools
```

## Key Project Locations

### Primary Implementation Files
- `ProGAN/model/progan_local.py` — **MAIN FILE**: Complete ProGAN implementation (Generator, Discriminator, training loop, data loader)
  - Most model and training logic changes happen here
  - Contains progressive training orchestration
  - Includes GPU monitoring and emergency save functionality
  
- `ProGAN/model/generator.py` — Generator architecture (currently integrated in progan_local.py)
- `ProGAN/model/discriminator.py` — Discriminator/Critic architecture (currently integrated in progan_local.py)

### Configuration & Setup
- `ProGAN/config/check_gpu.py` — Environment & GPU verification script (run before training)
- `ProGAN/scripts/setup_cuda.ps1` — PowerShell helper to install PyTorch+CUDA on Windows
- `ProGAN/docs/en/CONFIGURATION_GPU.md` — GPU configuration guide
- `ProGAN/docs/en/INSTRUCTIONS_DATASET.md` — Dataset preparation instructions

### Persistent Artifacts
- `weights/` — Pretrained model checkpoints (`.pth` files)
- `ProGAN/generated_images/` — Generated OCT images during/after training
- `ProGAN/logs/` — TensorBoard event files for training visualization

### Persistent Artifacts
- `weights/` — Pretrained model checkpoints (`.pth` files)
- `ProGAN/generated_images/` — Generated OCT images during/after training
- `ProGAN/logs/` — TensorBoard event files for training visualization

## Architecture & Design Philosophy

### Why This Organization?
This repository is a **local adaptation** of a Colab ProGAN notebook optimized for:
- **Windows environment** with RTX 3070 GPU (8GB VRAM)
- **Progressive training** from 4×4 to 512×512 resolution
- **Medical imaging specifics** (OCT retinal scans with DRUSEN pathology)

### Core Design Principles
1. **Single-file simplicity**: `progan_local.py` centralizes model, training, and utilities for easier debugging
2. **Progressive growth**: Resolution increases gradually (4→8→16→32→64→128→256→512)
3. **Memory efficiency**: Gradient accumulation + mixed precision to work within 8GB VRAM
4. **Robustness**: Emergency save on Ctrl+C, auto-checkpointing, GPU monitoring

### Data Flow
```
OCT2017 Dataset → ImageFolder Loader → Progressive Batches → Training Loop
      ↓                                           ↓
train/DRUSEN/                            Variable batch sizes by resolution
                                                   ↓
                                         Generator ← Noise Vector (Z_DIM=256)
                                                   ↓
                                         Discriminator (Critic with GP)
                                                   ↓
                                         TensorBoard Logs + Checkpoints
```

## Critical Developer Workflows

### Environment Setup (First Time)
```powershell
# Create conda environment
conda create -n oct_gans python=3.8 -y
conda activate oct_gans

# Install PyTorch with CUDA 11.8 (RTX 3070 compatible)
cd ProGAN
.\scripts\setup_cuda.ps1
# OR manually:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU and dependencies
python config/check_gpu.py
```

### Training Workflow
```powershell
# From repository root
python ProGAN/model/progan_local.py

# Monitor training
tensorboard --logdir=ProGAN/logs
# Open http://localhost:6006
```

### Generation (Using Saved Generator)
```powershell
# Edit progan_local.py: set CHECKPOINT_GEN to your .pth file
python ProGAN/model/progan_local.py generate
```

### Dataset Setup
Ensure dataset structure matches:
```
ProGAN/data/OCT2017/train/DRUSEN/*.jpeg
```
See `ProGAN/docs/en/INSTRUCTIONS_DATASET.md` for download instructions.

## Project-Specific Conventions

### Path Management
All paths use `Path(__file__).parent` for portability:
```python
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "OCT2017" / "train"
WEIGHTS_DIR = BASE_DIR / "weights"  # Always save checkpoints here
OUTPUT_DIR = BASE_DIR / "generated_images"
LOGS_DIR = BASE_DIR / "logs"
```

**Rule**: Use these constants instead of hardcoding paths.

### Resolution & Batch Size Mapping
```python
BATCH_SIZES = [16, 16, 12, 8, 6, 4, 3, 2, 1]  # Optimized for RTX 3070
PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES)

# Training loop uses:
batch_size = BATCH_SIZES[int(log2(image_size / 4))]
# 4×4 → BATCH_SIZES[0]=16
# 8×8 → BATCH_SIZES[1]=16
# ...
# 512×512 → BATCH_SIZES[8]=1
```

**Rule**: When adding resolutions, update both `BATCH_SIZES` and `PROGRESSIVE_EPOCHS`.

### Checkpoint API
```python
# Save
save_checkpoint(model, optimizer, filename)
# Example: save_checkpoint(gen, opt_gen, "weights/generator_DRUSEN_local.pth")

# Load
load_checkpoint(checkpoint_path, model, optimizer, lr)
```

**Rule**: 
- Always save to `weights/` directory
- Use descriptive names: `{model_type}_{class}_{version}.pth`
- Example: `generator_DRUSEN_v11.pth`, `critic_DRUSEN_local.pth`

### Emergency Save Behavior
```python
setup_emergency_save(gen, critic, opt_gen, opt_critic)
```
- Registers SIGINT (Ctrl+C) handler
- Saves timestamped emergency checkpoints: `EMERGENCY_generator_20250112_143022.pth`
- **Rule**: Preserve this mechanism when refactoring training loops

### Mixed Precision & Gradient Accumulation
```python
GRADIENT_ACCUMULATION_STEPS = 2
scaler = torch.cuda.amp.GradScaler()

# Training uses:
with torch.cuda.amp.autocast():
    # forward pass
# Accumulate gradients before optimizer step
```

**Rule**: Any batch logic changes must account for gradient accumulation to maintain effective batch size.

## Code Modification Guidelines

### ✅ Safe Changes
- Adjusting hyperparameters (learning rate, epochs)
- Adding logging statements
- Improving comments/docstrings
- Creating utility functions in separate files
- Adding validation metrics

### ⚠️ Careful Changes (Test Thoroughly)
- Modifying Generator/Discriminator architectures
- Changing progressive training schedule
- Altering loss functions (WGAN-GP sensitive)
- Batch size / accumulation changes
- Checkpoint save/load logic

### ❌ Avoid
- Removing emergency save functionality
- Hardcoding absolute paths
- Changing dataset directory structure without updating docs
- Breaking compatibility with existing `.pth` files

## Testing Checklist

Before proposing code changes, verify:

1. **Dataset exists**: `ProGAN/data/OCT2017/train/DRUSEN/` contains images
2. **Paths are relative**: Uses `BASE_DIR` and `Path` objects
3. **Checkpoint compatibility**: Existing `weights/*.pth` files still load
4. **Quick smoke test**: Run with `START_TRAIN_AT_IMG_SIZE=4` and `PROGRESSIVE_EPOCHS=[1,1,...]` for one epoch
5. **GPU monitoring works**: `get_gpu_stats()` doesn't crash on non-NVIDIA systems

## Integration Points

### Runtime Dependencies
- **Python 3.8+**
- **PyTorch** (torch, torchvision) with CUDA support
- **NumPy, Pillow, tqdm, tensorboard** (standard ML stack)

### External Tools
- `nvidia-smi` for GPU temperature/utilization (optional, gracefully degrades)
- `conda` for environment management (recommended)
- **TensorBoard** for training visualization

### Dataset Acquisition
- Kaggle API or manual download (`archive.zip`)
- Handled by `ProGAN/config/download_dataset.py`
- Instructions in `ProGAN/docs/en/INSTRUCTIONS_DATASET.md`

## Common Modification Patterns

### Change Starting Resolution
```python
# In progan_local.py (top of file)
START_TRAIN_AT_IMG_SIZE = 4  # Start from scratch
# or
START_TRAIN_AT_IMG_SIZE = 256  # Resume from 256×256
```

### Quick Generation Test (No Training)
```python
# 1. Set checkpoint paths
CHECKPOINT_GEN = "weights/generator_DRUSEN_v11.pth"
LOAD_MODEL = True

# 2. Run in generation mode
python ProGAN/model/progan_local.py generate
```

### Add New Utility Script
- Place in `ProGAN/utils/` or `ProGAN/scripts/`
- Import from project root: `from ProGAN.utils import my_util`
- Update docs if it's user-facing

## Documentation Requirements

When making changes that affect:
- **GPU/training behavior** → Update `ProGAN/docs/en/CONFIGURATION_GPU.md`
- **Dataset setup** → Update `ProGAN/docs/en/INSTRUCTIONS_DATASET.md`
- **CLI usage** → Update root `README.md`
- **API/checkpoints** → Add migration notes in commit message

## AI Assistant Collaboration Protocol

### When You Need More Context
Ask for:
- Specific line ranges: "Show me lines 200-300 of progan_local.py"
- Related files: "What's in utils/data_loader.py?"
- Error context: "Run the script and show me the traceback"

### When Proposing Changes
Provide:
1. **Rationale**: Why this change improves the code
2. **Smoke test**: One-line command to verify it works
3. **Rollback plan**: How to undo if it breaks
4. **Affected areas**: What else might be impacted

### Example
```
Change: Increase GRADIENT_ACCUMULATION_STEPS from 2 to 4

Rationale: Simulate larger effective batch size (2x) without exceeding VRAM

Smoke test:
python ProGAN/model/progan_local.py  # Run for 10 batches, check GPU memory

Rollback: Revert GRADIENT_ACCUMULATION_STEPS = 2

Affected areas:
- Training speed (slower per epoch, but may converge faster)
- Optimizer step frequency (half as many updates)
```

## Cross-Reference to Other Instruction Files
- **Architecture details** → `copilot-arquitecture.md`
- **Security & IP policies** → `copilot-guardrails.md`
- **Documentation standards** → `copilot-documentation.md`
- **Testing strategy** → `copilot-tests.md`
- **Contribution workflow** → `copilot-contributing.md`

---

**Last Updated**: November 12, 2025  
**Maintainer**: OCT_GANs Team  
**Questions?** Open an issue with label `documentation`