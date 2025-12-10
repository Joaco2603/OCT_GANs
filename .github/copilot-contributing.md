# Contributing to OCT_GANs

## Welcome! 🎉

Thank you for considering contributing to OCT_GANs! This document provides guidelines to make the contribution process smooth and effective.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Contribution Types](#contribution-types)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing Requirements](#testing-requirements)
8. [Documentation Requirements](#documentation-requirements)
9. [Review Process](#review-process)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Be respectful, constructive, and collaborative.

### Expected Behavior

- **Respectful Communication**: Critique code, not people
- **Constructive Feedback**: Offer solutions, not just problems
- **Collaborative Spirit**: Help others learn and grow
- **Acknowledge Contributions**: Credit ideas and work properly

### Unacceptable Behavior

- Personal attacks or insults
- Harassment of any kind
- Publishing private information
- Spam or off-topic discussions

**Report violations** to: [maintainer email or issue tracker]

## Getting Started

### Prerequisites

Before contributing:

1. **Familiarize yourself with the project**:
   - Read `README.md`
   - Review `copilot-instructions.md`
   - Run the smoke tests

2. **Set up development environment**:
```powershell
# Clone repository
git clone https://github.com/Joaco2603/OCT_GANs.git
cd OCT_GANs

# Create conda environment
conda create -n oct_gans_dev python=3.8 -y
conda activate oct_gans_dev

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8 mypy

# Verify setup
python ProGAN/config/check_gpu.py
```

3. **Find an issue or feature**:
   - Browse [open issues](https://github.com/Joaco2603/OCT_GANs/issues)
   - Look for labels: `good first issue`, `help wanted`, `bug`
   - Or propose a new feature (create an issue first)

## Development Workflow

### Branching Strategy

We use **Git Flow**:

```
main (stable releases)
  └── develop (integration branch)
      ├── feature/add-stylegan-mixing
      ├── fix/gpu-memory-leak
      └── docs/update-training-guide
```

**Branch Naming Convention**:
- `feature/<description>` - New features
- `fix/<description>` - Bug fixes
- `docs/<description>` - Documentation updates
- `refactor/<description>` - Code refactoring
- `test/<description>` - Test additions

### Development Process

1. **Create a branch**:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-awesome-feature
```

2. **Make changes iteratively**:
   - Write code
   - Add tests
   - Update documentation
   - Commit frequently with clear messages

3. **Keep branch updated**:
```bash
git fetch origin
git rebase origin/develop
```

4. **Test before pushing**:
```powershell
# Run tests
pytest tests/ -v

# Check code style
black ProGAN/ --check
flake8 ProGAN/

# Run smoke test
python ProGAN/model/progan_local.py --epochs 1 --start-res 4
```

5. **Push and create PR**:
```bash
git push origin feature/my-awesome-feature
```

## Contribution Types

### 1. Bug Fixes

**Process**:
1. Create issue with **bug report template**
2. Reproduce the bug locally
3. Write a failing test
4. Fix the bug
5. Verify test passes
6. Submit PR with issue reference

**Example Commit**:
```
fix: resolve CUDA memory leak in gradient accumulation

- Clear cache after each resolution transition
- Add memory monitoring test
- Reduces peak VRAM usage by 15%

Closes #42
```

### 2. New Features

**Process**:
1. Create issue with **feature proposal template**
2. Discuss design with maintainers (wait for approval)
3. Implement feature on branch
4. Add comprehensive tests
5. Update documentation
6. Submit PR with design rationale

**Example Commit**:
```
feat: add StyleGAN-inspired style mixing

- Implement adaptive instance normalization (AdaIN)
- Add mixing regularization during training
- Update generator architecture (backward compatible)
- Add examples to docs/en/ADVANCED_FEATURES.md

Refs #38
```

### 3. Documentation Improvements

**Process**:
1. Identify unclear/outdated documentation
2. Create issue (optional for minor fixes)
3. Update English docs first
4. Translate to Spanish (if user-facing)
5. Verify examples work
6. Submit PR

**Example Commit**:
```
docs: clarify batch size tuning for 6GB GPUs

- Add RTX 2060/3060 configuration examples
- Update memory usage table
- Fix outdated TensorBoard command

```

### 4. Performance Optimizations

**Process**:
1. Profile current performance (time, memory)
2. Create issue with benchmarks
3. Implement optimization
4. Re-profile and compare
5. Add performance test
6. Submit PR with before/after metrics

**Example Commit**:
```
perf: optimize data loading with prefetching

- Enable pin_memory and persistent_workers
- Increase num_workers from 2 to 4
- Reduces epoch time by 23% (128x128 resolution)

Benchmark: RTX 3070, batch_size=4, 1000 iterations
Before: 18.2 min/epoch
After: 14.0 min/epoch

```

## Pull Request Process

### PR Checklist

Before submitting, ensure:

- [ ] **Code Quality**
  - [ ] Follows PEP 8 style guide
  - [ ] No unused imports or variables
  - [ ] Type hints for new functions
  - [ ] Docstrings for public APIs

- [ ] **Testing**
  - [ ] All existing tests pass
  - [ ] New tests for new functionality
  - [ ] Coverage ≥ 80% for new code
  - [ ] Smoke test completes

- [ ] **Documentation**
  - [ ] README updated (if needed)
  - [ ] API reference updated
  - [ ] Inline comments for complex logic
  - [ ] Spanish translation (for user docs)

- [ ] **Git Hygiene**
  - [ ] Descriptive commit messages
  - [ ] Rebased on latest develop
  - [ ] No merge commits (use rebase)
  - [ ] Logical, atomic commits

### PR Template

```markdown
## Description
Brief description of changes (2-3 sentences)

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- Bullet list of specific changes
- Include file/function names

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

**Test Plan**:
```powershell
# Commands to verify the change
python ProGAN/model/progan_local.py --config new_feature
```

## Performance Impact
- Training speed: [No change | +5% faster | etc.]
- Memory usage: [No change | -200MB | etc.]
- Model quality: [No change | FID improved by X | etc.]

## Screenshots (if UI/visual change)
Before | After

## Related Issues
Closes #XX
Refs #YY

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Reviewed own changes
```

### PR Size Guidelines

- **Small** (< 100 lines): Fast review, merge within 1-2 days
- **Medium** (100-500 lines): Review within 3-5 days
- **Large** (> 500 lines): Break into smaller PRs if possible

**Too large?** Split into:
1. Refactoring/setup PR
2. Core functionality PR
3. Documentation PR

## Coding Standards

### Python Style

Follow **PEP 8** with these specifics:

```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces

# Imports: Group by stdlib, third-party, local
import os
import sys

import torch
import numpy as np

from ProGAN.model.generator import Generator
from ProGAN.utils.helpers import get_loader

# Function naming: snake_case
def calculate_gradient_penalty(critic, real, fake):
    pass

# Class naming: PascalCase
class ProgressiveGenerator(nn.Module):
    pass

# Constants: UPPER_CASE
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Type Hints

Add type hints for clarity:

```python
from typing import Tuple, Optional, List
import torch

def train_epoch(
    gen: torch.nn.Module,
    critic: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (generator_loss, critic_loss)
    """
    pass
```

### Error Handling

Fail gracefully with informative messages:

```python
# Bad
def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint['model']

# Good
def load_checkpoint(path: str) -> dict:
    """Load model checkpoint with validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Available checkpoints in weights/:\n"
            f"  {os.listdir('weights/')}"
        )
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint {path}. "
            f"File may be corrupted. Error: {e}"
        )
    
    required_keys = ['model_state_dict', 'optimizer_state_dict']
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint missing required keys: {missing}"
        )
    
    return checkpoint
```

### Configuration Management

Use constants at module top:

```python
# progan_local.py

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "OCT2017" / "train"
WEIGHTS_DIR = BASE_DIR / "weights"

# Training
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 12, 8, 6, 4, 3, 2, 1]

# Model
Z_DIM = 256
IN_CHANNELS = 256
```

**Don't** scatter magic numbers:
```python
# Bad
for i in range(9):  # What is 9?
    loader = get_loader(img_size=4*(2**i), batch_size=16)  # Why 16?

# Good
for i, batch_size in enumerate(BATCH_SIZES):
    resolution = 4 * (2 ** i)
    loader = get_loader(img_size=resolution, batch_size=batch_size)
```

## Testing Requirements

All contributions must include tests. See `copilot-tests.md` for details.

### Minimum Test Coverage

- **Bug fixes**: Test that reproduces the bug (should fail before fix)
- **New features**: Unit tests + integration test
- **Refactoring**: Maintain or increase coverage

### Running Tests Locally

```powershell
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=ProGAN --cov-report=html

# Specific test file
pytest tests/test_generator.py -v

# Specific test function
pytest tests/test_generator.py::test_output_shape -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Good Tests

```python
# test_checkpoint.py

def test_checkpoint_save_and_load_preserves_state():
    """
    Saving and loading a checkpoint should restore exact model state.
    
    Given: A trained generator with specific weights
    When: Checkpoint is saved then loaded into new model
    Then: New model produces identical outputs for same inputs
    """
    # Arrange
    gen_original = Generator(Z_DIM=256, IN_CHANNELS=256)
    opt = torch.optim.Adam(gen_original.parameters(), lr=1e-3)
    noise = torch.randn(4, 256, 1, 1)
    
    # Train for a few steps (modify weights)
    for _ in range(10):
        out = gen_original(noise, alpha=1.0, steps=0)
        loss = out.mean()
        loss.backward()
        opt.step()
    
    # Get output before save
    output_before = gen_original(noise, alpha=1.0, steps=0)
    
    # Act
    save_checkpoint(gen_original, opt, "test_checkpoint.pth")
    
    gen_loaded = Generator(Z_DIM=256, IN_CHANNELS=256)
    opt_loaded = torch.optim.Adam(gen_loaded.parameters(), lr=1e-3)
    load_checkpoint("test_checkpoint.pth", gen_loaded, opt_loaded, lr=1e-3)
    
    # Get output after load
    output_after = gen_loaded(noise, alpha=1.0, steps=0)
    
    # Assert
    assert torch.allclose(output_before, output_after, atol=1e-6), \
        "Checkpoint did not preserve model state"
    
    # Cleanup
    os.remove("test_checkpoint.pth")
```

## Documentation Requirements

See `copilot-documentation.md` for comprehensive standards.

### Quick Checklist

- [ ] Public functions have docstrings
- [ ] Complex algorithms have inline comments
- [ ] Configuration changes explained
- [ ] README updated if CLI changes
- [ ] Examples tested and working

## Review Process

### Reviewer Guidelines

Reviewers will check:

1. **Correctness** - Does it work? Are edge cases handled?
2. **Design** - Is this the right approach? Any better alternatives?
3. **Tests** - Adequate coverage? Edge cases tested?
4. **Style** - Follows coding standards?
5. **Documentation** - Clear and sufficient?
6. **Performance** - Any regressions?

### Addressing Feedback

- **Respond promptly** (within 2 days)
- **Ask questions** if feedback is unclear
- **Push new commits** for changes (don't force-push during review)
- **Mark conversations** as resolved when addressed

### Approval Process

- **1 approval required** for merge
- **Maintainer review** for architectural changes
- **CI must pass** (tests, linting)

## Recognition

Contributors are acknowledged in:
- `README.md` Contributors section
- Release notes for their contributions
- Git commit history (preserved via rebase)

## Questions?

- **Technical questions**: Open an issue with label `question`
- **Process questions**: Comment on existing PR
- **Private concerns**: Email maintainers

---

**Thank you for contributing to OCT_GANs!** 🚀

**Related Documents**:
- `copilot-instructions.md` - Development guide
- `copilot-tests.md` - Testing standards
- `copilot-documentation.md` - Documentation standards

**Last Updated**: November 12, 2025