# OCT_GANs
This repository contains code and resources to train and generate OCT images (primarily using ProGAN) with the goal of producing synthetic retinal images affected by DRUSEN. Generated images can be used for research, data augmentation, and exploratory clinical analysis.

Note: this repository is a fork of https://github.com/T0mSIlver/OCT_GANs.

## Quick summary
- Project: OCT image generation (DRUSEN) using ProGAN
- Example notebooks and scripts: `ProGAN/`
- Data root: `ProGAN/data/OCT2017/` (e.g. `train/DRUSEN/`)
- Pretrained weights: `weights/`

## Project description
Contains training and image generation code (and notebooks) for ProGAN experiments on OCT images. The focus is on generating realistic DRUSEN-affected OCT images for downstream tasks like augmentation and model evaluation.

### Example outputs
Generated sample images are available in `ProGAN/generated_images/` after training.

## Quick start (Windows - PowerShell, using conda)
Assumptions: conda/miniconda installed. GPU with CUDA is recommended for training. If a `requirements.txt` is not present, you can install required packages manually (see notes below).

1) Create and activate a conda environment (Python 3.8):

```powershell
conda create -n oct_gans python=3.8 -y
conda activate oct_gans
```

2) Install dependencies (if `requirements.txt` exists):

```powershell
pip install -r requirements.txt
```

Alternatively, install typical dependencies manually (GPU users should pick a matching torch+cuda build):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow tqdm tensorboard
```

3) Verify GPU/CUDA availability (included script):

```powershell
python ProGAN/config/preprocessing/check_gpu.py
```

4) Run training or generate images:

```powershell
cd ProGAN
python main.py           # Entrenar
python main.py generate  # Generar imágenes
```



## Important repository layout
- `ProGAN/` — code and utilities for training and generation
  - `model/` — ProGAN architecture and training logic
  - `config/` — configuration, checkpoints, and preprocessing utilities
  - `data/OCT2017/` — dataset (e.g. `train/DRUSEN/`)
  - `generated_images/` — generated images during/after training
  - `logs/` — TensorBoard events
  - `scripts/` — helper scripts (e.g. `setup_cuda.ps1`)
  - `main.py` — main entry point for training and generation
- `weights/` — pretrained models (e.g. `generator_DRUSEN_local.pth`, `critic_DRUSEN_local.pth`)

## Dataset
The expected dataset layout is under `ProGAN/data/OCT2017/`. Ensure images are organized by class (for example `DRUSEN/`) before training.

## Pretrained weights
The `weights/` folder contains saved models that can be used to generate images or as starting points for fine-tuning.

## References
- Wang, Lehan et al., "Fundus-Enhanced Disease-Aware Distillation Model for Retinal Disease Classification from OCT Images." (arXiv 2023). https://doi.org/10.48550/arXiv.2308.00291

This paper discusses knowledge distillation and multimodal complementary learning relevant to OCT and fundus imagery.

## Contributing
- Open issues for bugs or feature requests.
- For reproducible experiments, include the training script, seed, and parameter settings in the PR description.

## License
See the `LICENSE` file at the repository root.

## Contact
Open an issue to ask questions or propose collaborations.

## Notes and assumptions
- The project appears to use Python and PyTorch (weights are `.pth` files). If you want, I can:
  - Add a `requirements.txt` with pinned versions.
  - Provide a small `run_demo.ps1` script that activates the conda env and runs a quick generation test.



