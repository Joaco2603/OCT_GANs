#!/usr/bin/env python3
"""
ProGAN Training Entry Point
===========================
Main entry point for training ProGAN on OCT DRUSEN images.

Usage:
    python main.py           # Start/continue training
    python main.py generate  # Generate images from trained model

For more information, see the documentation in ProGAN/docs/
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import main, generation_mode, seed_everything


if __name__ == "__main__":
    # Set random seed for reproducibility
    seed_everything(42)
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generation_mode()
    else:
        main()
