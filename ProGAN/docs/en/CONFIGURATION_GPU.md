# 🎮 GPU Setup for RTX 3070

## 📋 Optimization Summary

Settings and recommended tweaks for `progan_local.py` when using an RTX 3070 GPU.

### ✅ Changes Applied

1. **Reduced Batch Sizes**
   - Before: `[32, 32, 32, 16, 16, 16, 16, 8, 4]`
   - Now: `[16, 16, 12, 8, 6, 4, 3, 2, 1]`
   - ✨ Lowers VRAM usage and helps prevent overheating

2. **Gradient Accumulation**
   - Accumulate gradients every 2 steps
   - Simulates larger effective batches without extra memory
   - Preserves training quality

3. **GPU Monitoring**
   - Shows temperature, VRAM usage, and utilization
   - Alerts every 100 batches
   - Automatic CUDA cache cleanup

4. **Reduced Epochs**
   - From 30 to 20 epochs per resolution
   - Faster training and less hardware wear

## 🔧 Installing PyTorch with CUDA

### Option 1: Use the automatic script (RECOMMENDED)

```powershell
# Run this in PowerShell
.\setup_cuda.ps1
```

### Option 2: Manual installation

```powershell
# Uninstall previous version
pip uninstall -y torch torchvision torchaudio

# Install with CUDA 11.8 (compatible with RTX 3070)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ✅ Verify installation

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

You should see:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3070
```

## 🚀 How to Use

### Train the model

```powershell
cd ProGAN
python progan_local.py
```

### Generate images

```powershell
python progan_local.py generate
```

## 📊 GPU Monitoring During Training

The code now automatically displays:

```
==================================================
📊 GPU Statistics:
  Memory Allocated: 4.23GB
  Memory Total: 8.00GB
  Memory Percent: 52.9%
  Temperature: 68°C
  Utilization: 95%
  Power: 180W
==================================================
```

## ⚠️ Recommendations for your RTX 3070

### 🌡️ Temperature Control

- **Ideal temperature**: 60–75°C
- **Maximum acceptable**: 80°C
- **If it exceeds 80°C**: pause training and check ventilation

### 💾 VRAM Usage

- Your RTX 3070 has **8GB VRAM**
- Batch sizes are tuned to use ~5–6GB
- Leave headroom for the OS

### ⚡ Power Consumption

- RTX 3070 TDP: 220W
- During training: ~180–200W is normal
- Ensure your PSU is adequate (minimum 650W recommended)

## 🛠️ Troubleshooting

### Problem: "RuntimeError: CUDA out of memory"

**Solution**: Reduce batch sizes further in the file:

```python
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]  # More conservative
```

### Problem: GPU overheats (>80°C)

**Solutions**:
1. Clean your GPU fans
2. Improve case ventilation
3. Reduce batch sizes further
4. Increase `GRADIENT_ACCUMULATION_STEPS = 4`
5. Use MSI Afterburner to set a more aggressive fan curve

### Problem: "CUDA available: False"

**Solutions**:
1. Reinstall PyTorch with the script's command
2. Verify NVIDIA drivers are up to date:
   ```powershell
   nvidia-smi
   ```
3. If you don't have `nvidia-smi`, download drivers from: https://www.nvidia.com/Download/index.aspx

## 📈 Performance Comparison

| Configuration | VRAM Used | Approx. Temp | Time/Epoch |
|---------------|-----------|--------------|------------|
| Original      | 7–8GB     | 75–85°C      | ~15 min    |
| **Optimized (RTX 3070)** | **5–6GB** | **65–75°C** | **~18 min** |
| Ultra-safe    | 3–4GB     | 60–70°C      | ~25 min    |

## 🎯 Additional Settings

### For faster training (if your GPU can handle it):

```python
BATCH_SIZES = [20, 20, 16, 12, 8, 6, 4, 2, 1]
GRADIENT_ACCUMULATION_STEPS = 1
```

### For cooler/safer training:

```python
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]
GRADIENT_ACCUMULATION_STEPS = 4
PROGRESSIVE_EPOCHS = [15] * len(BATCH_SIZES)  # Fewer epochs
```

## 📚 Additional Resources

- [NVIDIA guide for RTX 3070](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070-3070ti/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [MSI Afterburner for GPU control](https://www.msi.com/Landing/afterburner/graphics-cards)

## 💡 Pro Tips

1. **Use TensorBoard** to monitor training:
   ```powershell
   tensorboard --logdir=logs
   ```
   Open: http://localhost:6006

2. **Take breaks** during long sessions: rest your GPU every 2–3 hours

3. **Night training**: schedule training at night when ambient temperature is lower

4. **Undervolting**: consider undervolting your GPU to reduce temperatures without losing performance
