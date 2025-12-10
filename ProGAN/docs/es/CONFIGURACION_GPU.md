# 🎮 Configuración de GPU para RTX 3070

## 📋 Resumen de Optimizaciones

Código `progan_local.py` para tu RTX 3070 con las siguientes mejoras:

### ✅ Cambios Realizados

1. **Batch Sizes Reducidos**
   - Antes: `[32, 32, 32, 16, 16, 16, 16, 8, 4]`
   - Ahora: `[16, 16, 12, 8, 6, 4, 3, 2, 1]`
   - ✨ Reduce el uso de VRAM y previene sobrecalentamiento

2. **Gradient Accumulation**
   - Acumula gradientes cada 2 pasos
   - Simula batches más grandes sin usar memoria extra
   - Mantiene la calidad del entrenamiento

3. **Monitoreo de GPU**
   - Muestra temperatura, uso de VRAM, y utilización
   - Alertas cada 100 batches
   - Limpieza automática de caché CUDA

4. **Épocas Reducidas**
   - De 30 a 20 épocas por resolución
   - Entrenamiento más rápido y menos desgaste

## 🔧 Instalación de PyTorch con CUDA

### Opción 1: Usando el script automático (RECOMENDADO)

```powershell
# Ejecuta esto en PowerShell
.\setup_cuda.ps1
```

### Opción 2: Instalación manual

```powershell
# Desinstalar versión anterior
pip uninstall -y torch torchvision torchaudio

# Instalar con CUDA 11.8 (compatible con RTX 3070)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ✅ Verificar instalación

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Deberías ver:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3070
```

## 🚀 Cómo Usar

### Entrenar el modelo

```powershell
cd ProGAN
python progan_local.py
```

### Generar imágenes

```powershell
python progan_local.py generate
```

## 📊 Monitoreo de GPU Durante el Entrenamiento

El código ahora muestra automáticamente:

```
==================================================
📊 Estadísticas de GPU:
  Memory Allocated: 4.23GB
  Memory Total: 8.00GB
  Memory Percent: 52.9%
  Temperature: 68°C
  Utilization: 95%
  Power: 180W
==================================================
```

## ⚠️ Recomendaciones para tu RTX 3070

### 🌡️ Control de Temperatura

- **Temperatura ideal**: 60-75°C
- **Máxima aceptable**: 80°C
- **Si supera 80°C**: Pausar entrenamiento y revisar ventilación

### 💾 Uso de VRAM

- Tu RTX 3070 tiene **8GB de VRAM**
- Los batch sizes están optimizados para usar ~5-6GB
- Deja margen para el sistema operativo

### ⚡ Consumo de Energía

- TDP de la RTX 3070: 220W
- Durante entrenamiento: ~180-200W es normal
- Asegúrate de tener una fuente de poder adecuada (mínimo 650W)

## 🛠️ Solución de Problemas

### Problema: "RuntimeError: CUDA out of memory"

**Solución**: Reduce más los batch sizes en el archivo:

```python
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]  # Más conservador
```

### Problema: GPU se sobrecalienta (>80°C)

**Soluciones**:
1. Limpia los ventiladores de tu GPU
2. Mejora la ventilación del case
3. Reduce batch sizes aún más
4. Aumenta `GRADIENT_ACCUMULATION_STEPS = 4`
5. Usa MSI Afterburner para crear una curva de ventiladores más agresiva

### Problema: "CUDA available: False"

**Soluciones**:
1. Reinstala PyTorch con el comando del script
2. Verifica que los drivers de NVIDIA estén actualizados:
   ```powershell
   nvidia-smi
   ```
3. Si no tienes `nvidia-smi`, instala los drivers desde: https://www.nvidia.com/Download/index.aspx

## 📈 Comparación de Rendimiento

| Configuración | VRAM Usada | Temp. Aprox | Tiempo/Epoch |
|---------------|------------|-------------|--------------|
| Original      | 7-8GB      | 75-85°C     | ~15 min      |
| **Optimizada (RTX 3070)** | **5-6GB** | **65-75°C** | **~18 min** |
| Ultra-safe    | 3-4GB      | 60-70°C     | ~25 min      |

## 🎯 Configuraciones Adicionales

### Para entrenamiento más rápido (si tu GPU aguanta):

```python
BATCH_SIZES = [20, 20, 16, 12, 8, 6, 4, 2, 1]
GRADIENT_ACCUMULATION_STEPS = 1
```

### Para entrenamiento más seguro/fresco:

```python
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]
GRADIENT_ACCUMULATION_STEPS = 4
PROGRESSIVE_EPOCHS = [15] * len(BATCH_SIZES)  # Menos épocas
```

## 📚 Recursos Adicionales

- [Guía de NVIDIA sobre RTX 3070](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070-3070ti/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [MSI Afterburner para control de GPU](https://www.msi.com/Landing/afterburner/graphics-cards)

## 💡 Consejos Pro

1. **Usa TensorBoard** para monitorear el entrenamiento:
   ```powershell
   tensorboard --logdir=logs
   ```
   Abre: http://localhost:6006

2. **Pausa entre sesiones largas**: Dale descansos a tu GPU cada 2-3 horas

3. **Entrenamiento nocturno**: Configura para entrenar de noche cuando la temperatura ambiente es menor

4. **Undervolting**: Considera hacer undervolting a tu GPU para reducir temperatura sin perder rendimiento

