# ProGAN para Generación de Imágenes OCT DRUSEN - Versión Local

## 📖 Descripción

Este proyecto utiliza **Progressive GAN (ProGAN)** para generar imágenes sintéticas de OCT (Tomografía de Coherencia Óptica) que muestran DRUSEN, depósitos amarillentos bajo la retina que pueden indicar degeneración macular.

### ¿Qué hace este código?

**ProGAN NO convierte fundus a OCT**. En cambio:
- ✅ **Genera imágenes OCT sintéticas** completamente nuevas a partir de ruido aleatorio
- ✅ **Aprende la distribución** de imágenes OCT con DRUSEN
- ✅ **Aumenta el dataset** para entrenar otros modelos de diagnóstico

## 🎯 Requisitos

### Hardware
- **GPU NVIDIA recomendada** (CUDA) para entrenamiento
- CPU funcionará pero será muy lento
- Mínimo 8GB RAM (16GB recomendado)
- ~10GB de espacio en disco para dataset y modelos

### Software
- Python 3.8+
- PyTorch con soporte CUDA (si tienes GPU NVIDIA)
- Dependencias ya instaladas en tu entorno virtual

## 📁 Estructura de Archivos

```
ProGAN/
├── progan_local.py          # ← Script principal adaptado
├── README_LOCAL.md          # ← Este archivo
├── data/                    # ← Coloca tu dataset aquí
│   └── OCT2017/
│       └── train/
│           └── DRUSEN/      # Imágenes de entrenamiento
│               ├── img1.jpeg
│               ├── img2.jpeg
│               └── ...
├── weights/                 # Pesos del modelo (se crean automáticamente)
├── generated_images/        # Imágenes generadas
└── logs/                    # Logs de TensorBoard
```

## 📥 Paso 1: Obtener el Dataset

### ⚡ Método Rápido (Recomendado)

Usa el script de preparación automática:

```powershell
python download_dataset.py
```

Este script te guiará para:
1. Descargar el dataset de Kaggle
2. Extraer automáticamente las imágenes DRUSEN
3. Organizar todo en la estructura correcta

📖 **Para instrucciones detalladas**, ver: [INSTRUCCIONES_DATASET.md](INSTRUCCIONES_DATASET.md)

### Opción Manual: Dataset Kaggle

El dataset original es **Kermany et al. 2018**:
- **Link**: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
- **Tamaño**: ~5GB
- **Contenido**: Imágenes OCT de múltiples patologías

#### Instrucciones:

1. **Crea una cuenta en Kaggle** (si no tienes)

2. **Descarga el dataset**:
   - Ve al link: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
   - Click en "Download" (necesitas aceptar las reglas)
   - Se descargará `archive.zip` (~5GB)

3. **Extrae SOLO las imágenes DRUSEN**:
   ```powershell
   # Desde el directorio ProGAN
   # Extrae solo la carpeta DRUSEN del zip
   ```

4. **Estructura esperada**:
   ```
   ProGAN/data/OCT2017/train/DRUSEN/
   ├── DRUSEN-1000001-1.jpeg
   ├── DRUSEN-1000002-1.jpeg
   └── ... (más imágenes)
   ```

### Opción B: Dataset Personalizado

Si tienes tus propias imágenes OCT:
1. Organízalas en carpetas por clase
2. Colócalas en `ProGAN/data/OCT2017/train/DRUSEN/`
3. Formato soportado: JPEG, PNG
4. Tamaño recomendado: Al menos 500 imágenes

## 🚀 Paso 2: Entrenar el Modelo

### Configuración Rápida

Edita `progan_local.py` (líneas 32-45) según tus necesidades:

```python
# Para entrenar desde cero
START_TRAIN_AT_IMG_SIZE = 4   # Comienza con resolución 4x4
LOAD_MODEL = False            # No cargar pesos existentes
SAVE_MODEL = True             # Guardar checkpoints

# Para continuar entrenamiento
START_TRAIN_AT_IMG_SIZE = 256 # Empieza en resolución mayor
LOAD_MODEL = True             # Cargar pesos existentes
```

### Ejecutar Entrenamiento

```powershell
# Activar el entorno virtual (si no está activado)
.venv\Scripts\activate

# Entrenar el modelo
python progan_local.py
```

### ⏱️ Tiempo Estimado

| Resolución | GPU (RTX 3060) | CPU (i7) | Epochs |
|-----------|----------------|----------|--------|
| 4x4       | ~5 min         | ~30 min  | 30     |
| 8x8       | ~10 min        | ~1 hora  | 30     |
| 16x16     | ~20 min        | ~2 horas | 30     |
| 32x32     | ~40 min        | ~4 horas | 30     |
| 64x64     | ~1.5 horas     | ~8 horas | 30     |
| 128x128   | ~3 horas       | ~16 horas| 30     |
| 256x256   | ~6 horas       | ~32 horas| 30     |

**Total (4→256)**: ~12-15 horas en GPU, varios días en CPU

## 🖼️ Paso 3: Generar Imágenes

Una vez entrenado (o si tienes pesos pre-entrenados):

```powershell
# Generar 20 imágenes sintéticas
python progan_local.py generate
```

Las imágenes se guardarán en `generated_images/saved_examples/`

## 📊 Monitorear Entrenamiento con TensorBoard

```powershell
# En una terminal separada
tensorboard --logdir=logs
```

Luego abre tu navegador en: http://localhost:6006

Verás:
- Gráficos de pérdida del generador y discriminador
- Imágenes reales vs generadas durante el entrenamiento
- Progreso visual por epoch

## ⚙️ Configuración Avanzada

### Ajustar Hiperparámetros

En `progan_local.py`:

```python
# Batch sizes por resolución (ajusta según tu GPU/RAM)
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]

# Epochs por resolución (más epochs = mejor calidad pero más tiempo)
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)  # 30 epochs por cada resolución

# Learning rate
LEARNING_RATE = 1e-3  # Reduce si el entrenamiento es inestable

# Dimensión del espacio latente
Z_DIM = 256  # Más grande = más variedad (pero más lento)
```

### Reducir Uso de Memoria

Si te quedas sin memoria GPU:

```python
# Reducir batch sizes
BATCH_SIZES = [16, 16, 16, 8, 8, 4, 4, 2, 1]

# Reducir workers
NUM_WORKERS = 0  # Solo en Windows si hay problemas
```

### Entrenar Solo Resoluciones Bajas (Prueba Rápida)

```python
# Solo entrenar hasta 64x64
PROGRESSIVE_EPOCHS = [5, 5, 5, 5, 5]  # Solo 5 primeras resoluciones
BATCH_SIZES = BATCH_SIZES[:5]
```

## 🐛 Solución de Problemas

### Error: "Data directory not found"
```
⚠️ Verifica que el dataset esté en: ProGAN/data/OCT2017/train/DRUSEN/
```

### Error: "CUDA out of memory"
```python
# Reduce batch sizes en progan_local.py
BATCH_SIZES = [8, 8, 8, 4, 4, 2, 2, 1, 1]
```

### Error: "No module named 'torch'"
```powershell
# Reinstala dependencias
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### El entrenamiento es muy lento
- ✅ Verifica que estás usando GPU: El script imprime "Using device: cuda:0"
- ✅ Si dice "cpu", instala PyTorch con CUDA
- ✅ Reduce resoluciones de entrenamiento

### Las imágenes generadas se ven mal
- Necesitas más epochs de entrenamiento
- El dataset podría ser muy pequeño (< 500 imágenes)
- Aumenta el learning rate ligeramente

## 📚 Recursos Adicionales

### Papers
- **ProGAN**: [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196)
- **Dataset**: [Kermany et al. 2018](https://data.mendeley.com/datasets/rscbjbr9sj/2)

### Conceptos Clave
- **Progressive Training**: Entrena empezando con imágenes pequeñas (4x4) y gradualmente aumenta la resolución
- **WGAN-GP**: Usa Wasserstein GAN con Gradient Penalty para estabilidad
- **Pixel Normalization**: Normalización por pixel para evitar escalado de activaciones
- **Equalized Learning Rate**: Escala pesos durante el forward pass

## 📝 Notas Importantes

1. **Este modelo NO hace traducción fundus→OCT**. Para eso necesitas CycleGAN o pix2pix.

2. **El entrenamiento es progresivo**: 
   - Empieza en 4x4 píxeles
   - Aumenta gradualmente a 8→16→32→64→128→256→512 píxeles
   - Cada resolución toma varios epochs

3. **Checkpoints automáticos**: 
   - Se guardan después de cada resolución en `weights/`
   - Puedes continuar el entrenamiento desde cualquier punto

4. **Calidad vs Tiempo**:
   - Para pruebas rápidas: entrena hasta 64x64 (30 min)
   - Para resultados decentes: hasta 128x128 (2 horas)
   - Para mejor calidad: hasta 256x256 (6+ horas)

## 🎓 Para Entender Mejor

### ¿Qué es DRUSEN?
Depósitos amarillentos bajo la retina que aparecen en degeneración macular relacionada con la edad (AMD). Son visibles en imágenes OCT como áreas brillantes bajo el EPR.

### ¿Por qué generar imágenes sintéticas?
- **Privacidad**: No requiere datos reales de pacientes
- **Augmentación**: Más datos para entrenar clasificadores
- **Casos raros**: Generar ejemplos de patologías poco comunes
- **Investigación**: Estudiar variabilidad de la patología

## 🆘 Ayuda

Si tienes problemas, verifica:
1. ✅ Dataset en la ubicación correcta
2. ✅ Suficiente espacio en disco
3. ✅ GPU detectada (si tienes una)
4. ✅ Todas las dependencias instaladas

---

**¡Buena suerte con tu entrenamiento! 🚀**
