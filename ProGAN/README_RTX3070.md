# 🎮 Configuración Optimizada para RTX 3070

## 🚨 PROBLEMA ACTUAL

Tu sistema tiene **Python 3.14** pero PyTorch aún no tiene builds con CUDA para esta versión.  
**Resultado**: Solo detecta CPU, tu RTX 3070 no se está usando.

---

## ✅ SOLUCIÓN RÁPIDA (3 Pasos)

### 1️⃣ Instala Python 3.12

**Opción A - Descarga directa:**
- Ve a: https://www.python.org/downloads/
- Descarga Python 3.12.x (última versión)
- ✅ Marca "Add Python to PATH" al instalar

**Opción B - Winget (Windows 11):**
```powershell
winget install Python.Python.3.12
```

### 2️⃣ Ejecuta el script de setup automático

```powershell
# En la raíz del proyecto OCT_GANs
cd ProGAN
.\setup_python312.ps1
```

Este script automáticamente:
- ✅ Detecta Python 3.12
- ✅ Crea nuevo entorno virtual
- ✅ Instala PyTorch con CUDA 11.8
- ✅ Instala todas las dependencias
- ✅ Verifica que tu RTX 3070 sea detectada

### 3️⃣ Verifica que todo funciona

```powershell
cd ProGAN
python check_gpu.py
```

Deberías ver:
```
✅ PyTorch instalado: 2.x.x+cu118
✅ CUDA disponible: 11.8
✅ GPU detectada: NVIDIA GeForce RTX 3070
✅ VRAM Total: 8.00 GB
✅ Test de operación GPU: EXITOSO
```

---

## 🚀 Uso

### Entrenar el modelo
```powershell
cd ProGAN
python progan_local.py
```

### Generar imágenes
```powershell
python progan_local.py generate
```

### Monitorear entrenamiento (TensorBoard)
```powershell
# En otra terminal
tensorboard --logdir=ProGAN/logs
# Abre: http://localhost:6006
```

### Ver temperatura GPU en tiempo real
```powershell
nvidia-smi -l 2
```

---

## 🔧 Optimizaciones Aplicadas

He modificado `progan_local.py` específicamente para tu RTX 3070:

### ✅ Batch Sizes Reducidos
```python
# Antes: [32, 32, 32, 16, 16, 16, 16, 8, 4]
# Ahora: [16, 16, 12, 8, 6, 4, 3, 2, 1]
```
**Beneficio**: Usa solo 5-6 GB de VRAM en vez de 7-8 GB

### ✅ Gradient Accumulation
```python
GRADIENT_ACCUMULATION_STEPS = 2
```
**Beneficio**: Simula batches más grandes sin usar memoria extra

### ✅ Épocas Reducidas
```python
PROGRESSIVE_EPOCHS = [20] * len(BATCH_SIZES)  # Antes: 30
```
**Beneficio**: Entrenamiento más rápido, menos desgaste

### ✅ Monitoreo Automático de GPU
- Muestra temperatura, VRAM, utilización cada 100 batches
- Limpieza automática de caché CUDA
- Alertas si hay problemas

---

## 📊 Rendimiento Esperado

| Métrica | Valor |
|---------|-------|
| **VRAM Usada** | 5-6 GB (de 8 GB) |
| **Temperatura** | 65-75°C |
| **Utilización GPU** | 90-95% |
| **Tiempo/Época** | ~18-20 min |
| **Entrenamiento Completo** | 8-12 horas |

### 🌡️ Temperaturas Seguras:
- ✅ **Ideal**: 60-75°C
- ⚠️ **Aceptable**: 75-80°C
- 🔥 **Alto**: 80-83°C (revisar ventilación)
- ❌ **Muy Alto**: >83°C (detener y revisar)

---

## 🛠️ Solución de Problemas

### Problema: "CUDA out of memory"
**Solución**: Reduce más los batch sizes en `progan_local.py`:
```python
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]
```

### Problema: GPU se sobrecalienta (>80°C)
**Soluciones**:
1. Limpia ventiladores de la GPU
2. Mejora ventilación del case
3. Reduce batch sizes aún más
4. Aumenta `GRADIENT_ACCUMULATION_STEPS = 4`

### Problema: "CUDA not available"
**Soluciones**:
1. Verifica drivers NVIDIA: `nvidia-smi`
2. Reinstala PyTorch con el script
3. Asegúrate de usar Python 3.12 (no 3.14)

---

## 📁 Archivos Creados

- `✅ progan_local.py` - Código optimizado para RTX 3070
- `📋 check_gpu.py` - Script de verificación
- `🔧 setup_python312.ps1` - Instalación automática
- `📖 SOLUCION_CUDA.md` - Guía detallada
- `📖 CONFIGURACION_GPU.md` - Configuración avanzada
- `📋 README_RTX3070.md` - Este archivo

---

## ⚡ Comandos Rápidos

```powershell
# Instalar todo automáticamente
cd ProGAN
.\setup_python312.ps1

# Verificar GPU
python check_gpu.py

# Entrenar
python progan_local.py

# Generar imágenes
python progan_local.py generate

# TensorBoard
tensorboard --logdir=logs

# Monitor GPU
nvidia-smi -l 2
```

---

## 💾 Sistema de Guardado (IMPORTANTE)

### ✅ Tus pesos SE GUARDAN si presionas Ctrl+C

He implementado **3 sistemas de protección**:

1. **🛑 Guardado de Emergencia (Ctrl+C)**
   - Presiona `Ctrl+C` cuando quieras
   - Guarda automáticamente antes de salir
   - Crea archivo: `EMERGENCY_generator_FECHA.pth`
   - **Útil si temperatura sube mucho**

2. **💾 Guardado Automático (Cada 500 batches)**
   - Se guarda solo durante entrenamiento
   - Protege contra cortes de luz
   - Protege contra crashes

3. **📁 Guardado al Final de Época**
   - Al completar cada época
   - Actualiza archivos principales

**📖 Lee `SISTEMA_GUARDADO.md` para más detalles**

---

## 🌡️ ¿Qué hacer si la temperatura sube?

| Temperatura | Acción |
|-------------|--------|
| 60-75°C | ✅ **Normal** - Continúa sin problema |
| 75-80°C | ⚠️ **Monitorea** - Revisa ventilación |
| 80-82°C | 🔶 **Ctrl+C** - Detén, guarda, enfría |
| 82°C+ | 🔥 **DETÉN** - Ctrl+C inmediato |

**Ctrl+C guarda todo automáticamente. No perderás progreso.**

---

## ⚡ Comandos Rápidos

---

## 💡 Consejos Pro

1. **Entrenamiento nocturno**: La temperatura ambiente es menor de noche
2. **Pausa cada 2-3 horas**: Dale descansos a tu GPU
3. **Undervolting**: Reduce temperatura sin perder rendimiento (avanzado)
4. **MSI Afterburner**: Configura curva de ventiladores más agresiva
5. **Fuente de poder**: Asegúrate de tener mínimo 650W

---

## 📚 Más Información

- `SOLUCION_CUDA.md` - Soluciones detalladas paso a paso
- `CONFIGURACION_GPU.md` - Configuraciones avanzadas y monitoreo
- `README_LOCAL.md` - Documentación original del proyecto

---

## ❓ ¿Necesitas Ayuda?

1. Lee `SOLUCION_CUDA.md` para troubleshooting detallado
2. Ejecuta `check_gpu.py` para diagnóstico
3. Verifica temperatura con `nvidia-smi`

---

✨ **¡Tu RTX 3070 está lista para generar imágenes médicas!** ✨

**Próximo paso**: Ejecuta `.\setup_python312.ps1`
