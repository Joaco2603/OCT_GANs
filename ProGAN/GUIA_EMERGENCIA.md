# 🚨 GUÍA DE EMERGENCIA - RTX 3070

## Si la Temperatura Sube Rápidamente

### 🔥 PROCEDIMIENTO DE DETENCIÓN SEGURA

```
1. Presiona Ctrl+C UNA vez
2. Espera 5-10 segundos (verás "Guardando pesos...")
3. Verifica mensaje: "✅ Pesos guardados exitosamente"
4. Anota el nombre del archivo EMERGENCY_*.pth
5. Cierra la terminal
```

**⚠️ IMPORTANTE**: NO presiones Ctrl+C múltiples veces. Una es suficiente.

---

## 📋 Checklist Post-Detención

Después de detener por temperatura alta:

- [ ] Verifica que existan los archivos `EMERGENCY_*.pth` en `weights/`
- [ ] Espera 10-15 minutos a que GPU se enfríe
- [ ] Limpia ventiladores con aire comprimido
- [ ] Verifica temperatura ambiente
- [ ] Mejora ventilación del case (abre panel lateral si es necesario)

---

## 🔄 Cómo Reanudar Después de Detención

### Paso 1: Verificar Archivos

```powershell
cd ProGAN
dir weights\EMERGENCY_*.pth
```

Deberías ver algo como:
```
EMERGENCY_generator_20251106_153045.pth    (250 MB)
EMERGENCY_critic_20251106_153045.pth       (200 MB)
```

### Paso 2: Modificar Configuración

Abre `progan_local.py` y cambia:

```python
# Línea ~60-65
LOAD_MODEL = True  # Cambiar de False a True

# Línea ~70-75
CHECKPOINT_GEN = str(WEIGHTS_DIR / "EMERGENCY_generator_20251106_153045.pth")
CHECKPOINT_CRITIC = str(WEIGHTS_DIR / "EMERGENCY_critic_20251106_153045.pth")
```

**Usa el nombre exacto de tus archivos EMERGENCY.**

### Paso 3: Reducir Batch Sizes (Opcional)

Si la temperatura fue muy alta (>80°C), reduce batch sizes:

```python
# Línea ~55
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]  # Más conservador
```

### Paso 4: Reanudar Entrenamiento

```powershell
python progan_local.py
```

El entrenamiento continuará desde donde se detuvo.

---

## 🌡️ Temperaturas de Referencia RTX 3070

| Temperatura | Estado | Acción |
|-------------|--------|--------|
| <60°C | 🟢 Excelente | Ninguna |
| 60-70°C | 🟢 Muy bueno | Ninguna |
| 70-75°C | 🟢 Bueno | Monitorear |
| 75-78°C | 🟡 Aceptable | Revisar ventilación |
| 78-80°C | 🟡 Límite recomendado | Mejorar ventilación |
| 80-82°C | 🟠 Alto | **Ctrl+C** recomendado |
| 82-85°C | 🔴 Muy alto | **Ctrl+C** inmediato |
| >85°C | 🔴 Crítico | **DETENER** + revisar hardware |

**Nota**: La GPU se protege automáticamente a 93°C (thermal throttling).

---

## 🛠️ Soluciones por Nivel de Temperatura

### 75-78°C: Optimización Ligera

```python
# En progan_local.py
BATCH_SIZES = [12, 12, 10, 6, 4, 3, 2, 2, 1]
GRADIENT_ACCUMULATION_STEPS = 2
```

**Acción física:**
- Limpia ventiladores
- Abre panel lateral del case
- Verifica que ventiladores giren

---

### 78-82°C: Optimización Media

```python
# En progan_local.py
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]
GRADIENT_ACCUMULATION_STEPS = 4
AUTO_SAVE_EVERY_N_BATCHES = 250  # Guardar más seguido
```

**Acción física:**
- Limpia TODA la GPU con aire comprimido
- Mejora flujo de aire (más ventiladores case)
- Reduce temperatura ambiente (AC)
- Considera undervolting (avanzado)

---

### 82°C+: Optimización Máxima

```python
# En progan_local.py
BATCH_SIZES = [4, 4, 3, 2, 2, 1, 1, 1, 1]
GRADIENT_ACCUMULATION_STEPS = 8
PROGRESSIVE_EPOCHS = [15] * len(BATCH_SIZES)  # Menos épocas
AUTO_SAVE_EVERY_N_BATCHES = 100  # Guardar muy seguido
```

**Acción física:**
- Revisa pasta térmica (si tienes experiencia)
- Considera watercooling (avanzado)
- Entrena solo de noche (temperatura ambiente menor)
- Coloca ventilador externo apuntando al case

---

## 💻 Comandos de Monitoreo

### Monitor en Tiempo Real

```powershell
# Terminal 1: Entrenamiento
python progan_local.py

# Terminal 2: Monitoreo continuo
nvidia-smi -l 1  # Actualiza cada segundo
```

### Monitor Detallado

```powershell
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 2
```

### Registro de Temperatura

```powershell
# Guardar log de temperatura durante entrenamiento
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw --format=csv -l 5 > temp_log.csv
```

Luego puedes ver máximos:
```powershell
Get-Content temp_log.csv | Sort-Object | Select-Object -Last 10
```

---

## 🔍 Diagnóstico de Problemas

### Problema: GPU llega a 80°C+ rápidamente

**Causas posibles:**
1. ❌ Ventiladores de GPU no giran → Revisar configuración MSI Afterburner
2. ❌ Ventiladores case no funcionan → Revisar conexiones
3. ❌ Pasta térmica seca → Cambiar pasta térmica
4. ❌ Polvo acumulado → Limpiar con aire comprimido
5. ❌ Temperatura ambiente alta → Usar AC o entrenar de noche

**Diagnóstico:**
```powershell
# Ver velocidad de ventiladores
nvidia-smi --query-gpu=fan.speed --format=csv
```

---

### Problema: Temperatura estable pero alta (78-80°C)

**Causas posibles:**
1. ⚠️ Batch sizes todavía muy grandes
2. ⚠️ Case mal ventilado
3. ⚠️ GPU con undervoltage muy bajo
4. ⚠️ Temperatura ambiente alta

**Solución:**
- Reduce batch sizes más
- Mejora ventilación
- Entrena de noche

---

### Problema: Temperatura fluctúa mucho

**Causas posibles:**
1. ℹ️ Normal durante cambios de resolución
2. ℹ️ Ventiladores en modo automático
3. ⚠️ Thermal throttling activándose

**Solución:**
- Configura curva de ventiladores más agresiva en MSI Afterburner
- Establece mínimo 60% velocidad de ventiladores

---

## 📊 Ejemplo de Sesión de Entrenamiento Segura

```
Hora    | Temp | Acción
--------|------|----------------------------------
14:00   | 65°C | Inicio entrenamiento
14:30   | 72°C | Normal, continuando
15:00   | 75°C | Monitoreando
15:30   | 78°C | Observando de cerca
15:45   | 80°C | Ctrl+C → Guardado emergencia
15:46   | 80°C | Esperando enfriamiento
16:00   | 55°C | GPU enfriada
16:05   | 55°C | Reduce batch sizes
16:10   | 57°C | Reanuda entrenamiento
16:40   | 70°C | Temperatura estable ✅
17:00   | 72°C | Continuando sin problema ✅
```

---

## 🎯 Configuración Ultra-Safe

Si quieres **temperatura mínima** (sacrificando velocidad):

```python
# progan_local.py - Configuración más fría posible
BATCH_SIZES = [2, 2, 2, 1, 1, 1, 1, 1, 1]
GRADIENT_ACCUMULATION_STEPS = 16  # Simula batches grandes
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
AUTO_SAVE_EVERY_N_BATCHES = 50

# Resultado esperado:
# Temperatura: 55-65°C
# Velocidad: ~40 min/época (vs 18 min normal)
# Uso VRAM: 2-3 GB
```

---

## 📞 Contactos Útiles

### Recursos Hardware

- **MSI Afterburner**: https://www.msi.com/Landing/afterburner
- **HWInfo64**: https://www.hwinfo.com/ (monitoreo detallado)
- **Drivers NVIDIA**: https://www.nvidia.com/Download/index.aspx

### Foros y Ayuda

- r/MachineLearning
- r/NVIDIA
- Stack Overflow (tag: pytorch, cuda)

---

## ✅ Checklist Final

Antes de entrenar overnight:

- [ ] Temperatura estable <75°C en sesión de prueba
- [ ] `nvidia-smi -l 2` corriendo en terminal separada
- [ ] Guardado automático activado (cada 500 batches)
- [ ] Sistema de emergencia Ctrl+C verificado
- [ ] Ventilación del case óptima
- [ ] Temperatura ambiente <25°C
- [ ] Fuente de poder adecuada (>650W)

---

## 🚀 Resumen Ejecutivo

**¿Qué hacer si temperatura sube?**

1. Presiona `Ctrl+C` **UNA vez**
2. Espera mensaje "✅ Pesos guardados"
3. Deja enfriar 15 minutos
4. Reduce batch sizes
5. Reanuda con `LOAD_MODEL = True`

**Tu trabajo está protegido. Ctrl+C guarda automáticamente.** ✅

---

✨ **Mantén la calma, el sistema te protege** ✨
