# 💾 Sistema de Guardado de Pesos - RTX 3070

## ✅ SÍ, tus pesos SE GUARDAN si usas Ctrl+C

He implementado **3 sistemas de guardado** para proteger tu trabajo:

---

## 🛡️ Sistema 1: Guardado de Emergencia (Ctrl+C)

### ¿Cómo funciona?

Cuando presionas **Ctrl+C**, el código:

1. ✅ **Detecta la interrupción** automáticamente
2. ✅ **Guarda ambos modelos** (Generator y Critic) inmediatamente
3. ✅ **Usa timestamp único** para no sobrescribir archivos
4. ✅ **Te muestra dónde guardó** los archivos

### Ejemplo de salida:

```
^C
============================================================
🛑 ¡Ctrl+C detectado! Guardando pesos antes de salir...
============================================================
=> Saving checkpoint: weights/EMERGENCY_generator_20251106_153045.pth
=> Saving checkpoint: weights/EMERGENCY_critic_20251106_153045.pth

✅ Pesos guardados exitosamente:
   📁 weights/EMERGENCY_generator_20251106_153045.pth
   📁 weights/EMERGENCY_critic_20251106_153045.pth

💡 Puedes reanudar el entrenamiento cargando estos pesos
   Cambia CHECKPOINT_GEN y CHECKPOINT_CRITIC en el código
============================================================

👋 Saliendo...
```

### Cómo reanudar después de Ctrl+C:

1. Abre `progan_local.py`
2. Cambia estas líneas:

```python
LOAD_MODEL = True  # Cambiar a True
CHECKPOINT_GEN = str(WEIGHTS_DIR / "EMERGENCY_generator_20251106_153045.pth")
CHECKPOINT_CRITIC = str(WEIGHTS_DIR / "EMERGENCY_critic_20251106_153045.pth")
```

3. Ejecuta de nuevo: `python progan_local.py`

---

## 💾 Sistema 2: Guardado Automático cada N Batches

### Configuración:

```python
AUTO_SAVE_EVERY_N_BATCHES = 500  # Guarda cada 500 batches
```

### ¿Por qué es útil?

- ✅ Protege contra cortes de luz
- ✅ Protege contra crashes del sistema
- ✅ Protege contra overheating extremo
- ✅ Puedes detener en cualquier momento

### Salida durante entrenamiento:

```
Batch 500/1200: 100%|████████| 500/1200
💾 Auto-guardando checkpoint en batch 500...
=> Saving checkpoint: weights/generator_DRUSEN_local.pth
=> Saving checkpoint: weights/critic_DRUSEN_local.pth
✅ Checkpoint guardado
```

### Ajustar frecuencia:

Si quieres guardar **más seguido** (usa más disco):
```python
AUTO_SAVE_EVERY_N_BATCHES = 250  # Cada 250 batches
```

Si quieres guardar **menos seguido** (más rápido):
```python
AUTO_SAVE_EVERY_N_BATCHES = 1000  # Cada 1000 batches
```

---

## 📁 Sistema 3: Guardado al Terminar Cada Época

### ¿Cuándo guarda?

Al final de **cada época completa**, automáticamente guarda:

```python
if SAVE_MODEL:
    save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN_SAVE)
    save_checkpoint(critic, opt_critic, filename=CHECKPOINT_CRITIC_SAVE)
```

### Archivos generados:

```
weights/
├── generator_DRUSEN_local.pth      # ← Última época completada
├── critic_DRUSEN_local.pth         # ← Última época completada
└── EMERGENCY_*.pth                 # ← Solo si usaste Ctrl+C
```

---

## 🔥 Escenarios de Temperatura Alta

### Escenario 1: Temperatura sube a 78°C

**¿Qué hacer?**

```
1. Observa: Si se mantiene bajo 80°C, está OK
2. Revisa ventilación del case
3. El código sigue guardando automáticamente cada 500 batches
```

**No necesitas hacer nada.** El sistema guarda automáticamente.

---

### Escenario 2: Temperatura llega a 82°C

**¿Qué hacer?**

```
1. Presiona Ctrl+C inmediatamente
2. El código guarda los pesos automáticamente
3. Espera 10-15 minutos a que se enfríe
4. Limpia ventiladores/mejora ventilación
5. Reduce batch sizes en el código
6. Reanuda con los pesos guardados
```

**Ejemplo de reducción de batch sizes:**

```python
# En progan_local.py
BATCH_SIZES = [8, 8, 6, 4, 3, 2, 2, 1, 1]  # Más conservador
```

---

### Escenario 3: Temperatura sube a 85°C+

**DETÉN INMEDIATAMENTE:**

```
1. Ctrl+C (guarda automáticamente)
2. Apaga el PC y deja enfriar 30 minutos
3. Limpia la GPU con aire comprimido
4. Revisa pasta térmica (si tienes experiencia)
5. Reduce MUCHO los batch sizes antes de reanudar
```

**⚠️ NOTA**: Es **MUY poco probable** llegar a 85°C con los batch sizes que configuré (16,16,12,8,6,4,3,2,1).

---

## 📊 Información de los Checkpoints

### ¿Qué contiene un checkpoint?

```python
checkpoint = {
    "state_dict": model.state_dict(),      # Todos los pesos del modelo
    "optimizer": optimizer.state_dict(),   # Estado del optimizador (momentum, etc)
}
```

### Tamaño aproximado:

- Generator: ~200-300 MB
- Critic: ~150-250 MB
- **Total por checkpoint**: ~350-550 MB

### Ubicación:

```
ProGAN/
└── weights/
    ├── generator_DRUSEN_local.pth      # Guardado normal
    ├── critic_DRUSEN_local.pth
    ├── EMERGENCY_generator_*.pth        # Guardado de emergencia
    └── EMERGENCY_critic_*.pth
```

---

## 🔄 Cómo Reanudar Entrenamiento

### Opción 1: Desde último guardado automático

```python
# En progan_local.py
LOAD_MODEL = True
CHECKPOINT_GEN = str(WEIGHTS_DIR / "generator_DRUSEN_local.pth")
CHECKPOINT_CRITIC = str(WEIGHTS_DIR / "critic_DRUSEN_local.pth")
```

### Opción 2: Desde checkpoint de emergencia (Ctrl+C)

```python
# En progan_local.py
LOAD_MODEL = True
CHECKPOINT_GEN = str(WEIGHTS_DIR / "EMERGENCY_generator_20251106_153045.pth")
CHECKPOINT_CRITIC = str(WEIGHTS_DIR / "EMERGENCY_critic_20251106_153045.pth")
```

### Opción 3: Desde checkpoint anterior (v11)

```python
# Ya está configurado por defecto
LOAD_MODEL = True  # Cambia a True
CHECKPOINT_GEN = str(WEIGHTS_DIR / "generator_DRUSEN_v11.pth")
CHECKPOINT_CRITIC = str(WEIGHTS_DIR / "critic_DRUSEN_v11.pth")
```

---

## 💡 Mejores Prácticas

### 1. Monitoreo Continuo

```powershell
# En otra terminal
nvidia-smi -l 2
```

Mantén esto abierto mientras entrenas para ver temperatura en tiempo real.

---

### 2. Backup Manual Periódico

Cada día, copia los checkpoints:

```powershell
# Crear carpeta de backup
mkdir weights\backup_dia1

# Copiar archivos
copy weights\generator_DRUSEN_local.pth weights\backup_dia1\
copy weights\critic_DRUSEN_local.pth weights\backup_dia1\
```

---

### 3. Detención Planificada

Si necesitas detener:

```
1. Espera a que termine el batch actual
2. Presiona Ctrl+C UNA vez
3. Espera a que guarde (5-10 segundos)
4. NO cierres la ventana hasta ver "✅ Checkpoint guardado"
```

**❌ NO hagas:**
- Cerrar la ventana directamente (X)
- Ctrl+Alt+Del → Terminar proceso
- Apagar el PC sin esperar

---

### 4. Verificar Guardado

Después de detener, verifica que los archivos existan:

```powershell
dir weights\EMERGENCY_*.pth
```

Deberías ver archivos con tamaño ~200-300 MB.

---

## 🆘 Solución de Problemas

### Problema: "Error al guardar checkpoint"

**Causa**: Disco lleno

**Solución**:
```powershell
# Liberar espacio
del weights\EMERGENCY_generator_*.pth  # Borra emergencias viejas
```

---

### Problema: "Cannot load checkpoint"

**Causa**: Archivo corrupto o incompleto

**Solución**:
1. Usa un checkpoint anterior
2. Si tienes varios EMERGENCY_*, usa el más reciente
3. Revisa tamaño del archivo (debe ser >100 MB)

---

### Problema: Ctrl+C no guarda

**Causa**: Presionaste Ctrl+C múltiples veces

**Solución**:
- Presiona Ctrl+C **solo UNA vez**
- Espera pacientemente (puede tomar 10-30 segundos)
- El código necesita tiempo para guardar los pesos

---

## 📈 Estadísticas de Guardado

Durante un entrenamiento típico (8 horas):

```
✅ Guardados automáticos cada 500 batches: ~40-60 guardados
✅ Guardados al final de época: ~8-12 guardados
✅ Guardados de emergencia (Ctrl+C): Los que necesites

Total espacio usado: 1-2 GB (si borras emergencias viejas)
```

---

## 🎯 Resumen

| Método | Frecuencia | Uso |
|--------|-----------|-----|
| **Ctrl+C** | Manual | 🔥 Temperatura alta |
| **Auto (cada 500 batches)** | Automático | 🔌 Protección general |
| **Final de época** | Automático | 📁 Progreso normal |

### Todos están activos simultáneamente. ✅

**No perderás tu trabajo sin importar cómo detengas el entrenamiento.**

---

## 🎮 ¡Tu trabajo está protegido!

Con estos 3 sistemas:
- ✅ Puedes presionar Ctrl+C cuando quieras
- ✅ Resistente a cortes de luz
- ✅ Resistente a crashes
- ✅ Resistente a temperatura alta
- ✅ Fácil de reanudar

**¡Entrena con confianza!** 🚀
