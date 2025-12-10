# 📥 Instrucciones para Descargar el Dataset

## Opción 1: Descarga desde Kaggle (Recomendada)

### Paso 1: Crear cuenta en Kaggle
1. Ve a [Kaggle.com](https://www.kaggle.com/)
2. Crea una cuenta gratuita si no tienes una

### Paso 2: Descargar el dataset
1. Ve al dataset: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
2. Click en el botón **"Download"** (⚠️ El archivo pesa ~5GB)
3. El archivo descargado se llama `archive.zip`

### Paso 3: Colocar el archivo
1. Mueve el archivo `archive.zip` a la carpeta:
   ```
   OCT_GANs\ProGAN\
   ```
2. **NO lo descomprimas manualmente**, el script lo hará automáticamente

### Paso 4: Ejecutar el script de preparación
```powershell
cd "c:\Users\joaco\Documents\Programming\OCT\preexisting_repositorys\OCT_GANs\ProGAN"
python download_dataset.py
```

El script:
- ✅ Detectará automáticamente el archivo zip
- ✅ Extraerá SOLO las imágenes DRUSEN (~8k imágenes)
- ✅ Las organizará en la estructura correcta
- ✅ Limpiará archivos temporales

---

## Opción 2: Usar Kaggle API (Más rápido)

### Requisitos:
1. Tener cuenta en Kaggle
2. Crear un API token

### Pasos:

1. **Obtener credenciales de Kaggle**:
   - Ve a https://www.kaggle.com/settings
   - Scroll hasta "API" 
   - Click en "Create New API Token"
   - Se descargará un archivo `kaggle.json`

2. **Configurar Kaggle API**:
   ```powershell
   # Instalar kaggle
   pip install kaggle
   
   # Crear directorio para credenciales (si no existe)
   mkdir $env:USERPROFILE\.kaggle -Force
   
   # Copiar el archivo kaggle.json al directorio
   Copy-Item "C:\ruta\donde\descargaste\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
   ```

3. **Descargar dataset automáticamente**:
   ```powershell
   cd "c:\Users\joaco\Documents\Programming\OCT\preexisting_repositorys\OCT_GANs\ProGAN"
   
   # Descargar dataset
   kaggle datasets download -d paultimothymooney/kermany2018
   
   # Ejecutar script de preparación
   python download_dataset.py
   ```

---

## 🔍 Verificar que todo está listo

Después de ejecutar el script, deberías ver:
```
✅ ¡Listo! Tienes XXXX imágenes DRUSEN en C:\...\ProGAN\data\OCT2017\train\DRUSEN
```

La estructura de directorios será:
```
ProGAN/
├── data/
│   └── OCT2017/
│       └── train/
│           └── DRUSEN/
│               ├── imagen1.jpeg
│               ├── imagen2.jpeg
│               └── ...
├── download_dataset.py
└── progan_local.py
```

---

## ⚠️ Solución de Problemas

### "No se encontró ningún archivo zip"
- Verifica que el archivo `archive.zip` esté en la carpeta `ProGAN/`
- El nombre puede ser `archive.zip`, `kermany2018.zip` o `OCT2017.zip`

### "Error al extraer"
- Verifica que el archivo zip no esté corrupto
- Vuelve a descargarlo si es necesario
- Asegúrate de tener suficiente espacio en disco (~10GB libres)

### "No se encontraron imágenes DRUSEN"
- El script busca automáticamente carpetas con "DRUSEN" en el nombre
- Verifica que descargaste el dataset correcto

---

## 🚀 Siguiente Paso

Una vez que tengas el dataset preparado, ejecuta:
```powershell
python progan_local.py
```

Esto iniciará el entrenamiento o generación de imágenes según tu configuración.
