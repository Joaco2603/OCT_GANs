@echo off
:: Guía de Comandos para Configurar RTX 3070
:: Este archivo es solo informativo, copia y pega los comandos en PowerShell

echo ================================================================
echo    GUIA DE COMANDOS - Configuracion RTX 3070
echo ================================================================
echo.
echo PASO 1: Instalar Python 3.12
echo ----------------------------------------------------------------
echo Opcion A - Descargar manualmente:
echo    1. Ve a: https://www.python.org/downloads/
echo    2. Descarga Python 3.12.x
echo    3. Instala y marca "Add Python to PATH"
echo.
echo Opcion B - Usar Winget (Windows 11):
echo    winget install Python.Python.3.12
echo.
echo ================================================================
echo PASO 2: Ejecutar Script de Setup
echo ================================================================
echo cd ProGAN
echo .\setup_python312.ps1
echo.
echo Si da error de permisos:
echo    powershell -ExecutionPolicy Bypass -File setup_python312.ps1
echo.
echo ================================================================
echo PASO 3: Verificar GPU
echo ================================================================
echo python check_gpu.py
echo.
echo Deberia mostrar:
echo    - CUDA disponible: 11.8
echo    - GPU: NVIDIA GeForce RTX 3070
echo    - VRAM Total: 8.00 GB
echo.
echo ================================================================
echo PASO 4: Entrenar el Modelo
echo ================================================================
echo python progan_local.py
echo.
echo Para generar imagenes:
echo    python progan_local.py generate
echo.
echo ================================================================
echo COMANDOS ADICIONALES
echo ================================================================
echo Ver temperatura GPU en tiempo real:
echo    nvidia-smi -l 2
echo.
echo Abrir TensorBoard:
echo    tensorboard --logdir=logs
echo    (Abre http://localhost:6006)
echo.
echo Activar entorno virtual:
echo    ..\.venv\Scripts\Activate.ps1
echo.
echo ================================================================
echo IMPORTANTE
echo ================================================================
echo - Tu RTX 3070 tiene 8GB de VRAM
echo - Temperatura segura: 60-80C
echo - Si se calienta mucho: reduce batch sizes
echo - El entrenamiento completo toma 8-12 horas
echo.
echo Lee README_RTX3070.md para mas informacion
echo ================================================================
pause
