# Script de instalación de PyTorch con CUDA para RTX 3070
# Ejecutar en PowerShell como administrador

Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 49) -ForegroundColor Green
Write-Host "🚀 Instalación de PyTorch con soporte CUDA para RTX 3070" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 49) -ForegroundColor Green

# Verificar versión de Python
Write-Host "`n📌 Verificando Python..." -ForegroundColor Yellow
python --version

# Desinstalar versión anterior de PyTorch (si existe)
Write-Host "`Desinstalando versiones anteriores de PyTorch..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

# Instalar PyTorch con CUDA 11.8 (compatible con RTX 3070)
Write-Host "`n📦 Instalando PyTorch con CUDA 11.8..." -ForegroundColor Yellow
Write-Host "   Esto puede tomar varios minutos..." -ForegroundColor Gray
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar instalación
Write-Host "`n✅ Verificando instalación..." -ForegroundColor Yellow
python check_gpu.py

Write-Host "`n✨ ¡Instalación completada!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 49) -ForegroundColor Green

# Write-Host "`n📝 Ahora puedes ejecutar:" -ForegroundColor Cyan
# Write-Host '   python progan_local.py' -ForegroundColor White
# Write-Host "`n   o para generar imagenes:" -ForegroundColor White
# Write-Host '   python progan_local.py generate' -ForegroundColor White
