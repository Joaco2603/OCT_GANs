# Script para configurar Python 3.12 + PyTorch CUDA automáticamente
# Ejecutar en PowerShell

Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
Write-Host "🔧 Setup Automático: Python 3.12 + PyTorch CUDA para RTX 3070" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green

# Función para verificar si Python 3.12 está instalado
function Get-Python312Path {
    $paths = @(
        "C:\Python312\python.exe",
        "C:\Program Files\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )
    
    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    # Buscar en PATH
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $version = & $pythonCmd --version 2>&1
        if ($version -match "Python 3\.12") {
            return $pythonCmd.Source
        }
    }
    
    return $null
}

Write-Host "`n📌 Paso 1: Verificando Python 3.12..." -ForegroundColor Yellow

$python312 = Get-Python312Path

if ($python312) {
    Write-Host "   ✅ Python 3.12 encontrado en: $python312" -ForegroundColor Green
} else {
    Write-Host "   ❌ Python 3.12 NO encontrado" -ForegroundColor Red
    Write-Host "`n   🔧 Opciones:" -ForegroundColor Yellow
    Write-Host "      1. Descarga e instala Python 3.12 desde:" -ForegroundColor White
    Write-Host "         https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host "      2. O usa Anaconda/Miniconda (ver SOLUCION_CUDA.md)" -ForegroundColor White
    Write-Host "`n   📖 Lee SOLUCION_CUDA.md para instrucciones detalladas" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n📌 Paso 2: Verificando versión de Python..." -ForegroundColor Yellow
$version = & $python312 --version
Write-Host "   $version" -ForegroundColor Green

Write-Host "`n📌 Paso 3: Backup del entorno virtual actual..." -ForegroundColor Yellow
$venvPath = ".\.venv"
if (Test-Path $venvPath) {
    $backupPath = ".\.venv_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Host "   📦 Creando backup en: $backupPath" -ForegroundColor Gray
    Rename-Item -Path $venvPath -NewName $backupPath
    Write-Host "   ✅ Backup creado" -ForegroundColor Green
} else {
    Write-Host "   ℹ️  No hay entorno virtual previo" -ForegroundColor Gray
}

Write-Host "`n📌 Paso 4: Creando nuevo entorno virtual con Python 3.12..." -ForegroundColor Yellow
& $python312 -m venv $venvPath
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Entorno virtual creado" -ForegroundColor Green
} else {
    Write-Host "   ❌ Error al crear entorno virtual" -ForegroundColor Red
    exit 1
}

Write-Host "`n📌 Paso 5: Activando entorno virtual..." -ForegroundColor Yellow
$activateScript = ".\. venv\Scripts\Activate.ps1"
if (Test-Path "$venvPath\Scripts\Activate.ps1") {
    Write-Host "   ✅ Entorno virtual listo para activar" -ForegroundColor Green
} else {
    Write-Host "   ❌ Script de activación no encontrado" -ForegroundColor Red
    exit 1
}

Write-Host "`n📌 Paso 6: Actualizando pip..." -ForegroundColor Yellow
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip | Out-Null
Write-Host "   ✅ pip actualizado" -ForegroundColor Green

Write-Host "`n📌 Paso 7: Instalando PyTorch con CUDA 11.8..." -ForegroundColor Yellow
Write-Host "   ⏳ Esto puede tomar 5-10 minutos (descargando ~2.5 GB)..." -ForegroundColor Gray
& "$venvPath\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ PyTorch instalado con CUDA" -ForegroundColor Green
} else {
    Write-Host "   ❌ Error al instalar PyTorch" -ForegroundColor Red
    Write-Host "   🔧 Intenta manualmente:" -ForegroundColor Yellow
    Write-Host "      .\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor White
    exit 1
}

Write-Host "`n📌 Paso 8: Instalando dependencias adicionales..." -ForegroundColor Yellow
& "$venvPath\Scripts\python.exe" -m pip install opencv-python scipy tqdm tensorboard

if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Dependencias instaladas" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Algunas dependencias fallaron (puedes instalarlas después)" -ForegroundColor Yellow
}

Write-Host "`n📌 Paso 9: Verificando instalación..." -ForegroundColor Yellow
Write-Host ""
& "$venvPath\Scripts\python.exe" "ProGAN\check_gpu.py"

Write-Host "`n" + "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green
Write-Host "✅ ¡CONFIGURACIÓN COMPLETADA!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Green -NoNewline; Write-Host ("=" * 59) -ForegroundColor Green

Write-Host "`n📝 Para usar el entorno:" -ForegroundColor Cyan
Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`n📝 Para entrenar:" -ForegroundColor Cyan
Write-Host "   cd ProGAN" -ForegroundColor White
Write-Host "   python progan_local.py" -ForegroundColor White
Write-Host "`n📝 Para generar imágenes:" -ForegroundColor Cyan
Write-Host "   python progan_local.py generate" -ForegroundColor White
Write-Host "`n📝 Para monitorear entrenamiento:" -ForegroundColor Cyan
Write-Host "   tensorboard --logdir=ProGAN/logs" -ForegroundColor White
Write-Host "`n🎮 Tu RTX 3070 está lista para trabajar!" -ForegroundColor Green
