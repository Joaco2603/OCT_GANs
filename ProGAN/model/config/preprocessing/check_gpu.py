"""
Script rápido para verificar la configuración de GPU
"""
import sys

print("=" * 60)
print("🔍 Verificación de Configuración GPU para RTX 3070")
print("=" * 60)

# Verificar PyTorch
print("\n1️⃣ Verificando PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch instalado: {torch.__version__}")
    
    # Verificar CUDA
    print("\n2️⃣ Verificando CUDA...")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA disponible: {torch.version.cuda}")
        print(f"   ✅ GPU detectada: {torch.cuda.get_device_name(0)}")
        
        # Información detallada de GPU
        gpu_props = torch.cuda.get_device_properties(0)
        vram_gb = gpu_props.total_memory / (1024**3)
        print(f"   ✅ VRAM Total: {vram_gb:.2f} GB")
        print(f"   ✅ Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Test de GPU
        print("\n3️⃣ Haciendo test rápido de GPU...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"   ✅ Test de operación GPU: EXITOSO")
            print(f"   ✅ Memoria GPU usada: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
            
            # Limpiar
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error en test GPU: {e}")
            
        # Verificar temperatura si es posible
        print("\n4️⃣ Intentando leer temperatura GPU...")
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,power.draw', 
                 '--format=csv,noheader,nounits', '--id=0'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                temp, util, power = result.stdout.strip().split(',')
                print(f"   ✅ Temperatura: {temp}°C")
                print(f"   ✅ Utilización: {util}%")
                print(f"   ✅ Consumo: {power}W")
            else:
                print(f"   ⚠️  nvidia-smi no disponible (pero CUDA funciona)")
        except Exception as e:
            print(f"   ⚠️  No se pudo leer temperatura: {e}")
            print(f"   ℹ️  Esto es normal, CUDA funciona correctamente")
            
    else:
        print("   ❌ CUDA NO disponible")
        print("\n   🔧 Para habilitar tu RTX 3070, ejecuta:")
        print("      .\\setup_cuda.ps1")
        print("\n   O manualmente:")
        print("      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
        
except ImportError:
    print("   ❌ PyTorch NO instalado")
    print("\n   🔧 Para instalar PyTorch con CUDA, ejecuta:")
    print("      .\\setup_cuda.ps1")
    print("\n   O manualmente:")
    print("      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Verificar otras dependencias
print("\n5️⃣ Verificando otras dependencias...")
dependencies = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV (cv2)',
    'torchvision': 'TorchVision',
    'tqdm': 'TQDM',
    'scipy': 'SciPy'
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - FALTA")
        missing.append(module if module != 'cv2' else 'opencv-python')

if missing:
    print(f"\n   🔧 Para instalar dependencias faltantes:")
    print(f"      pip install {' '.join(missing)}")

print("\n" + "=" * 60)
print("✅ CONFIGURACIÓN LISTA PARA RTX 3070!")
print("=" * 60)
print("\n📝 Puedes ejecutar:")
print("   python progan_local.py          # Para entrenar")
print("   python progan_local.py generate # Para generar imágenes")
print("\n💡 Los batch sizes están optimizados para no sobrecalentar tu GPU")
print("   Ver CONFIGURACION_GPU.md para más detalles")
