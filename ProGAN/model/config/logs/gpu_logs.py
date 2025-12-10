"""GPU monitoring and emergency save functionality."""
import sys
import signal
import torch

from ..envs.constansts_envs import WEIGHTS_DIR
from ..checkpoints.save_checkpoint import save_checkpoint

# Variables globales para guardado de emergencia
emergency_save_models = {}


def setup_emergency_save(gen, critic, opt_gen, opt_critic):
    """Configurar guardado de emergencia en caso de Ctrl+C"""
    global emergency_save_models
    emergency_save_models = {
        'gen': gen,
        'critic': critic,
        'opt_gen': opt_gen,
        'opt_critic': opt_critic
    }
    
    def signal_handler(sig, frame):
        print("\n\n" + "=" * 60)
        print("🛑 ¡Ctrl+C detectado! Guardando pesos antes de salir...")
        print("=" * 60)
        
        try:
            # Guardar con timestamp para no sobrescribir
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_gen = str(WEIGHTS_DIR / f"EMERGENCY_generator_{timestamp}.pth")
            emergency_critic = str(WEIGHTS_DIR / f"EMERGENCY_critic_{timestamp}.pth")
            
            save_checkpoint(emergency_save_models['gen'], 
                          emergency_save_models['opt_gen'], 
                          emergency_gen)
            save_checkpoint(emergency_save_models['critic'], 
                          emergency_save_models['opt_critic'], 
                          emergency_critic)
            
            print(f"\n✅ Pesos guardados exitosamente:")
            print(f"   📁 {emergency_gen}")
            print(f"   📁 {emergency_critic}")
            print("\n💡 Puedes reanudar el entrenamiento cargando estos pesos")
            print("   Cambia CHECKPOINT_GEN y CHECKPOINT_CRITIC en el código")
            print("=" * 60)
        except Exception as e:
            print(f"\n❌ Error al guardar: {e}")
            print("⚠️  Los pesos NO se pudieron guardar")
        
        print("\n👋 Saliendo...")
        sys.exit(0)
    
    # Registrar handler para Ctrl+C (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    print("✅ Sistema de guardado de emergencia activado (Ctrl+C guardará pesos)")


def get_gpu_stats():
    """Obtener estadísticas de la GPU si está disponible"""
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Intentar obtener temperatura (requiere nvidia-smi)
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
                    return {
                        'memory_allocated': f"{gpu_memory_allocated:.2f}GB",
                        'memory_total': f"{gpu_memory_total:.2f}GB",
                        'memory_percent': f"{(gpu_memory_allocated/gpu_memory_total)*100:.1f}%",
                        'temperature': f"{temp}°C",
                        'utilization': f"{util}%",
                        'power': f"{power}W"
                    }
            except:
                pass
            
            return {
                'memory_allocated': f"{gpu_memory_allocated:.2f}GB",
                'memory_total': f"{gpu_memory_total:.2f}GB",
                'memory_percent': f"{(gpu_memory_allocated/gpu_memory_total)*100:.1f}%"
            }
        except:
            return None
    return None


def print_gpu_stats():
    """Imprimir estadísticas de GPU de forma legible"""
    stats = get_gpu_stats()
    if stats:
        print("\n" + "=" * 50)
        print("📊 Estadísticas de GPU:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("=" * 50)
