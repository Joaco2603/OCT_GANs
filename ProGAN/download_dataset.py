"""
Script helper para descargar y preparar el dataset OCT DRUSEN
"""

import os
import zipfile
from pathlib import Path
import shutil

def setup_directories():
    """Crear estructura de directorios"""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "OCT2017" / "train" / "DRUSEN"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directorio creado: {data_dir}")
    return data_dir

def extract_drusen_from_zip(zip_path, data_dir):
    """Extraer solo imágenes DRUSEN del archivo zip"""
    if not Path(zip_path).exists():
        print(f"❌ Archivo no encontrado: {zip_path}")
        return False
    
    print(f"📦 Extrayendo imágenes DRUSEN de {zip_path}...")
    print("⏳ Esto puede tomar varios minutos...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Buscar archivos que contengan DRUSEN en el path
            drusen_files = [f for f in zip_ref.namelist() 
                          if 'DRUSEN' in f and f.endswith(('.jpeg', '.jpg', '.png'))]
            
            if not drusen_files:
                print("❌ No se encontraron imágenes DRUSEN en el archivo zip")
                return False
            
            print(f"📸 Encontradas {len(drusen_files)} imágenes DRUSEN")
            
            # Extraer solo los archivos DRUSEN
            for file_path in drusen_files:
                # Obtener solo el nombre del archivo
                filename = Path(file_path).name
                
                # Extraer a un directorio temporal
                zip_ref.extract(file_path, path=Path(__file__).parent / "temp")
                
                # Mover al directorio correcto
                src = Path(__file__).parent / "temp" / file_path
                dst = data_dir / filename
                
                if src.exists():
                    shutil.move(str(src), str(dst))
            
            # Limpiar directorio temporal
            temp_dir = Path(__file__).parent / "temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            print(f"✅ {len(drusen_files)} imágenes extraídas exitosamente!")
            return True
            
    except Exception as e:
        print(f"❌ Error al extraer: {e}")
        return False

def count_images(data_dir):
    """Contar imágenes en el directorio"""
    extensions = ('.jpeg', '.jpg', '.png')
    images = [f for f in data_dir.iterdir() if f.suffix.lower() in extensions]
    return len(images)

def main():
    print("="*60)
    print("  Preparación del Dataset OCT DRUSEN")
    print("="*60)
    
    # Crear directorios
    data_dir = setup_directories()
    
    # Verificar si ya hay imágenes
    existing_images = count_images(data_dir)
    if existing_images > 0:
        print(f"\n📸 Ya hay {existing_images} imágenes en el directorio")
        response = input("¿Deseas continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            print("❌ Operación cancelada")
            return
    
    print("\n" + "="*60)
    print("  INSTRUCCIONES")
    print("="*60)
    print("\n1. Descarga el dataset Kermany2018 de Kaggle:")
    print("   🔗 https://www.kaggle.com/datasets/paultimothymooney/kermany2018")
    print("\n2. El archivo descargado se llama 'archive.zip' (~5GB)")
    print("\n3. Colócalo en la carpeta ProGAN/ (la misma donde está este script)")
    print("\n4. Vuelve a ejecutar este script con el path del zip:")
    print("   python download_dataset.py <ruta_al_zip>")
    
    # Intentar encontrar el archivo zip automáticamente
    base_dir = Path(__file__).parent
    possible_paths = [
        base_dir / "archive.zip",
        base_dir / "kermany2018.zip",
        base_dir / "OCT2017.zip",
    ]
    
    zip_path = None
    for path in possible_paths:
        if path.exists():
            print(f"\n✅ Encontrado: {path}")
            zip_path = path
            break
    
    if zip_path:
        response = input(f"\n¿Deseas extraer las imágenes de {zip_path.name}? (s/n): ")
        if response.lower() == 's':
            if extract_drusen_from_zip(zip_path, data_dir):
                final_count = count_images(data_dir)
                print(f"\n✅ ¡Listo! Tienes {final_count} imágenes DRUSEN en {data_dir}")
                print("\n🚀 Ahora puedes ejecutar: python progan_local.py")
            else:
                print("\n❌ Hubo un problema al extraer las imágenes")
    else:
        print("\n⚠️  No se encontró ningún archivo zip en el directorio")
        print("\n📥 Por favor descarga el dataset y colócalo aquí:")
        print(f"   {base_dir}")
        print("\nLuego ejecuta este script nuevamente.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Si se proporciona un path como argumento
        zip_path = Path(sys.argv[1])
        if zip_path.exists():
            data_dir = setup_directories()
            extract_drusen_from_zip(zip_path, data_dir)
        else:
            print(f"❌ Archivo no encontrado: {zip_path}")
    else:
        main()
