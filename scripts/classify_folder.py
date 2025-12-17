import argparse
import glob
import os
import sys
import time

# Añadir src al path si se ejecuta directamente
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ai_detector.utils import load_image_from_path
from app.services.classification import ClassificationService


def main():
    parser = argparse.ArgumentParser(description="Clasificador de imágenes Real vs IA")
    parser.add_argument(
        "folder",
        nargs="?",
        default="data/raw",
        help="Carpeta con imágenes (default: data/raw)",
    )
    args = parser.parse_args()

    data_dir = args.folder

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))

    if not image_files:
        print(f"❌ No se encontraron imágenes en '{data_dir}'")
        return

    print("🚀 Iniciando servicio de clasificación...")
    service = ClassificationService()  # Entrena al iniciar

    print(f"\n📂 Analizando {len(image_files)} imágenes en '{data_dir}'...")
    print("=" * 80)
    print(
        f"{'ARCHIVO':<30} | {'PREDICCIÓN':<10} | {'REAL':<6} | {'FAKE':<6} | {'TIEMPO'}"
    )
    print("-" * 80)

    results = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            img = load_image_from_path(img_path)

            start = time.time()
            res = service.predict(img)
            duration = (time.time() - start) * 1000

            print(
                f"{filename[:30]:<30} | {res['prediction']:<10} | "
                f"{res['probability_real'] * 100:5.1f}% | {res['probability_fake'] * 100:5.1f}% | "
                f"{duration:4.0f}ms"
            )

            results.append(res)

        except Exception as e:
            print(f"{filename[:30]:<30} | ERROR: {e}")

    print("=" * 80)
    print("✅ Análisis completado.")


if __name__ == "__main__":
    main()
