import os
from PIL import Image
import json

# ====== CONFIGURACIÓN ======}
"""Esta ruta procesa imágenes sin etiquetas asociadas a JSON, redimensionándolas y guardándolas en una carpeta específica.
la ruta donde esta el data set original es diferrente a la ruta de salida donde se guardan las imágenes procesadas.

Este script unicamente separa las imágenes que no tienen un archivo JSON asociado, redimensionándolas y guardándolas en una 
carpeta específica, pero este set aún no esta listo para realizar el entrenamiento.

"""
# Rutas de los datasets
DATASETS = [
    r"C:\Users\GMADRO04\Documents\PROYECTOML\3_blade_1_15_with_labeldata",
    r"C:\Users\GMADRO04\Documents\PROYECTOML\3_blade_16_30_with_labeldata"
]
# Ruta de salida para las imágenes sin etiquetas
OUTPUT_DIR = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\no_etiquetadas"  # Nueva carpeta de salida

#Redimensionamiento de imágenes 
TARGET_SIZE = (1080, 720)

# Función para redimensionar imágenes
def redimensionar_imagen(ruta_img): 
    img = Image.open(ruta_img).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    return img

# ====== PROCESAMIENTO ======
"""Esta función examina cada dataset, busca imágenes sin archivos JSON asociados, las redimensiona 
y las guarda en la carpeta de salida."""
def procesar_imagenes_sin_json():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    lista_imagenes = []
    contador = 0

    for dataset_path in DATASETS:
        for blade_folder in os.listdir(dataset_path):
            blade_path = os.path.join(dataset_path, blade_folder)
            if not os.path.isdir(blade_path):
                continue

            for subfolder in os.listdir(blade_path):
                sub_path = os.path.join(blade_path, subfolder)
                if not os.path.isdir(sub_path):
                    continue

                archivos = os.listdir(sub_path)
                for archivo in archivos:
                    if archivo.endswith(".jpg"):
                        base_name = os.path.splitext(archivo)[0]
                        json_path = os.path.join(sub_path, base_name + ".json")

                        # Solo procesa si NO tiene JSON asociado
                        if not os.path.exists(json_path):
                            try:
                                img_path = os.path.join(sub_path, archivo)
                                img_resized = redimensionar_imagen(img_path)

                                nombre_salida = f"{os.path.basename(dataset_path)}_{blade_folder}_{subfolder}_{archivo}"
                                ruta_salida = os.path.join(OUTPUT_DIR, nombre_salida)
                                img_resized.save(ruta_salida)

                                lista_imagenes.append(nombre_salida)
                                contador += 1
                            except Exception as e:
                                print(f"Error procesando {archivo}: {e}")

    print(f"\n Total imágenes NO etiquetadas procesadas: {contador}")
    
    # Guardar lista como JSON
    with open(os.path.join(OUTPUT_DIR, "imagenes_no_etiquetadas.json"), "w") as f:
        json.dump(lista_imagenes, f, indent=2)

    return lista_imagenes

# ====== EJECUCIÓN ======
if __name__ == "__main__":
    imagenes = procesar_imagenes_sin_json()
