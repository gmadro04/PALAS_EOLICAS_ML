import os
import json
from PIL import Image
import shutil

# ====== CONFIGURACIÓN ======
"""Esta ruta procesa imágenes asociadas a JSON que se entiende como defectuosas, redimensionándolas y guardándolas en una carpeta específica.
la ruta donde esta el data set original es diferente a la ruta de salida donde se guardan las imágenes procesadas.

Este script unicamente separa las imágenes que tienen un archivo JSON asociado, redimensionándolas y guardándolas en una 
carpeta específica, pero este set aún no esta listo para realizar el entrenamiento.

"""
# Directorios de los datos originales
DATASETS = [
    r"C:\Users\GMADRO04\Documents\PROYECTOML\3_blade_1_15_with_labeldata",
    r"C:\Users\GMADRO04\Documents\PROYECTOML\3_blade_16_30_with_labeldata"
]
# Directorio de salida para las imágenes defectuosas
OUTPUT_DIR = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\defectuosas" # Directorio de salida solo para imágenes defectuosas
TARGET_SIZE = (1080, 720) # nueva dimension de la imagen para optimizar procesamiento

# Funcion que usa PIL para redimensionar imagenes
def redimensionar_imagen(ruta_img): 
    img = Image.open(ruta_img)
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS,)
    return img

# ====== PROCESAMIENTO ======
# Procesar imágenes defectuosas y asociar con JSON para determinar si son defectuosas
def procesar_imagenes_con_json():
    # Verifica si el directorio de salida existe, si no lo crea
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    datos_json = []  # Lista para almacenar los datos de las imágenes y sus JSON asociados
    contador = 0     # Contador de imágenes procesadas

    # Recorre cada dataset especificado en DATASETS
    for dataset_path in DATASETS:
        # Recorre cada carpeta de blade dentro del dataset
        for blade_folder in os.listdir(dataset_path):
            blade_path = os.path.join(dataset_path, blade_folder)
            if not os.path.isdir(blade_path):
                continue  # Si no es una carpeta, la ignora

            # Recorre cada subcarpeta dentro de la carpeta de blade
            for subfolder in os.listdir(blade_path):
                sub_path = os.path.join(blade_path, subfolder)
                if not os.path.isdir(sub_path):
                    continue  # Si no es una carpeta, la ignora

                archivos = os.listdir(sub_path)
                # Recorre cada archivo en la subcarpeta
                for archivo in archivos:
                    if archivo.endswith(".jpg"):  # Solo procesa archivos de imagen JPG
                        base_name = os.path.splitext(archivo)[0]
                        json_path = os.path.join(sub_path, base_name + ".json")

                        # Verifica si existe el archivo JSON asociado a la imagen
                        if os.path.exists(json_path):
                            img_path = os.path.join(sub_path, archivo)
                            try:
                                # Redimensiona la imagen usando la función definida
                                img_resized = redimensionar_imagen(img_path)

                                # Genera el nombre de salida incluyendo información de origen
                                nombre_salida = f"{os.path.basename(dataset_path)}_{blade_folder}_{subfolder}_{archivo}"
                                ruta_salida = os.path.join(OUTPUT_DIR, nombre_salida)
                                img_resized.save(ruta_salida)  # Guarda la imagen redimensionada

                                # Lee el archivo JSON asociado
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                                # Guarda la información de la imagen y su JSON en la lista
                                datos_json.append({
                                    "img_file": nombre_salida,
                                    "json_data": data,
                                    "label": 1
                                })

                                contador += 1  # Incrementa el contador de imágenes procesadas
                            except Exception as e:
                                print(f" ----- X ---- Error procesando {archivo}: {e} ----- X ---- ")

    # Muestra el total de imágenes procesadas
    print(f"\nTotal imágenes defectuosas procesadas: {contador}")
    return datos_json

# ====== EJECUCIÓN ======
#  Procesar las imágenes y guardar los datos
if __name__ == "__main__":
    metadata = procesar_imagenes_con_json()
    # Guardar metadatos para análisis posterior
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
