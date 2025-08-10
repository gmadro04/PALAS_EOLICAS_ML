import os, random
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np
from PIL import Image, ImageOps
import pyfiglet

# === CONFIG ===
SANA_SRC   = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\B_sanas_pros"   # carpeta sanas
DANADA_SRC = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\A_defectos_pros" # carpeta dañadas
OUT_ROOT   = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\data_bin"       # se creará: train/val/test/sanas|danadas

# Tamaño deseado (ANCHO x ALTO)
WIDTH, HEIGHT = 1080, 720

# Para PIL: (W,H) ; Para Keras: (H,W)
# Esto se hace para que PIL y Keras tengan el mismo tamaño de imagen y evitar problemas de dimensiones en el entrenamiento.
IMG_SIZE_PIL = (WIDTH, HEIGHT)
IMG_SIZE_KERAS = (HEIGHT, WIDTH)

SEED = 42
SPLIT = (0.8, 0.10, 0.10)  # train / val / test
TARGET_PER_CLASS_TRAIN = 650 # aumentar solo hasta cubrir esta cantidad en el set de entrenamiento

# Augmentation para entrenamiento configuraciones
train_aug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=8,
    zoom_range=[0.5, 1.0],
    brightness_range=(0.85, 1.15),
    horizontal_flip=False,
    fill_mode='nearest'
)

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def list_images(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in glob(os.path.join(d, "*")) if p.lower().endswith(exts)])

def copy_and_resize(src_path, dst_path):
    """
    Copia normalizando orientación EXIF, convirtiendo a RGB,
    y redimensionando a 1080x720 (ancho x alto).
    """
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)  # corrige orientación EXIF
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE_PIL, Image.Resampling.LANCZOS)
        im.save(dst_path, quality=95)

def strat_copy(src_files, dst_class_root):
    """Copia a train/val/test para UNA clase, redimensionando a 1080x720."""
    tr_p = os.path.join(OUT_ROOT, "train", dst_class_root)
    va_p = os.path.join(OUT_ROOT, "val",   dst_class_root)
    te_p = os.path.join(OUT_ROOT, "test",  dst_class_root)
    for p in [tr_p, va_p, te_p]: ensure_dir(p)

    # split 80/10/10
    train_files, rest = train_test_split(src_files, test_size=1-SPLIT[0], random_state=SEED, shuffle=True)
    rel = SPLIT[1] / (SPLIT[1] + SPLIT[2])
    val_files, test_files = train_test_split(rest, test_size=1-rel, random_state=SEED, shuffle=True)

    def _copy(files, dst):
        for s in files:
            dst_path = os.path.join(dst, os.path.basename(s))
            copy_and_resize(s, dst_path)

    _copy(train_files, tr_p); _copy(val_files, va_p); _copy(test_files, te_p)
    return len(train_files), len(val_files), len(test_files)

def augment_to_target(train_dir, target_count):
    """
    Aumenta SOLO lo que falta para alcanzar target_count
    manteniendo 1080x720.
    """
    originals = list_images(train_dir)
    n_now = len(originals)
    if n_now >= target_count:
        print(f"[{os.path.basename(train_dir)}] Ya hay {n_now} >= {target_count}. No se aumenta.")
        return

    need = target_count - n_now
    print(f"[{os.path.basename(train_dir)}] Aumentando {need} imágenes (de {n_now} a {target_count}).")

    random.seed(SEED)
    i = 0
    while need > 0:
        src = random.choice(originals)
        # Keras usa (alto, ancho)
        img = load_img(src, target_size=IMG_SIZE_KERAS)
        x = img_to_array(img)[None, ...]  # (1, H, W, C)

        x_aug = next(train_aug.flow(x, batch_size=1))[0].astype(np.uint8)
        # Convertimos a PIL y aseguramos tamaño (debería salir ya HxW correcto)
        aug_img = array_to_img(x_aug)  # PIL
        # Por robustez, forzamos tamaño final exacto W×H
        aug_img = aug_img.resize(IMG_SIZE_PIL, Image.Resampling.LANCZOS)

        base, _ = os.path.splitext(os.path.basename(src))
        aug_name = f"{base}_aug_{i:05d}.jpg"
        aug_img.save(os.path.join(train_dir, aug_name), quality=95)
        i += 1
        need -= 1

def main():
    # Estructura
    for split in ["train", "val", "test"]:
        for cls in ["sanas", "danadas"]:
            ensure_dir(os.path.join(OUT_ROOT, split, cls))

    sanas_all   = list_images(SANA_SRC)
    danadas_all = list_images(DANADA_SRC)
    print(f"----- Sanas curadas: {len(sanas_all)} | Dañadas curadas: {len(danadas_all)}")

    tS, vS, teS = strat_copy(sanas_all,   "sanas")
    tD, vD, teD = strat_copy(danadas_all, "danadas")
    print(f"----- Sanas  -> train:{tS} val:{vS} test:{teS}")
    print(f"----- Dañadas-> train:{tD} val:{vD} test:{teD}")

    # Aumentar SOLO lo faltante para alcanzar TARGET_PER_CLASS_TRAIN
    train_sanas   = os.path.join(OUT_ROOT, "train", "sanas")
    train_danadas = os.path.join(OUT_ROOT, "train", "danadas")
    augment_to_target(train_sanas,   TARGET_PER_CLASS_TRAIN)
    augment_to_target(train_danadas, TARGET_PER_CLASS_TRAIN)

    print(pyfiglet.figlet_format("Data set listo", font="digital"))
    print(" ---------- Directorio de salida:", OUT_ROOT)
    print(f" ---------- Tamaño estándar final: {WIDTH}x{HEIGHT} (ancho x alto)")
    print(" ---------- Nota: val/test no fueron aumentados.")

if __name__ == "__main__":
    main()
