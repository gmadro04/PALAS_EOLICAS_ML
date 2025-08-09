# extract_features_csv.py
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

# === CONFIG ===
DATA_ROOT   = r"C:\...\data_bin"    # salida del script anterior
IMG_SIZE    = (500, 500)
BINS        = 32
LABEL_MAP   = {"danadas": 0, "sanas": 1}   # <<-- como pediste

# HOG params
HOG_ORI = 9
HOG_PPC = (16,16)       # pixels_per_cell
HOG_CPB = (2,2)         # cells_per_block
HOG_BLOCK_NORM = "L2-Hys"

def list_images(d):
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    return sorted([p for p in glob(os.path.join(d, "*")) if p.lower().endswith(exts)])

def rgb_hist(img_pil, bins=BINS):
    img = img_pil.resize(IMG_SIZE).convert("RGB")
    r,g,b = img.split()
    r = np.array(r).ravel(); g = np.array(g).ravel(); b = np.array(b).ravel()
    hr,_ = np.histogram(r, bins=bins, range=(0,256), density=True)
    hg,_ = np.histogram(g, bins=bins, range=(0,256), density=True)
    hb,_ = np.histogram(b, bins=bins, range=(0,256), density=True)
    return np.concatenate([hr,hg,hb])

def hog_feats(img_pil):
    img = img_pil.resize(IMG_SIZE).convert("RGB")
    g = rgb2gray(np.array(img))  # [0,1]
    feats = hog(
        g, orientations=HOG_ORI,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True
    )
    return feats

def extract_split(split):
    rows = []
    for cls in ["danadas","sanas"]:
        cls_dir = os.path.join(DATA_ROOT, split, cls)
        files = list_images(cls_dir)
        for fp in files:
            try:
                pil = Image.open(fp)
                h = rgb_hist(pil)
                f = hog_feats(pil)
                feats = np.concatenate([h, f]).astype(float)
                rows.append({
                    "file": os.path.basename(fp),
                    "split": split,
                    "label": LABEL_MAP[cls],
                    **{f"bin_{i}": v for i, v in enumerate(h)},
                    **{f"hog_{i}": v for i, v in enumerate(f)}
                })
            except Exception as e:
                print("Error con", fp, "->", e)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_ROOT, f"features_{split}.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Guardado {out_csv} | {len(df)} filas")
    return df

def main():
    dfs = []
    for split in ["train","val","test"]:
        dfs.append(extract_split(split))
    # opcional: combinado
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(DATA_ROOT, "features_all.csv"), index=False)
    print("📦 features_all.csv listo.")

if __name__ == "__main__":
    main()
