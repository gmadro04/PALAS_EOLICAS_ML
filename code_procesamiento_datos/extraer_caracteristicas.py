import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import pyfiglet
import cv2
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern, canny, graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.signal import convolve2d

# ========= CONFIG =========
DATA_ROOT = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\data_bin"  # train/val/test/sanas|danadas
WIDTH, HEIGHT = 1080, 720              # (ancho x alto) destino para features
CENTER_BAND_RATIO = 0.60               # banda central para reducir fondo (0.5–0.7 suele ir bien)
BINS_RGB = 32

# HOG (sobre ROI reescalado a WIDTHxHEIGHT)
HOG_ORI = 9
HOG_PPC = (24, 24)     # 1080/24=45, 720/24=30
HOG_CPB = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# LBP
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"  # -> P+2 = 10 bins
LBP_BINS = LBP_P + 2

# GLCM
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

LABEL_MAP = {"danadas": 0, "sanas": 1}  # <- como pediste

# ========= HELPERS =========
def list_images(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in glob(os.path.join(d, "*")) if p.lower().endswith(exts)])

def pil_load_and_size(path, size_wh):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.LANCZOS)
    return im

def gray_world_normalize(img_rgb_u8):
    """Balance de blancos sencillo canal a canal."""
    img = img_rgb_u8.astype(np.float32) + 1e-6
    means = img.reshape(-1,3).mean(axis=0)
    scale = means.mean() / means
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_on_luminance(img_rgb_u8):
    """CLAHE en L (LAB) para estabilizar contraste."""
    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def crop_center_band(pil_img, band_ratio=CENTER_BAND_RATIO):
    """Recorta banda vertical centrada para suprimir fondo."""
    w,h = pil_img.size
    bw = int(max(1, w * band_ratio))
    x0 = (w - bw)//2
    return pil_img.crop((x0, 0, x0+bw, h))

def build_roi(pil_img):
    """
    Parametros y configuraciones a usar para construir la ROI y extraer características:
    - resize fijo (1080x720)
    - banda central para reducir fondo
    - normalización (Gray-World + CLAHE)
    - reescala ROI a 1080x720 para que HOG tenga shape estable
    """
    if pil_img.size != (WIDTH, HEIGHT):
        pil_img = pil_img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

    roi = crop_center_band(pil_img, CENTER_BAND_RATIO)
    rgb = np.asarray(roi.convert("RGB"))
    rgb = gray_world_normalize(rgb)
    rgb = clahe_on_luminance(rgb)

    # regresamos a PIL y normalizamos tamaño final para HOG
    roi_pil = Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return roi_pil

# -------- FEATURES --------
def rgb_histogram(pil_img, bins=BINS_RGB):
    r,g,b = pil_img.split()
    r = np.array(r).ravel(); g = np.array(g).ravel(); b = np.array(b).ravel()
    hr,_ = np.histogram(r, bins=bins, range=(0,256), density=True)
    hg,_ = np.histogram(g, bins=bins, range=(0,256), density=True)
    hb,_ = np.histogram(b, bins=bins, range=(0,256), density=True)
    return np.concatenate([hr,hg,hb])

def hsv_histogram(pil_img, bins_h=24, bins_s=16):
    rgb = np.asarray(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0].ravel()  # [0,180)
    S = hsv[:,:,1].ravel()  # [0,255]
    h_hist,_ = np.histogram(H, bins=bins_h, range=(0,180), density=True)
    s_hist,_ = np.histogram(S, bins=bins_s, range=(0,256), density=True)
    return np.concatenate([h_hist, s_hist])  # 40

def hog_features(pil_img):
    g = rgb2gray(np.asarray(pil_img))  # [0,1]
    feats = hog(
        g, orientations=HOG_ORI,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True
    )
    return feats

def lbp_hist(pil_img):
    g = rgb2gray(np.asarray(pil_img))
    lbp = local_binary_pattern(g, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    hist,_ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return hist

def glcm_stats(pil_img):
    g = rgb2gray(np.asarray(pil_img))
    g8 = img_as_ubyte(g)
    glcm = graycomatrix(g8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=256, symmetric=True, normed=True)
    vals = []
    for prop in GLCM_PROPS:
        v = graycoprops(glcm, prop)
        vals.append(v.mean())
    return np.array(vals, dtype=float)

def edge_and_focus(pil_img):
    g = rgb2gray(np.asarray(pil_img))
    edges = canny(g, sigma=1.2)
    edge_density = edges.mean()
    lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    lap = convolve2d(g, lap_kernel, mode="same", boundary="symm")
    lap_var = float(lap.var())
    return np.array([edge_density, lap_var], dtype=float)

# -------- EXTRACCIÓN POR SPLIT --------
def extract_from_split(split):
    rows = []
    for cls in ["danadas", "sanas"]:
        cls_dir = os.path.join(DATA_ROOT, split, cls)
        files = list_images(cls_dir)
        print(f"[{split}/{cls}] {len(files)} imágenes")
        for fp in tqdm(files):
            try:
                pil_orig = pil_load_and_size(fp, (WIDTH, HEIGHT))
                roi = build_roi(pil_orig)            # PIL, (WIDTH, HEIGHT)

                f_rgb = rgb_histogram(roi)           # 96
                f_hsv = hsv_histogram(roi)           # 40
                f_hog = hog_features(roi)
                f_lbp = lbp_hist(roi)                # 10
                f_glcm= glcm_stats(roi)              # 6
                f_edge= edge_and_focus(roi)          # 2

                row = {"file": os.path.basename(fp), "split": split, "label": LABEL_MAP[cls]}
                for i,v in enumerate(f_rgb):  row[f"rgb_{i}"]  = float(v)
                for i,v in enumerate(f_hsv):  row[f"hsv_{i}"]  = float(v)
                for i,v in enumerate(f_hog):  row[f"hog_{i}"]  = float(v)
                for i,v in enumerate(f_lbp):  row[f"lbp_{i}"]  = float(v)
                for i,v in enumerate(f_glcm): row[f"glcm_{i}"] = float(v)
                row["edge_density"]  = float(f_edge[0])
                row["laplacian_var"] = float(f_edge[1])
                rows.append(row)
            except Exception as e:
                print(" X :c ----- Error con", fp, "->", e)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_ROOT, f"features_{split}.csv")
    df.to_csv(out_csv, index=False)
    print(f"---------- Guardado {out_csv} | {len(df)} filas  | {df.shape[1]} columnas")
    return df

def main():
    dfs = []
    for split in ["train", "val", "test"]:
        dfs.append(extract_from_split(split))
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(DATA_ROOT, "features_all.csv"), index=False)
    print(pyfiglet.print_figlet("Features Extracted", font="slant"))
    print("---------- features_all.csv listo: \n", df_all.shape)

if __name__ == "__main__":
    main()