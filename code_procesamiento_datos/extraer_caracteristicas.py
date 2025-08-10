import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern, canny, graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.signal import convolve2d

# librerias para ahcer un analisis de las caracteristicas para el dataset y quedarnos con las más relevantes
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# ========= CONFIG =========
DATA_ROOT = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\data_bin"  # train/val/test/sanas|danadas
WIDTH, HEIGHT = 1080, 720   # dimension de las imagenes (ancho x alto)
CENTER_BAND_RATIO = 0.60    # banda central para reducir fondo de las imagenes  

# --- SOLO HSV histograma con hsv ---
HSV_BINS_H = 16     
HSV_BINS_S = 12   

# --- HOG parametros de configuracion ---
# 1080x720 con ppc=(60,60) -> 18x12 celdas -> (17x11)=187 bloques
# cada bloque = orientations * 2 * 2 = 6*4 = 24  -> 187*24 = 4488 features aprox.
HOG_ORI = 6
HOG_PPC = (60, 60)
HOG_CPB = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# LBP (10 bins) parametros de lbp para extraer características
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_P + 2

# GLCM parametros para extraer características
GLCM_DISTANCES = [2, 4]
GLCM_ANGLES = [0, np.pi/2]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

LABEL_MAP = {"danadas": 0, "sanas": 1} # etiquetas para clasificación 

# --- Curado de features ---
VAR_THRESHOLD = 1e-6         # quita columnas casi constantes (se aplica con TRAIN)
CORR_THRESHOLD = 0.95        # umbral para eliminar alta correlación colinear
CORR_SAMPLE_ROWS = 1200      # usa hasta 1200 filas de TRAIN para calcular correlación (acelera)

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
    Configurciones para extraer características de las imágenes:
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

    roi_pil = Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return roi_pil

# -------- FEATURES --------
def hsv_histogram(pil_img, bins_h=HSV_BINS_H, bins_s=HSV_BINS_S):
    rgb = np.asarray(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0].ravel()  # [0,180)
    S = hsv[:,:,1].ravel()  # [0,255]
    h_hist,_ = np.histogram(H, bins=bins_h, range=(0,180), density=True)
    s_hist,_ = np.histogram(S, bins=bins_s, range=(0,256), density=True)
    return np.concatenate([h_hist, s_hist])  # 16 + 12 = 28

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
    # usar uint8 para evitar warning de LBP con floats
    g8 = img_as_ubyte(g)
    lbp = local_binary_pattern(g8, P=LBP_P, R=LBP_R, method=LBP_METHOD)
    hist,_ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return hist

def glcm_stats(pil_img):
    g = rgb2gray(np.asarray(pil_img))
    g8 = img_as_ubyte(g)
    glcm = graycomatrix(g8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=256, symmetric=True, normed=True)
    vals = []
    for prop in GLCM_PROPS:
        v = graycoprops(glcm, prop)  # (len(dist), len(angles))
        vals.append(v.mean())
    return np.array(vals, dtype=float)

def edge_and_focus(pil_img):
    g = rgb2gray(np.asarray(pil_img))
    edges = canny(g, sigma=1.2)
    edge_density = float(edges.mean())
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

                f_hsv = hsv_histogram(roi)           # 28
                f_hog = hog_features(roi)            # ~4.5k
                f_lbp = lbp_hist(roi)                # 10
                f_glcm= glcm_stats(roi)              # 6
                f_edge= edge_and_focus(roi)          # 2

                row = {"file": os.path.basename(fp), "split": split, "label": LABEL_MAP[cls]}
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
    print(f" *****----- Guardado {out_csv} | {len(df)} filas  | {df.shape[1]} columnas")
    return df

def extract_all_splits():
    dfs = []
    for split in ["train", "val", "test"]:
        dfs.append(extract_from_split(split))
    df_all = pd.concat(dfs, ignore_index=True)
    all_csv = os.path.join(DATA_ROOT, "features_all.csv")
    df_all.to_csv(all_csv, index=False)
    print(" *****----- features_all.csv listo:", df_all.shape)
    return df_all

# -------- CURADO: VAR THRESH + PRUNING CORRELACIÓN --------
def curate_by_variance_and_corr(df_train, df_val, df_test, exclude_cols=("file","split","label"),
                                var_threshold=VAR_THRESHOLD, corr_threshold=CORR_THRESHOLD,
                                sample_rows=CORR_SAMPLE_ROWS):

    # 1) columnas numéricas de features
    feat_cols = [c for c in df_train.columns if c not in exclude_cols]
    Xtr = df_train[feat_cols].astype(np.float32)
    Xva = df_val[feat_cols].astype(np.float32)
    Xte = df_test[feat_cols].astype(np.float32)

    # 2) VarianceThreshold en TRAIN
    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(Xtr)
    cols_var = [feat_cols[i] for i, keep in enumerate(vt.get_support()) if keep]
    if len(cols_var) < len(feat_cols):
        print(f" ***** VarianceThreshold: {len(feat_cols)-len(cols_var)} columnas eliminadas por baja varianza.")
    else:
        print(" ***** VarianceThreshold: ninguna columna eliminada.")

    Xtr_v = Xtr[cols_var]
    # 3) Correlación en TRAIN (muestra para acelerar)
    if sample_rows and len(Xtr_v) > sample_rows:
        Xcorr = Xtr_v.sample(n=sample_rows, random_state=42)
    else:
        Xcorr = Xtr_v

    # Corr matrix (Pearson)
    corr = np.corrcoef(Xcorr.values, rowvar=False)
    upper = np.triu_indices_from(corr, k=1)
    to_drop = set()
    for i, j in zip(*upper):
        if abs(corr[i, j]) >= corr_threshold:
            # drop j por ejemplo
            to_drop.add(j)
    cols_corr = [c for k, c in enumerate(cols_var) if k not in to_drop]
    print(f" ---------- Correlación: {len(cols_var)-len(cols_corr)} columnas eliminadas (>|{corr_threshold}|).")
    print(f" ---------- Total columnas finales: {len(cols_corr)}")

    # Aplicar selección a los tres splits
    keep = ["file", "split", "label"] + cols_corr
    tr_cur = df_train[keep].copy()
    va_cur = df_val[keep].copy()
    te_cur = df_test[keep].copy()
    return tr_cur, va_cur, te_cur, cols_corr

def main():
    # 1) Si ya tienes features_{split}.csv creados, puedes saltarte extracción.
    #    Si no, descomenta la siguiente línea:
    # df_all = extract_all_splits()

    # 2) Cargar features existentes
    f_train = os.path.join(DATA_ROOT, "features_train.csv")
    f_val   = os.path.join(DATA_ROOT, "features_val.csv")
    f_test  = os.path.join(DATA_ROOT, "features_test.csv")
    if not (os.path.exists(f_train) and os.path.exists(f_val) and os.path.exists(f_test)):
        print("No se encontraron features por split. Extrayendo ahora...")
        extract_all_splits()

    df_train = pd.read_csv(f_train)
    df_val   = pd.read_csv(f_val)
    df_test  = pd.read_csv(f_test)

    print("\n=== Curado de features  ===")
    tr_cur, va_cur, te_cur, cols_finales = curate_by_variance_and_corr(df_train, df_val, df_test)

    # 3) Guardar curados
    tr_cur.to_csv(os.path.join(DATA_ROOT, "features_train_curado.csv"), index=False)
    va_cur.to_csv(os.path.join(DATA_ROOT, "features_val_curado.csv"), index=False)
    te_cur.to_csv(os.path.join(DATA_ROOT, "features_test_curado.csv"), index=False)

    df_all_cur = pd.concat([tr_cur, va_cur, te_cur], ignore_index=True)
    df_all_cur.to_csv(os.path.join(DATA_ROOT, "features_all_curado.csv"), index=False)

    with open(os.path.join(DATA_ROOT, "selected_columns.txt"), "w", encoding="utf-8") as f:
        for c in cols_finales:
            f.write(c + "\n")

    print("---------- Guardados: ---------- \n")
    print(" - features_train_curado.csv")
    print(" - features_val_curado.csv")
    print(" - features_test_curado.csv")
    print(" - features_all_curado.csv")
    print(" - selected_columns.txt")

if __name__ == "__main__":
    main()
