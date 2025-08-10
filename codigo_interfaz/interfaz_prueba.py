import os, io, glob, json
import numpy as np
import pandas as pd
import joblib
from PIL import Image, ImageOps, ImageDraw
import streamlit as st
import cv2
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern, canny, graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.signal import convolve2d

# =================== RUTAS DE LOS DATOS Y DIRECTORIOS DONDE ESTAN ARCHIVOS ===================
DATA_ROOT   = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\processed_data\data_bin"
MODEL_PATH  = r"C:\Users\GMADRO04\Documents\PALAS_EOLICAS_ML\code_modelo\svm_rbf_pipeline.joblib"
COLS_TXT    = os.path.join(DATA_ROOT, "selected_columns.txt")
ANNOT_DIRS  = [
    # agrega aquí rutas donde existan .json (LabelMe) con anotaciones de defectos SI LOS HAY SI NO NO HAY PROBLEMAS
    r"C:\Users\GMADRO04\Documents\PROYECTOML\processed_data\defectuosas",  # ejemplo
]

# =================== CONFIGURACIONES DE EXTRACCION DE CARACTERISTICAS ===========
WIDTH, HEIGHT = 1080, 720
CENTER_BAND_RATIO_DEFAULT = 0.60

HSV_BINS_H = 16
HSV_BINS_S = 12

HOG_ORI = 6
HOG_PPC = (60, 60)
HOG_CPB = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_P + 2

GLCM_DISTANCES = [2, 4]
GLCM_ANGLES = [0, np.pi/2]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

# =================== FUNCIONES DE PROCESAMIENTO DE LA INTERFAZ Y APLICACION DEL MODELO  ===================
def pil_from_bytes(file_bytes):
    return Image.open(io.BytesIO(file_bytes))

def pil_load_and_size_from_pil(pil_img, size_wh):
    im = ImageOps.exif_transpose(pil_img).convert("RGB")
    if im.size != size_wh:
        im = im.resize(size_wh, Image.Resampling.LANCZOS)
    return im

def gray_world_normalize(img_rgb_u8):
    img = img_rgb_u8.astype(np.float32) + 1e-6
    means = img.reshape(-1,3).mean(axis=0)
    scale = means.mean() / means
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)

def clahe_on_luminance(img_rgb_u8):
    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2,a,b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def crop_center_band(pil_img, band_ratio):
    w,h = pil_img.size
    bw = int(max(1, w * band_ratio))
    x0 = (w - bw)//2
    return pil_img.crop((x0, 0, x0+bw, h))

def build_roi(pil_img, band_ratio):
    if pil_img.size != (WIDTH, HEIGHT):
        pil_img = pil_img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    roi = crop_center_band(pil_img, band_ratio)
    rgb = np.asarray(roi.convert("RGB"))
    rgb = gray_world_normalize(rgb)
    rgb = clahe_on_luminance(rgb)
    roi_pil = Image.fromarray(rgb).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    return roi_pil

def hsv_histogram(pil_img, bins_h=HSV_BINS_H, bins_s=HSV_BINS_S):
    rgb = np.asarray(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0].ravel()
    S = hsv[:,:,1].ravel()
    h_hist,_ = np.histogram(H, bins=bins_h, range=(0,180), density=True)
    s_hist,_ = np.histogram(S, bins=bins_s, range=(0,256), density=True)
    return np.concatenate([h_hist, s_hist])  # 28

def hog_features_and_vis(pil_img):
    """Devuelve features HOG y visualización HOG para heatmap explicativo de la imagen."""
    g = rgb2gray(np.asarray(pil_img))
    feats, hog_image = hog(
        g, orientations=HOG_ORI,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
        visualize=True
    )
    # normalizar hog_image a 0..255 para overlay
    hmin, hmax = hog_image.min(), hog_image.max()
    if hmax > hmin:
        hog_vis = ((hog_image - hmin) / (hmax - hmin) * 255).astype(np.uint8)
    else:
        hog_vis = (hog_image * 0).astype(np.uint8)
    return feats, hog_vis

def lbp_hist(pil_img):
    g = rgb2gray(np.asarray(pil_img))
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
        v = graycoprops(glcm, prop)
        vals.append(v.mean())
    return np.array(vals, dtype=float)

def edge_and_focus_map(pil_img):
    """Devuelve densidad de bordes y mapa de bordes para overlay."""
    g = rgb2gray(np.asarray(pil_img))
    edges = canny(g, sigma=1.2)
    edge_density = float(edges.mean())
    # mapa binario de bordes a uint8
    edge_map = (edges.astype(np.uint8) * 255)
    # Varianza del laplaciano (número, no mapa)
    lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    lap = convolve2d(g, lap_kernel, mode="same", boundary="symm")
    lap_var = float(lap.var())
    return edge_density, lap_var, edge_map

def extract_features_row_from_pil(pil_img, band_ratio):
    pil_orig = pil_load_and_size_from_pil(pil_img, (WIDTH, HEIGHT))
    roi = build_roi(pil_orig, band_ratio)

    f_hsv = hsv_histogram(roi)
    f_hog, hog_vis = hog_features_and_vis(roi)
    f_lbp = lbp_hist(roi)
    f_glc = glcm_stats(roi)
    edge_density, lap_var, edge_map = edge_and_focus_map(roi)

    row = {}
    for i,v in enumerate(f_hsv): row[f"hsv_{i}"]  = float(v)
    for i,v in enumerate(f_hog): row[f"hog_{i}"]  = float(v)
    for i,v in enumerate(f_lbp): row[f"lbp_{i}"]  = float(v)
    for i,v in enumerate(f_glc): row[f"glcm_{i}"] = float(v)
    row["edge_density"]  = float(edge_density)
    row["laplacian_var"] = float(lap_var)

    return row, pil_orig, roi, hog_vis, edge_map

def align_to_selected_columns(row_dict, selected_cols):
    x = np.zeros((len(selected_cols),), dtype=np.float32)
    for i, c in enumerate(selected_cols):
        x[i] = float(row_dict.get(c, 0.0))
    return x.reshape(1, -1)

def try_load_annotation_polygon(img_basename, roi_w=WIDTH, roi_h=HEIGHT):
    """
    Busca un .json con mismo basename en ANNOT_DIRS formato LabelMe: shapes->points SOLO SI EXISTE.
    Devuelve lista de polígonos escalados al tamaño (roi_w, roi_h) o [vacio] si no hay.
    """
    name_noext = os.path.splitext(img_basename)[0]
    json_path = None
    for d in ANNOT_DIRS:
        cand = os.path.join(d, name_noext + ".json")
        if os.path.isfile(cand):
            json_path = cand
            break
    if not json_path:
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        polys = []
        imgW = data.get("imageWidth", roi_w)
        imgH = data.get("imageHeight", roi_h)
        for sh in data.get("shapes", []):
            pts = sh.get("points", [])
            scaled = [(p[0] * roi_w / imgW, p[1] * roi_h / imgH) for p in pts]
            polys.append(scaled)
        return polys
    except Exception:
        return []

def draw_polygons_on_pil(pil_img, polygons, color="red", width=3):
    out = pil_img.copy()
    drw = ImageDraw.Draw(out)
    for poly in polygons:
        drw.polygon(poly, outline=color, width=width)
    return out

# =================== CARGA DE MODELO Y COLUMNAS DE CARACTERISTICAS ===================
@st.cache_resource(show_spinner=False)
def load_model_and_columns(model_path, cols_txt):
    pipe = joblib.load(model_path)
    with open(cols_txt, "r", encoding="utf-8") as f:
        selected_cols = [line.strip() for line in f if line.strip()]
    return pipe, selected_cols

def predict_one(pipe, selected_cols, pil_img, band_ratio):
    row, pil_orig, roi, hog_vis, edge_map = extract_features_row_from_pil(pil_img, band_ratio)
    X = align_to_selected_columns(row, selected_cols)
    pred = int(pipe.predict(X)[0])  # 0=dañada, 1=sana
    # score
    try:
        prob1 = float(pipe.predict_proba(X)[:,1][0])
        score_txt = f"Prob. Sana = {prob1:.3f}"
    except Exception:
        try:
            d = float(pipe.decision_function(X)[0])
            score_txt = f"Margen = {d:.3f}"
        except Exception:
            score_txt = "Score no disponible"
    return pred, score_txt, pil_orig, roi, hog_vis, edge_map, row

# =================== UI interfaz de uso de la aplicación ===================
st.set_page_config(page_title="Clasificador de Palas Eólicas", layout="wide")
st.title("🧭 Clasificador de Palas Eólicas (Sana / Dañada)")
st.caption("Modelo: SVM RBF | Features: HSV + HOG reducido + LBP + GLCM + Edge/Focus | Banda central para reducir fondo")

with st.sidebar:
    st.header("Configuración")
    model_path = st.text_input("Ruta del modelo (.joblib)", MODEL_PATH)
    cols_path  = st.text_input("Ruta selected_columns.txt", COLS_TXT)
    band_ratio = st.slider("Banda central (proporción del ancho)", 0.40, 0.80, CENTER_BAND_RATIO_DEFAULT, 0.01)
    st.markdown("---")
    show_explain = st.checkbox("Mostrar explicaciones (HOG/Canny)", value=True)
    use_annotations = st.checkbox("Dibujar defectos si hay JSON", value=True)
    st.info("Tip: usa 0.55–0.65 para minimizar fondo sin perder pala.")

pipe, selected_cols = load_model_and_columns(model_path, cols_path)

# ---------- Inferencia individual ----------
st.subheader("🔹 Inferencia individual")
up = st.file_uploader("Sube una imagen (JPG/PNG/BMP/WEBP)", type=["jpg","jpeg","png","bmp","webp"])
if up is not None:
    pil = pil_from_bytes(up.read()).convert("RGB")
    pred, score_txt, pil_orig, roi, hog_vis, edge_map, row = predict_one(pipe, selected_cols, pil, band_ratio)

    # Intentar dibujar anotaciones si existen
    annotated_roi = roi
    if use_annotations:
        polys = try_load_annotation_polygon(up.name, roi_w=roi.size[0], roi_h=roi.size[1])
        if polys:
            annotated_roi = draw_polygons_on_pil(roi, polys, color="red", width=3)

    etiqueta = "✅ Sana (1)" if pred==1 else "⚠️ Dañada (0)"

    c1, c2, c3 = st.columns([1,1,1], gap="large")
    with c1:
        st.image(pil_orig, caption="Imagen (1080×720)", use_column_width=True)
    with c2:
        st.image(annotated_roi, caption=f"ROI (banda central {band_ratio:.2f})", use_column_width=True)
    with c3:
        if show_explain:
            # construir overlay HOG + Canny sobre ROI
            roi_rgb = np.asarray(roi.convert("RGB"))
            hog_color = cv2.applyColorMap(hog_vis, cv2.COLORMAP_JET)
            hog_overlay = cv2.addWeighted(roi_rgb, 0.6, hog_color, 0.4, 0)
            # pintar bordes canny en rojo
            overlay_edges = hog_overlay.copy()
            overlay_edges[edge_map > 0] = (255, 50, 50)
            st.image(overlay_edges, caption="Explicación: HOG heatmap + bordes Canny", use_column_width=True)

    st.success(f"**Predicción:** {etiqueta} | {score_txt}")

# ---------- Lote de interfaz ----------
st.markdown("---")
st.subheader("🔹 Lote de imágenes")
ups = st.file_uploader("Sube varias imágenes", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)
if ups:
    rows = []
    for f in ups:
        pil = pil_from_bytes(f.read()).convert("RGB")
        pred, score_txt, _, _, _, _, _ = predict_one(pipe, selected_cols, pil, band_ratio)
        rows.append({"file": f.name, "pred": pred, "score": score_txt})
    df = pd.DataFrame(rows).sort_values("file")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", csv, file_name="predicciones_svm.csv", mime="text/csv")
