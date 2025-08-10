# 🌀 Clasificación de Palas Eólicas: Sana vs Dañada

Este proyecto implementa un sistema de **clasificación automática de palas eólicas** usando **Machine Learning clásico** y técnicas de **procesamiento de imágenes**.  
Incluye:
- Extracción de *features* (HSV, HOG, LBP, GLCM, bordes, enfoque).
- Entrenamiento de varios modelos (SVM, Random Forest, MLP, kNN, Regresión Logística).
- **Interfaz interactiva** en Streamlit para predecir el estado de nuevas imágenes.
- Resaltado de zonas dañadas a partir de anotaciones JSON o *heatmaps* explicativos.

---

## 📂 Estructura del proyecto

PALAS_EOLICAS_ML/
│
├── code_procesamiento_datos/ # Scripts de extracción y curado de características
├── code_entrenamiento_modelo/ # Scripts de entrenamiento y evaluación de modelos
├── code_modelo/ # Modelos entrenados (.joblib) y columnas seleccionadas
├── codigo_interfaz/ # Código de la interfaz en Streamlit
├── processed_data/
│ ├── data_bin/ # CSV de features y columnas seleccionadas
│ ├── train/ val/ test/ # Datos organizados por split
│
└── README.md # Este archivo
---
## 🛠️ Requisitos

- Python 3.9+
- Librerías:
  ```bash
    pip install streamlit scikit-learn opencv-python-headless pillow numpy pandas matplotlib scikit-image joblib ó
    pip install requirements.txt
    ```