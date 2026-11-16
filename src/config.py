"""Parámetros compartidos por la versión secuencial y la paralela.

Ambas versiones DEBEN usar exactamente estos valores para que la comparación
de tiempos sea justa (mismos datos, misma arquitectura, mismos hiperparámetros).
Los valores replican los del Laboratorio 3 (original/lab3_cnn.ipynb).
"""
from pathlib import Path

# --- Rutas ---
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "archive" / "asl_alphabet_train" / "asl_alphabet_train"
DIR_BENCHMARKS = RAIZ / "benchmarks"
DIR_MODELOS = RAIZ / "modelos"

# --- Datos (idénticos al Lab 3) ---
SEMILLA = 42
N_POR_CLASE = 600          # submuestra estratificada por clase
TAM_OBJETIVO = (64, 64)    # resolución de entrada al modelo
HASH_SIZE = 16             # tamaño del perceptual hash
UMBRAL_MSE = 15            # umbral para confirmar duplicados
N_MUESTRA_PROMEDIO = 200   # imágenes por clase para la imagen promedio (EDA)

# --- Modelo: CNN 2, configuración 2 (mejor modelo del Lab 3) ---
MODELO = "cnn2"
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.5
