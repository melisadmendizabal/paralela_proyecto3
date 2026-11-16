"""Fase de preparación de datos (CPU).

Versión SECUENCIAL: porta sin cambios de lógica el pipeline de
original/lab3_cnn.ipynb y original/procesamiento.py. Cada imagen se procesa
una por una dentro de un bucle de Python, en un solo proceso (un núcleo).

Operaciones (numeración de la Tabla 2 del informe):
    1  escanear_dataset          recorrido del dataset + metadatos
    5  submuestrear              submuestreo estratificado
    6  calcular_hash             perceptual hash
    7  limpiar_duplicados        verificación de duplicados por MSE
    9  dividir_train_val_test    partición estratificada 70/15/15
    10 cargar_y_preprocesar      carga + RGB + resize + normalización
    11 codificar_etiquetas       LabelEncoder
"""
import os

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config


# ----------------------------------------------------------------------------
# Op. 1 - Recorrido del dataset
# ----------------------------------------------------------------------------
def leer_metadatos(ruta):
    """Abre un JPG (solo cabecera) y devuelve sus metadatos, o None si falla."""
    try:
        with Image.open(ruta) as imagen:
            ancho, alto = imagen.size
            return {"ancho": ancho, "alto": alto, "modo": imagen.mode, "formato": imagen.format}
    except Exception:
        return None


def listar_rutas(ruta_dataset=config.RUTA_DATASET):
    """Lista (clase, archivo, ruta) en el mismo orden que el Lab 3."""
    ruta_dataset = str(ruta_dataset)
    entradas = []
    for clase in sorted(os.listdir(ruta_dataset)):
        carpeta = os.path.join(ruta_dataset, clase)
        if not os.path.isdir(carpeta):
            continue
        for archivo in os.listdir(carpeta):
            entradas.append((clase, archivo, os.path.join(carpeta, archivo)))
    return entradas


def escanear_dataset(ruta_dataset=config.RUTA_DATASET):
    """Recorre todas las carpetas y registra los metadatos de cada imagen."""
    datos_img = []
    for clase, archivo, ruta in listar_rutas(ruta_dataset):
        meta = leer_metadatos(ruta)
        if meta is None:
            print(f"No se pudo abrir {ruta}")
            continue
        datos_img.append({"clase": clase, "archivo": archivo, "ruta": ruta, **meta})
    return pd.DataFrame(datos_img)


# ----------------------------------------------------------------------------
# Op. 5 - Submuestreo estratificado
# ----------------------------------------------------------------------------
def submuestrear(datos_img, n_por_clase=config.N_POR_CLASE, semilla=config.SEMILLA):
    partes = []
    for _, grupo in datos_img.groupby("clase"):
        n = min(n_por_clase, len(grupo))
        partes.append(grupo.sample(n=n, random_state=semilla))
    return pd.concat(partes).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Op. 6 y 7 - Perceptual hash y verificación de duplicados (procesamiento.py)
# ----------------------------------------------------------------------------
def calcular_hash(ruta, hash_size=config.HASH_SIZE):
    try:
        with Image.open(ruta) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None


def son_realmente_duplicados(ruta1, ruta2, tam=config.TAM_OBJETIVO, umbral_mse=config.UMBRAL_MSE):
    """Verificación adicional: compara los píxeles de verdad, no solo el hash."""
    img1 = np.array(Image.open(ruta1).convert("RGB").resize(tam), dtype=np.float64)
    img2 = np.array(Image.open(ruta2).convert("RGB").resize(tam), dtype=np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse < umbral_mse, mse


def marcar_duplicado_verificado(grupo_df, umbral_mse=config.UMBRAL_MSE):
    """Dentro de un grupo con mismo phash, solo confirma duplicado si el MSE es bajo."""
    indices_a_eliminar = []
    rutas = grupo_df["ruta"].tolist()
    indices = grupo_df.index.tolist()
    referencia = rutas[0]
    for idx, ruta in zip(indices[1:], rutas[1:]):
        es_dup, _ = son_realmente_duplicados(referencia, ruta, umbral_mse=umbral_mse)
        if es_dup:
            indices_a_eliminar.append(idx)
    return indices_a_eliminar


def calcular_hashes(submuestra):
    """Op. 6 secuencial: un hash por imagen, en un bucle."""
    return submuestra["ruta"].apply(lambda r: calcular_hash(r, hash_size=config.HASH_SIZE))


def limpiar_duplicados(submuestra):
    """Op. 7: elimina los duplicados confirmados. Devuelve (df_limpio, info)."""
    duplicados = submuestra[submuestra.duplicated(subset="phash", keep=False)]
    indices_reales = []
    for _, grupo in duplicados.groupby("phash"):
        indices_reales.extend(marcar_duplicado_verificado(grupo))
    limpia = submuestra.drop(index=indices_reales).reset_index(drop=True)
    info = {
        "imagenes_hash_repetido": int(len(duplicados)),
        "grupos_hash_repetido": int(duplicados["phash"].nunique()),
        "duplicados_eliminados": int(len(indices_reales)),
        "submuestra_final": int(len(limpia)),
    }
    return limpia, info


# ----------------------------------------------------------------------------
# Op. 9 - Partición estratificada 70/15/15
# ----------------------------------------------------------------------------
def dividir_train_val_test(df, col_clase="clase", semilla=config.SEMILLA,
                           test_size=0.30, val_size_of_temp=0.50):
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df[col_clase], random_state=semilla)
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size_of_temp, stratify=temp_df[col_clase], random_state=semilla)
    return train_df, val_df, test_df


# ----------------------------------------------------------------------------
# Op. 10 - Carga y preprocesamiento
# ----------------------------------------------------------------------------
def cargar_imagen(ruta, tam=config.TAM_OBJETIVO):
    """Abre, convierte a RGB, redimensiona y normaliza a [0, 1] una imagen."""
    with Image.open(ruta) as img:
        return np.array(img.convert("RGB").resize(tam), dtype=np.float32) / 255.0


def cargar_y_preprocesar(df, tam=config.TAM_OBJETIVO):
    """Versión secuencial: un bucle sobre todas las rutas."""
    imagenes = [cargar_imagen(ruta, tam) for ruta in df["ruta"]]
    X = np.stack(imagenes)
    y = df["clase"].values
    return X, y


# ----------------------------------------------------------------------------
# Op. 11 - Codificación de etiquetas
# ----------------------------------------------------------------------------
def codificar_etiquetas(y_train_raw, y_val_raw, y_test_raw):
    codificador = LabelEncoder()
    codificador.fit(y_train_raw)
    return (codificador,
            codificador.transform(y_train_raw),
            codificador.transform(y_val_raw),
            codificador.transform(y_test_raw))
