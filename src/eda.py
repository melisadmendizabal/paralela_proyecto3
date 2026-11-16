"""Operaciones del análisis exploratorio que procesan imágenes (CPU).

Versión SECUENCIAL, portada de original/lab3_EDA.ipynb. Solo se incluyen las
operaciones con carga de cómputo relevante (Tabla 2 del informe):
    3  imagen_promedio_por_clase   carga 200 imágenes por clase y promedia
    4  distancias_entre_clases     distancia euclidiana entre imágenes promedio
    8  brillo_por_clase            brillo medio de cada imagen limpia
Los gráficos del EDA original no se reproducen: no forman parte de la medición.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist, squareform

from . import config


# ----------------------------------------------------------------------------
# Op. 3 - Imagen promedio por clase
# ----------------------------------------------------------------------------
def rutas_muestra_clase(df, clase, n_muestra=config.N_MUESTRA_PROMEDIO, semilla=config.SEMILLA):
    rng = np.random.default_rng(semilla)
    rutas = df[df["clase"] == clase]["ruta"].values
    n = min(n_muestra, len(rutas))
    return rng.choice(rutas, size=n, replace=False)


def imagen_promedio_por_clase(df, clase, tam=config.TAM_OBJETIVO,
                              n_muestra=config.N_MUESTRA_PROMEDIO, semilla=config.SEMILLA):
    muestra = rutas_muestra_clase(df, clase, n_muestra, semilla)
    acumulador = np.zeros((*tam[::-1], 3), dtype=np.float64)
    for ruta in muestra:
        with Image.open(ruta) as img:
            acumulador += np.array(img.convert("RGB").resize(tam), dtype=np.float64)
    return (acumulador / len(muestra)).astype(np.uint8)


def promedios_por_clase(df):
    clases = sorted(df["clase"].unique())
    return {clase: imagen_promedio_por_clase(df, clase) for clase in clases}


# ----------------------------------------------------------------------------
# Op. 4 - Distancias entre clases
# ----------------------------------------------------------------------------
def distancias_entre_clases(promedios):
    clases = sorted(promedios)
    matriz = np.array([promedios[c].astype(np.float64).flatten() for c in clases])
    distancias = squareform(pdist(matriz, metric="euclidean"))
    pares = [(clases[i], clases[j], distancias[i, j])
             for i, j in combinations(range(len(clases)), 2)]
    pares_df = pd.DataFrame(pares, columns=["clase_1", "clase_2", "distancia"]).sort_values("distancia")
    return pd.DataFrame(distancias, index=clases, columns=clases), pares_df


# ----------------------------------------------------------------------------
# Op. 8 - Brillo por clase
# ----------------------------------------------------------------------------
def calcular_brillo(ruta, tam=config.TAM_OBJETIVO):
    with Image.open(ruta) as img:
        return np.array(img.convert("RGB").resize(tam), dtype=np.float64).mean()


def brillo_por_clase(df, brillos):
    tmp = df[["clase"]].copy()
    tmp["brillo"] = brillos
    return tmp.groupby("clase")["brillo"].agg(["mean", "std"]).sort_values("mean")


def calcular_brillos(df):
    """Versión secuencial: un bucle sobre todas las imágenes limpias."""
    return df["ruta"].apply(calcular_brillo).values
