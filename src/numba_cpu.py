"""Variante D: parte numérica del procesamiento de imágenes compilada con Numba.

Las funciones se compilan a código máquina con @njit y nogil=True, de modo que
el hilo que las ejecuta suelta el GIL mientras corren. Reemplazan los pasos
"mixtos" de la sección 4.2 (conversión a float64 + acumulación, y la media
del brillo). La decodificación y el redimensionado siguen en Pillow.

Los resultados son idénticos a los de numpy: las sumas son de enteros 0-255,
que se representan sin error, así que el orden de la suma no importa.
Este módulo solo se importa en la variante D (Numba no soporta Python sin GIL).
"""
import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def acumular(acumulador, img):
    """acumulador (float64, H×W×3) += img (uint8, H×W×3)."""
    alto, ancho, canales = img.shape
    for i in range(alto):
        for j in range(ancho):
            for k in range(canales):
                acumulador[i, j, k] += img[i, j, k]


@njit(nogil=True, cache=True)
def brillo(img):
    """Media de todos los valores de una imagen uint8 (igual que np.mean en float64)."""
    total = 0
    plano = img.ravel()
    for v in plano:
        total += v
    return total / plano.size


def compilar():
    """Fuerza la compilación (o la carga desde la caché) antes de medir."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    acumular(np.zeros((64, 64, 3)), img)
    brillo(img)
