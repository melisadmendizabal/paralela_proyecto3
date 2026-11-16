"""PASO 5 - Pipeline de CPU paralelo: preparación de imágenes (variantes A–E).

Implementa el diseño de la sección 5.1 del informe:
  - cada fase divide su lista de imágenes en 4 × N bloques contiguos;
  - los bloques se reparten dinámicamente entre N trabajadores;
  - los resultados se recogen en el orden de los bloques, así que la salida es
    idéntica a la de la versión secuencial (src/datos.py, src/eda.py).

Variantes:
  secuencial  A  sin trabajadores (mismo código de bloques, en un bucle)
  hilos       B  ThreadPoolExecutor
  procesos    C  ProcessPoolExecutor + memoria compartida para F3
  numba       D  hilos + parte numérica compilada con Numba (src/numba_cpu.py)
  E (Python sin GIL) = variante "hilos" ejecutada con python3.13t.

IMPORTANTE (Windows): las funciones _tarea_* deben estar en un módulo
importable para que los procesos las encuentren, y el programa que crea el pool
debe estar protegido con  if __name__ == "__main__".  Este módulo no importa
torch para que cada proceso trabajador arranque rápido.
"""
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
from PIL import Image

from . import config, datos, eda
from .energia import desactivar_ecoqos

VARIANTES = ("secuencial", "hilos", "procesos", "numba")
BLOQUES_POR_TRABAJADOR = 4


def gil_activo():
    """True/False en Python 3.13+, None en versiones sin la función."""
    f = getattr(sys, "_is_gil_enabled", None)
    return f() if f else None


def dividir(n, trabajadores):
    """Límites [inicio, fin) de 4 × trabajadores bloques contiguos."""
    k = max(1, min(n, trabajadores * BLOQUES_POR_TRABAJADOR))
    lim = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(lim[:-1], lim[1:]) if b > a]


# ============================================================================
# Tareas (se ejecutan en los trabajadores; reciben un bloque de rutas)
# ============================================================================
def _calentar(_):
    """Fuerza la creación del trabajador y la importación de las librerías."""
    import time
    time.sleep(0.05)
    return gil_activo()


def _tarea_metadatos(rutas):
    salida = []
    for r in rutas:
        m = datos.leer_metadatos(r)
        salida.append(None if m is None else (m["ancho"], m["alto"], m["modo"], m["formato"]))
    return salida


def _tarea_hash(rutas):
    return [datos.calcular_hash(r, hash_size=config.HASH_SIZE) for r in rutas]


def _cargar_uint8(ruta, tam=config.TAM_OBJETIVO):
    """Mismas operaciones que datos.cargar_imagen, sin convertir a float."""
    with Image.open(ruta) as img:
        return np.asarray(img.convert("RGB").resize(tam), dtype=np.uint8)


def _tarea_brillo(rutas, usar_numba=False):
    if usar_numba:
        from . import numba_cpu
        return np.array([numba_cpu.brillo(_cargar_uint8(r)) for r in rutas])
    return np.array([eda.calcular_brillo(r) for r in rutas])


def _tarea_promedio(pares, usar_numba=False):
    """pares = [(clase, ruta)]. Devuelve {clase: [suma float64 64×64×3, cuenta]}."""
    if usar_numba:
        from . import numba_cpu
    parciales = {}
    for clase, ruta in pares:
        if clase not in parciales:
            parciales[clase] = [np.zeros((*config.TAM_OBJETIVO[::-1], 3), dtype=np.float64), 0]
        acc = parciales[clase]
        if usar_numba:
            numba_cpu.acumular(acc[0], _cargar_uint8(ruta))
        else:
            with Image.open(ruta) as img:
                acc[0] += np.array(img.convert("RGB").resize(config.TAM_OBJETIVO), dtype=np.float64)
        acc[1] += 1
    return parciales


def _escribir(destino, rutas, inicio):
    for i, r in enumerate(rutas):
        destino[inicio + i] = _cargar_uint8(r)


def _tarea_carga_compartida(nombre, forma, rutas, inicio):
    """Versión para procesos: escribe en un bloque de memoria compartida."""
    shm = shared_memory.SharedMemory(name=nombre)
    try:
        destino = np.ndarray(forma, dtype=np.uint8, buffer=shm.buf)
        _escribir(destino, rutas, inicio)
        del destino
    finally:
        shm.close()
    return inicio


# ============================================================================
# Pool de trabajadores
# ============================================================================
class PoolDatos:
    """Ejecuta funciones sobre bloques con la variante elegida."""

    def __init__(self, variante, trabajadores):
        if variante not in VARIANTES:
            raise ValueError(f"variante desconocida: {variante}")
        self.variante = variante
        self.trabajadores = 1 if variante == "secuencial" else trabajadores
        self.usar_numba = variante == "numba"
        if variante in ("hilos", "numba"):
            self.pool = ThreadPoolExecutor(self.trabajadores)
        elif variante == "procesos":
            # Cada proceso trabajador también pide no ser limitado por EcoQoS
            self.pool = ProcessPoolExecutor(self.trabajadores, initializer=desactivar_ecoqos)
        else:
            self.pool = None

    def iniciar(self):
        """Crea todos los trabajadores (y compila Numba) antes de medir las fases."""
        if self.usar_numba:
            from . import numba_cpu
            numba_cpu.compilar()
        if self.pool is not None:
            # Tareas lentas en paralelo obligan al pool a crear todos sus trabajadores
            list(self.pool.map(_calentar, range(self.trabajadores * 2)))

    def mapear(self, fn, items, **kw):
        """Aplica fn a bloques de items y devuelve los resultados por bloque, en orden."""
        trozos = [items[a:b] for a, b in dividir(len(items), self.trabajadores)]
        if self.pool is None:
            return [fn(t, **kw) for t in trozos]
        return list(self.pool.map(_Llamada(fn, kw), trozos))

    def cerrar(self):
        if self.pool is not None:
            self.pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()


class _Llamada:
    """fn(trozo, **kw) que se puede enviar a otro proceso (las lambdas no)."""

    def __init__(self, fn, kw):
        self.fn, self.kw = fn, kw

    def __call__(self, trozo):
        return self.fn(trozo, **self.kw)


# ============================================================================
# Fases del pipeline (mismas entradas y salidas que la versión secuencial)
# ============================================================================
def escanear_dataset(pool, ruta_dataset=config.RUTA_DATASET):
    """F1: equivalente a datos.escanear_dataset."""
    entradas = datos.listar_rutas(ruta_dataset)
    metas = [m for bloque in pool.mapear(_tarea_metadatos, [e[2] for e in entradas]) for m in bloque]
    filas = []
    for (clase, archivo, ruta), m in zip(entradas, metas):
        if m is None:
            print(f"No se pudo abrir {ruta}")
            continue
        filas.append({"clase": clase, "archivo": archivo, "ruta": ruta,
                      "ancho": m[0], "alto": m[1], "modo": m[2], "formato": m[3]})
    return pd.DataFrame(filas)


def promedios_por_clase(pool, df):
    """EDA: equivalente a eda.promedios_por_clase (reducción de sumas parciales)."""
    clases = sorted(df["clase"].unique())
    pares = [(c, r) for c in clases for r in eda.rutas_muestra_clase(df, c)]
    total = {}
    for parcial in pool.mapear(_tarea_promedio, pares, usar_numba=pool.usar_numba):
        for c, (suma, n) in parcial.items():
            if c in total:
                total[c][0] += suma
                total[c][1] += n
            else:
                total[c] = [suma, n]
    return {c: (total[c][0] / total[c][1]).astype(np.uint8) for c in clases}


def calcular_hashes(pool, submuestra):
    """F2: equivalente a datos.calcular_hashes."""
    hashes = [h for b in pool.mapear(_tarea_hash, list(submuestra["ruta"])) for h in b]
    return pd.Series(hashes, index=submuestra.index, name="ruta")


def calcular_brillos(pool, df):
    """EDA: equivalente a eda.calcular_brillos."""
    return np.concatenate(pool.mapear(_tarea_brillo, list(df["ruta"]), usar_numba=pool.usar_numba))


def cargar_uint8(pool, rutas):
    """F3: imágenes 64×64×3 en uint8 en un arreglo reservado de antemano.

    X_uint8 / 255 (en float32) es idéntico a datos.cargar_y_preprocesar.
    """
    rutas = list(rutas)
    forma = (len(rutas), *config.TAM_OBJETIVO[::-1], 3)
    limites = dividir(len(rutas), pool.trabajadores)
    if pool.variante == "procesos":
        shm = shared_memory.SharedMemory(create=True, size=int(np.prod(forma)))
        try:
            futuros = [pool.pool.submit(_tarea_carga_compartida, shm.name, forma, rutas[a:b], a)
                       for a, b in limites]
            for f in futuros:
                f.result()
            X = np.ndarray(forma, dtype=np.uint8, buffer=shm.buf).copy()
        finally:
            shm.close()
            shm.unlink()
        return X
    X = np.empty(forma, dtype=np.uint8)
    if pool.pool is None:
        for a, b in limites:
            _escribir(X, rutas[a:b], a)
    else:
        for f in [pool.pool.submit(_escribir, X, rutas[a:b], a) for a, b in limites]:
            f.result()
    return X


def preparar_datos(pool, crono, con_eda=True):
    """Ejecuta F1–F3 (+ EDA) con el pool dado. Devuelve un diccionario con todo.

    crono: objeto con un context manager fase(nombre), como medicion.Cronometro.
    """
    r = {}
    with crono.fase("F1_ingesta"):
        datos_img = escanear_dataset(pool)
    r["datos_img"] = datos_img
    if con_eda:
        with crono.fase("EDA_promedios"):
            r["promedios"] = promedios_por_clase(pool, datos_img)
            _, r["pares"] = eda.distancias_entre_clases(r["promedios"])
    with crono.fase("F2_limpieza"):
        sub = datos.submuestrear(datos_img)
        sub["phash"] = calcular_hashes(pool, sub)
        limpia, info = datos.limpiar_duplicados(sub)
    r["limpia"], r["info_limpieza"] = limpia, info
    if con_eda:
        with crono.fase("EDA_brillo"):
            brillos = calcular_brillos(pool, limpia)
            r["brillo_clase"] = eda.brillo_por_clase(limpia, brillos)
        r["brillos"] = brillos
    with crono.fase("F3_preprocesamiento"):
        train_df, val_df, test_df = datos.dividir_train_val_test(limpia)
        # Una sola pasada sobre las tres particiones: mejor reparto de la carga
        X = cargar_uint8(pool, list(train_df["ruta"]) + list(val_df["ruta"]) + list(test_df["ruta"]))
        n1, n2 = len(train_df), len(train_df) + len(val_df)
        codificador, y_train, y_val, y_test = datos.codificar_etiquetas(
            train_df["clase"].values, val_df["clase"].values, test_df["clase"].values)
    r.update(X_train=X[:n1], X_val=X[n1:n2], X_test=X[n2:], y_train=y_train, y_val=y_val, y_test=y_test,
             codificador=codificador, particion={"train": len(train_df), "val": len(val_df), "test": len(test_df)})
    return r
