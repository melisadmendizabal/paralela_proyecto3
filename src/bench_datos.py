"""PASO 5 - Barrido del pipeline de CPU: variantes × número de trabajadores.

Mide las fases de preparación de datos (F1–F3 + EDA) con cada variante de
src/datos_paralelo.py y comprueba que todas producen EXACTAMENTE los mismos
datos que el código secuencial original (huellas SHA-256 de cada resultado).

Uso (Python estándar, variantes A–D):
    python -m src.bench_datos
    python -m src.bench_datos --variantes hilos procesos --trabajadores 8 --repeticiones 1   # prueba rápida
Uso (variante E, Python sin GIL; no admite numba):
    python3.13t -m src.bench_datos --variantes secuencial hilos --etiqueta 313t

Salidas en benchmarks/datos_paralelo/:
    referencia.json           huellas de los resultados del código secuencial original
    bench_datos[_etiqueta].csv  una fila por corrida: tiempos por fase, total, verificación
    log_bench[_etiqueta].txt  (si se redirige la salida)
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, datos, eda
from .datos_paralelo import PoolDatos, gil_activo, preparar_datos
from .energia import desactivar_ecoqos

DIR = config.DIR_BENCHMARKS / "datos_paralelo"
FASES = ["F0_pool", "F1_ingesta", "EDA_promedios", "F2_limpieza", "EDA_brillo", "F3_preprocesamiento"]


class Crono:
    def __init__(self):
        self.tiempos = {}

    @contextmanager
    def fase(self, nombre):
        t0 = time.perf_counter()
        yield
        self.tiempos[nombre] = time.perf_counter() - t0


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def _sha_df(df):
    return _sha(pd.util.hash_pandas_object(df, index=True).values.tobytes())


def huella(r):
    """Huella de cada resultado. Las imágenes se comparan como float32 / 255."""
    h = {"metadatos": _sha_df(r["datos_img"]),
         "promedios": _sha(b"".join(r["promedios"][c].tobytes() for c in sorted(r["promedios"]))),
         "limpia": _sha_df(r["limpia"]),
         "brillos": _sha(np.asarray(r["brillos"], dtype=np.float64).tobytes())}
    for k in ("X_train", "X_val", "X_test"):
        X = r[k]
        if X.dtype == np.uint8:
            X = X.astype(np.float32) / np.float32(255)
        h[k] = _sha(np.ascontiguousarray(X).tobytes())
    for k in ("y_train", "y_val", "y_test"):
        h[k] = _sha(np.asarray(r[k], dtype=np.int64).tobytes())
    return h


def referencia():
    """Ejecuta el código secuencial ORIGINAL (src/datos.py, src/eda.py) y devuelve sus huellas."""
    r = {"datos_img": datos.escanear_dataset()}
    r["promedios"] = eda.promedios_por_clase(r["datos_img"])
    sub = datos.submuestrear(r["datos_img"])
    sub["phash"] = datos.calcular_hashes(sub)
    r["limpia"], _ = datos.limpiar_duplicados(sub)
    r["brillos"] = eda.calcular_brillos(r["limpia"])
    tr, va, te = datos.dividir_train_val_test(r["limpia"])
    r["X_train"], ytr = datos.cargar_y_preprocesar(tr)
    r["X_val"], yva = datos.cargar_y_preprocesar(va)
    r["X_test"], yte = datos.cargar_y_preprocesar(te)
    _, r["y_train"], r["y_val"], r["y_test"] = datos.codificar_etiquetas(ytr, yva, yte)
    return huella(r)


def una_corrida(variante, trabajadores, verificar, ref):
    crono = Crono()
    with crono.fase("F0_pool"):
        pool = PoolDatos(variante, trabajadores)
        gil_trabajador = pool.iniciar()
    try:
        r = preparar_datos(pool, crono)
    finally:
        pool.cerrar()
    fila = {"variante": variante, "trabajadores": pool.trabajadores}
    fila.update({f: round(crono.tiempos.get(f, float("nan")), 4) for f in FASES})
    fila["total_s"] = round(sum(crono.tiempos.values()), 4)
    fila["total_sin_pool_s"] = round(fila["total_s"] - crono.tiempos["F0_pool"], 4)
    fila["imagenes_limpias"] = int(len(r["limpia"]))
    if verificar:
        h = huella(r)
        distintos = [k for k in ref if ref[k] != h.get(k)] if ref else ["(sin referencia)"]
        fila["verificado"] = not distintos
        fila["distintos"] = ";".join(distintos)
    del r
    return fila


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variantes", nargs="+", default=["secuencial", "hilos", "procesos", "numba"])
    p.add_argument("--trabajadores", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16])
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--etiqueta", default="", help="sufijo de los archivos (p. ej. 313t)")
    p.add_argument("--rehacer-referencia", action="store_true")
    args = p.parse_args()
    ecoqos = desactivar_ecoqos()
    DIR.mkdir(parents=True, exist_ok=True)
    suf = f"_{args.etiqueta}" if args.etiqueta else ""
    salida = DIR / f"bench_datos{suf}.csv"

    entorno = {"python": sys.version, "implementacion": platform.python_implementation(),
               "gil_activo_al_iniciar": gil_activo(), "ecoqos_desactivado": ecoqos, "numpy": np.__version__, "pandas": pd.__version__,
               "fecha": datetime.now().isoformat(timespec="seconds"), "argumentos": vars(args)}
    print("Entorno:", json.dumps(entorno, ensure_ascii=False), flush=True)

    ruta_ref = DIR / "referencia.json"
    if ruta_ref.exists() and not args.rehacer_referencia:
        ref = json.loads(ruta_ref.read_text())
    else:
        print("Calculando la referencia con el código secuencial original (~1 min)...", flush=True)
        ref = referencia()
        ruta_ref.write_text(json.dumps(ref, indent=2))
    print("Referencia:", ref, flush=True)

    # Calentamiento de la caché de disco (no se registra)
    print("Calentamiento...", flush=True)
    calent = "hilos" if "hilos" in args.variantes else args.variantes[0]
    una_corrida(calent, 8, False, ref)

    filas = []
    plan = [(v, n) for v in args.variantes for n in (args.trabajadores if v != "secuencial" else [1])]
    for rep in range(1, args.repeticiones + 1):
        for v, n in plan:
            t = datetime.now()
            fila = {"repeticion": rep, **una_corrida(v, n, verificar=(rep == 1), ref=ref),
                    "python": platform.python_version() + ("t" if gil_activo() is False else ""),
                    "gil_activo": gil_activo(), "inicio": t.isoformat(timespec="seconds")}
            filas.append(fila)
            pd.DataFrame(filas).to_csv(salida, index=False)
            print(f"rep {rep} {v:10s} N={n:2d}  total={fila['total_s']:6.2f} s  "
                  f"(pool {fila['F0_pool']:.2f} s)  verificado={fila.get('verificado', '-')}  "
                  f"GIL={fila['gil_activo']}", flush=True)
    print("Guardado en", salida)


if __name__ == "__main__":
    main()
