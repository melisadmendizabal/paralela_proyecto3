"""PASO 3 - Perfilado de la versión secuencial para identificar qué paralelizar.

Tres mediciones, todas sobre el código secuencial sin modificar su lógica:

  A. Desglose por función de las fases de datos (una corrida real completa):
     separa el bucle por imagen (paralelizable) de las partes que se ejecutan
     una sola vez sobre todo el conjunto (pandas, partición, np.stack, ...).

  B. Desglose por paso DENTRO del procesamiento de una imagen (muestra de
     imágenes): abrir, decodificar JPEG, convertir, redimensionar, numpy, ...
     Cada paso se clasifica según si se ejecuta en C liberando el GIL o en
     Python reteniéndolo. Esto predice cuánto pueden ayudar los hilos.

  C. Desglose del entrenamiento en CPU (torch.profiler): adelante / atrás /
     optimizador y operaciones de PyTorch que más tiempo consumen. Indica qué
     capas conviene implementar como kernels CUDA propios.

Uso:   python -m src.perfil                 (≈ 2 min)
Salida: benchmarks/perfil/
"""
import argparse
import os
import time
from collections import defaultdict

import imagehash
import numpy as np
import pandas as pd
import scipy.fftpack
import torch
from PIL import Image
from torch import nn

from . import config, datos, eda
from .entrenamiento import CL, a_tensores
from .medicion import guardar_json, info_entorno
from .modelo import CNN2
from .energia import desactivar_ecoqos

DIR = config.DIR_BENCHMARKS / "perfil"
REPETICIONES = 3
T = time.perf_counter

# Clasificación de cada paso según el GIL (ver el informe, sección 4):
#   "C, libera GIL"   código C de Pillow/E-S que suelta el GIL mientras trabaja
#   "Python, retiene" intérprete de Python (análisis de cabecera, cadenas, bucles)
#   "Mixto"           llamada a numpy/scipy sobre arreglos pequeños: el cálculo
#                     interno es C sin GIL, pero el costo fijo de la llamada es Python
GIL = {
    "leer bytes del archivo": "C, libera GIL",
    "analizar cabecera JPEG": "Python, retiene GIL",
    "abrir (cabecera)": "Python, retiene GIL",
    "decodificar JPEG": "C, libera GIL",
    "convertir modo": "C, libera GIL",
    "redimensionar": "C, libera GIL",
    "a numpy": "Mixto",
    "acumular / media": "Mixto",
    "normalizar /255": "Mixto",
    "DCT 2D": "Mixto",
    "mediana + umbral": "Mixto",
    "hash a texto": "Python, retiene GIL",
    "orquestación Python": "Python, retiene GIL",
}


# ============================================================================
# A. Desglose por función de las fases de datos
# ============================================================================
def perfil_funciones():
    """Devuelve (tabla, segundos de la primera pasada de F1 = caché de disco fría)."""
    c = T(); datos.escanear_dataset(); t_frio = T() - c     # calentamiento, como en el Paso 2
    t = {}
    c = T(); entradas = datos.listar_rutas(); t[("F1", "listar carpetas (os.listdir)")] = T() - c
    c = T()
    filas = []
    for clase, archivo, ruta in entradas:
        meta = datos.leer_metadatos(ruta)
        if meta is not None:
            filas.append({"clase": clase, "archivo": archivo, "ruta": ruta, **meta})
    t[("F1", "bucle por imagen: leer metadatos")] = T() - c
    c = T(); datos_img = pd.DataFrame(filas); t[("F1", "construir DataFrame")] = T() - c

    clases = sorted(datos_img["clase"].unique())
    c = T(); promedios = {k: eda.imagen_promedio_por_clase(datos_img, k) for k in clases}
    t[("EDA_promedios", "bucle por imagen: cargar y acumular (29×200)")] = T() - c
    c = T(); eda.distancias_entre_clases(promedios); t[("EDA_promedios", "distancias entre clases")] = T() - c

    c = T(); sub = datos.submuestrear(datos_img); t[("F2", "submuestreo (pandas)")] = T() - c
    c = T(); sub["phash"] = datos.calcular_hashes(sub); t[("F2", "bucle por imagen: perceptual hash")] = T() - c
    c = T(); limpia, _ = datos.limpiar_duplicados(sub); t[("F2", "verificar duplicados (MSE, 22 imágenes)")] = T() - c

    c = T(); brillos = eda.calcular_brillos(limpia); t[("EDA_brillo", "bucle por imagen: brillo")] = T() - c
    c = T(); eda.brillo_por_clase(limpia, brillos); t[("EDA_brillo", "agrupar por clase")] = T() - c

    c = T(); tr, va, te = datos.dividir_train_val_test(limpia); t[("F3", "partición estratificada")] = T() - c
    imgs, t_bucle = [], 0.0
    for df in (tr, va, te):
        c = T(); lista = [datos.cargar_imagen(r) for r in df["ruta"]]; t_bucle += T() - c
        imgs.append(lista)
    t[("F3", "bucle por imagen: cargar + resize + normalizar")] = t_bucle
    c = T(); Xs = [np.stack(l) for l in imgs]; t[("F3", "np.stack")] = T() - c
    c = T(); datos.codificar_etiquetas(tr["clase"].values, va["clase"].values, te["clase"].values)
    t[("F3", "codificar etiquetas")] = T() - c
    c = T()
    for X in Xs:
        a_tensores(X, np.zeros(len(X), dtype=np.int64))
    t[("F3", "a tensores NCHW (copia)")] = T() - c

    n_img = {"F1": len(entradas), "EDA_promedios": 29 * config.N_MUESTRA_PROMEDIO,
             "F2": len(sub), "EDA_brillo": len(limpia), "F3": len(limpia)}
    filas = [{"fase": f, "parte": p, "segundos": s, "por_imagen": p.startswith("bucle"),
              "ms_por_imagen": 1000 * s / n_img[f] if p.startswith("bucle") else None}
             for (f, p), s in t.items()]
    return pd.DataFrame(filas), t_frio


# ============================================================================
# B. Desglose por paso dentro del procesamiento de una imagen
# ============================================================================
def _pasos_metadatos(ruta, acc):
    c = T()
    with open(ruta, "rb") as f:
        f.read(4096)
    acc["leer bytes del archivo"] += T() - c
    c = T()
    with Image.open(ruta) as img:
        _ = (img.size, img.mode, img.format)
    acc["abrir (cabecera)"] += T() - c


def _cargar_rgb(ruta, acc, tam=config.TAM_OBJETIVO):
    c = T(); img = Image.open(ruta); acc["abrir (cabecera)"] += T() - c
    c = T(); img.load(); acc["decodificar JPEG"] += T() - c
    c = T(); img2 = img.convert("RGB"); acc["convertir modo"] += T() - c
    c = T(); img3 = img2.resize(tam); acc["redimensionar"] += T() - c
    img.close()
    return img3


def _pasos_hash(ruta, acc, hash_size=config.HASH_SIZE):
    """Réplica paso a paso de imagehash.phash (imagehash 4.3.2)."""
    lado = hash_size * 4
    c = T(); img = Image.open(ruta); acc["abrir (cabecera)"] += T() - c
    c = T(); img.load(); acc["decodificar JPEG"] += T() - c
    c = T(); gris = img.convert("L"); acc["convertir modo"] += T() - c
    c = T(); peq = gris.resize((lado, lado), Image.Resampling.LANCZOS); acc["redimensionar"] += T() - c
    c = T(); px = np.asarray(peq); acc["a numpy"] += T() - c
    c = T(); dct = scipy.fftpack.dct(scipy.fftpack.dct(px, axis=0), axis=1); acc["DCT 2D"] += T() - c
    c = T(); baja = dct[:hash_size, :hash_size]; bits = baja > np.median(baja); acc["mediana + umbral"] += T() - c
    c = T(); h = str(imagehash.ImageHash(bits)); acc["hash a texto"] += T() - c
    img.close()
    return h


def _pasos_promedio(ruta, acc, acumulador):
    img = _cargar_rgb(ruta, acc)
    c = T(); a = np.array(img, dtype=np.float64); acc["a numpy"] += T() - c
    c = T(); acumulador += a; acc["acumular / media"] += T() - c


def _pasos_brillo(ruta, acc):
    img = _cargar_rgb(ruta, acc)
    c = T(); a = np.array(img, dtype=np.float64); acc["a numpy"] += T() - c
    c = T(); a.mean(); acc["acumular / media"] += T() - c


def _pasos_carga(ruta, acc):
    img = _cargar_rgb(ruta, acc)
    c = T(); a = np.array(img, dtype=np.float32); acc["a numpy"] += T() - c
    c = T(); a / 255.0; acc["normalizar /255"] += T() - c


def perfil_pasos(n_muestra, n_metadatos):
    entradas = datos.listar_rutas()
    rng = np.random.default_rng(config.SEMILLA)
    rutas_meta = [entradas[i][2] for i in rng.choice(len(entradas), n_metadatos, replace=False)]
    rutas = [entradas[i][2] for i in rng.choice(len(entradas), n_muestra, replace=False)]
    for r in set(rutas_meta) | set(rutas):          # calentar la caché de disco
        with open(r, "rb") as f:
            f.read()

    # Comprobación: la réplica paso a paso da el mismo hash que imagehash.phash
    for r in rutas[:50]:
        assert _pasos_hash(r, defaultdict(float)) == datos.calcular_hash(r), "réplica de phash distinta"

    operaciones = {
        "Op. 1 Metadatos (F1)": (rutas_meta, _pasos_metadatos, lambda r: datos.leer_metadatos(r)),
        "Op. 3 Imagen promedio (EDA)": (rutas, None, None),   # funciones definidas abajo
        "Op. 6 Perceptual hash (F2)": (rutas, _pasos_hash, lambda r: datos.calcular_hash(r)),
        "Op. 8 Brillo (EDA)": (rutas, _pasos_brillo, lambda r: eda.calcular_brillo(r)),
        "Op. 10 Carga y preprocesamiento (F3)": (rutas, _pasos_carga, lambda r: datos.cargar_imagen(r)),
    }
    filas = []
    for nombre, (rs, pasos, real) in operaciones.items():
        if nombre.startswith("Op. 3 "):
            def pasos(r, acc, _a=np.zeros((64, 64, 3))):
                _pasos_promedio(r, acc, _a)

            def real(r, _a=np.zeros((64, 64, 3))):
                with Image.open(r) as img:
                    _a += np.array(img.convert("RGB").resize(config.TAM_OBJETIVO), dtype=np.float64)
        for r in rs[:20]:                       # calentamiento (primeras llamadas, imports)
            pasos(r, defaultdict(float)); real(r)
        # Se alternan réplica paso a paso y función real, REPETICIONES veces;
        # se usa la mediana para no depender del orden ni de picos del sistema.
        reps_pasos, reps_real = [], []
        for _ in range(REPETICIONES):
            acc_rep = defaultdict(float)
            for r in rs:
                pasos(r, acc_rep)
            reps_pasos.append(acc_rep)
            c = T()
            for r in rs:
                real(r)
            reps_real.append(T() - c)
        acc = {k: float(np.median([a[k] for a in reps_pasos])) for k in reps_pasos[0]}
        t_real = float(np.median(reps_real))
        if nombre.startswith("Op. 1 "):
            # Image.open ya incluye leer el archivo: se separa la lectura (E/S)
            # del análisis de la cabecera en Python.
            leer = acc["leer bytes del archivo"]
            acc["analizar cabecera JPEG"] = max(acc.pop("abrir (cabecera)") - leer, 0.0)
        # Lo que la función real tarda de más respecto a la suma de pasos es
        # costo del intérprete (llamadas, with, excepciones, objetos temporales).
        t_pasos = sum(acc.values())
        acc["orquestación Python"] = max(t_real - t_pasos, 0.0)
        total = max(t_real, t_pasos)
        for paso, s in acc.items():
            filas.append({"operacion": nombre, "paso": paso, "gil": GIL[paso],
                          "ms_por_imagen": 1000 * s / len(rs), "pct": 100 * s / total, "n": len(rs),
                          "ms_real_por_imagen": 1000 * t_real / len(rs)})
    return pd.DataFrame(filas)


def amdahl(f, n):
    return 1.0 / ((1 - f) + f / n)


# ============================================================================
# C. Desglose del entrenamiento en CPU
# ============================================================================
def categoria_op(nombre):
    n = nombre.lower()
    if "convolution_backward" in n or "convolutionbackward" in n:
        return "Convolución (atrás)"
    if "convolution" in n or "conv2d" in n:
        return "Convolución (adelante)"
    if "relu" in n or "threshold" in n or "clamp_min" in n:
        return "ReLU"
    if any(k in n for k in ("addmm", "aten::mm", "linear")) or n == "aten::t":
        return "Capa densa (matmul)"
    if "max_pool" in n or "maxpool" in n:
        return "MaxPool"
    return "Otros (Adam, dropout, pérdida, copias)"


def perfil_entrenamiento(n_lotes):
    torch.manual_seed(config.SEMILLA)
    modelo = CNN2(dropout=config.DROPOUT).to(memory_format=CL)
    opt = torch.optim.Adam(modelo.parameters(), lr=config.LEARNING_RATE, eps=1e-7)
    crit = nn.CrossEntropyLoss()
    bs = config.BATCH_SIZE
    X = torch.rand(bs * (n_lotes + 3), 3, 64, 64).contiguous(memory_format=CL)
    y = torch.randint(0, 29, (bs * (n_lotes + 3),))
    fases = defaultdict(float)

    def paso(i, medir):
        xb, yb = X[i * bs:(i + 1) * bs], y[i * bs:(i + 1) * bs]
        c = T(); opt.zero_grad(); salida = modelo(xb); perdida = crit(salida, yb); t1 = T()
        perdida.backward(); t2 = T()
        opt.step(); t3 = T()
        if medir:
            fases["Propagación hacia adelante + pérdida"] += t1 - c
            fases["Propagación hacia atrás"] += t2 - t1
            fases["Optimizador Adam"] += t3 - t2

    modelo.train()
    for i in range(3):
        paso(i, False)
    for i in range(3, 3 + n_lotes):
        paso(i, True)
    df_fases = pd.DataFrame([{"fase": k, "ms_por_lote": 1000 * v / n_lotes} for k, v in fases.items()])
    df_fases["pct"] = 100 * df_fases["ms_por_lote"] / df_fases["ms_por_lote"].sum()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for i in range(3, 3 + n_lotes):
            paso(i, False)
    ops = pd.DataFrame([{"operacion": e.key, "llamadas": e.count,
                         "self_cpu_ms_por_lote": e.self_cpu_time_total / 1000 / n_lotes}
                        for e in prof.key_averages()])
    ops = ops[ops["self_cpu_ms_por_lote"] > 0].sort_values("self_cpu_ms_por_lote", ascending=False)
    ops["categoria"] = ops["operacion"].map(categoria_op)
    ops["pct"] = 100 * ops["self_cpu_ms_por_lote"] / ops["self_cpu_ms_por_lote"].sum()
    cat = ops.groupby("categoria")[["self_cpu_ms_por_lote", "pct"]].sum() \
             .sort_values("pct", ascending=False).reset_index()
    return df_fases, ops, cat


# ============================================================================
def grafica_pasos(pasos, ruta):
    """Barras 100 %: fracción del tiempo de cada operación según el GIL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .graficas import AZUL, NARANJA, TINTA, TINTA_2, REJILLA
    AQUA = "#1baf7a"
    colores = {"C, libera GIL": AZUL, "Mixto": AQUA, "Python, retiene GIL": NARANJA}
    orden = list(dict.fromkeys(pasos["operacion"]))            # orden del pipeline
    orden = sorted(orden, key=lambda o: ["Op. 1 ", "Op. 3 ", "Op. 6 ", "Op. 8 ", "Op. 10"].index(o[:6]))
    tabla = pasos.pivot_table(index="operacion", columns="gil", values="pct", aggfunc="sum").fillna(0)
    tabla = tabla.reindex(index=orden[::-1], columns=[c for c in colores if c in tabla.columns])
    ms = pasos.groupby("operacion")["ms_real_por_imagen"].first()
    fig, ax = plt.subplots(figsize=(10, 3.9))
    izq = np.zeros(len(tabla))
    for col in tabla.columns:
        ax.barh(tabla.index, tabla[col], left=izq, color=colores[col], height=0.6,
                edgecolor="white", linewidth=2, label=col)
        for y, (x0, w) in enumerate(zip(izq, tabla[col])):
            if w >= 8:
                ax.text(x0 + w / 2, y, f"{w:.0f} %", ha="center", va="center", fontsize=8, color="white")
        izq += tabla[col].to_numpy()
    for y, op in enumerate(tabla.index):
        ax.text(101.5, y, f"{ms[op]:.2f} ms/imagen", va="center", fontsize=9, color=TINTA)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% del tiempo de procesar una imagen", color=TINTA_2)
    ax.set_title("Dónde se va el tiempo al procesar una imagen (versión secuencial)",
                 color=TINTA, loc="left", fontsize=12, pad=28)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--muestra", type=int, default=1000, help="imágenes para el desglose por paso")
    p.add_argument("--muestra-metadatos", type=int, default=3000)
    p.add_argument("--lotes", type=int, default=20, help="lotes para perfilar el entrenamiento")
    p.add_argument("--sin-funciones", action="store_true", help="omitir la medición A (corrida completa)")
    args = p.parse_args()
    desactivar_ecoqos()
    DIR.mkdir(parents=True, exist_ok=True)
    guardar_json(info_entorno(), DIR / "entorno.json")

    resumen = {}
    if not args.sin_funciones:
        print("A. Desglose por función (corrida completa de las fases de datos)...", flush=True)
        fun, t_frio = perfil_funciones()
        resumen["F1_primera_pasada_cache_fria_s"] = t_frio
        print(f"   F1 primera pasada (caché fría): {t_frio:.1f} s", flush=True)
        fun.to_csv(DIR / "perfil_funciones_datos.csv", index=False)
        print(fun.round(3).to_string(index=False), flush=True)
        total = fun["segundos"].sum()
        bucle = fun.loc[fun["por_imagen"], "segundos"].sum()
        resumen["datos_total_s"] = total
        resumen["datos_bucle_por_imagen_s"] = bucle
        resumen["datos_fraccion_bucle_por_imagen"] = bucle / total

    print("\nB. Desglose por paso dentro de cada imagen...", flush=True)
    pasos = perfil_pasos(args.muestra, args.muestra_metadatos)
    pasos.to_csv(DIR / "perfil_pasos_por_imagen.csv", index=False)
    print(pasos.round(3).to_string(index=False), flush=True)
    gil = pasos.groupby(["operacion", "gil"])["pct"].sum().unstack(fill_value=0)
    resumen["fraccion_sin_gil_por_operacion"] = (gil.get("C, libera GIL", 0) / 100).round(3).to_dict()
    resumen["fraccion_mixta_por_operacion"] = (gil.get("Mixto", 0) / 100).round(3).to_dict()
    # Rango de la predicción: conservadora (solo los pasos en C sin GIL son
    # paralelos) y optimista (también los pasos "mixtos" de numpy/scipy).
    resumen["prediccion_hilos_amdahl"] = {
        op: {f"{n}_hilos": [round(amdahl(f, n), 2),
                            round(amdahl(f + resumen["fraccion_mixta_por_operacion"][op], n), 2)]
             for n in (2, 4, 8, 16)}
        for op, f in resumen["fraccion_sin_gil_por_operacion"].items()}
    grafica_pasos(pasos, DIR / "perfil_pasos_por_imagen.png")

    print("\nC. Desglose del entrenamiento en CPU...", flush=True)
    fases, ops, cat = perfil_entrenamiento(args.lotes)
    fases.to_csv(DIR / "perfil_entrenamiento_fases.csv", index=False)
    ops.to_csv(DIR / "perfil_entrenamiento_ops.csv", index=False)
    cat.to_csv(DIR / "perfil_entrenamiento_categorias.csv", index=False)
    print(fases.round(2).to_string(index=False))
    print(cat.round(2).to_string(index=False))
    print(ops.head(15).round(2).to_string(index=False))
    resumen["entrenamiento_hilos_torch"] = torch.get_num_threads()
    guardar_json(resumen, DIR / "resumen_perfil.json")
    print("\nGuardado en", DIR)


if __name__ == "__main__":
    main()
