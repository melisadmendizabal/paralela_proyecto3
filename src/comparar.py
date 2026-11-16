"""PASO 6 - Comparación de tiempos y aceleración (speedup).

  datos     barrido del pipeline de CPU (benchmarks/datos_paralelo/bench_datos*.csv):
            tiempo medio por variante y número de trabajadores, aceleración frente a
            la variante secuencial medida en las mismas condiciones, y gráfica.
  total     versión secuencial (benchmarks/secuencial) frente a cada configuración
            de la versión paralela (benchmarks/paralelo/*): tiempo y aceleración por fase.

Uso:    python -m src.comparar datos
        python -m src.comparar total
Salida: benchmarks/comparacion/
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import config

DIR = config.DIR_BENCHMARKS / "comparacion"
FASES_DATOS = ["F1_ingesta", "EDA_promedios", "F2_limpieza", "EDA_brillo", "F3_preprocesamiento"]
COLORES = {"hilos": "#2a78d6", "procesos": "#eb6834", "numba": "#1baf7a", "hilos (sin GIL)": "#eda100"}
ETIQUETAS = {"hilos": "B Hilos", "procesos": "C Procesos", "numba": "D Hilos + Numba",
             "hilos (sin GIL)": "E Hilos, Python sin GIL"}


def cargar_barrido():
    d = config.DIR_BENCHMARKS / "datos_paralelo"
    partes = []
    for f in sorted(d.glob("bench_datos*.csv")):
        if "prueba" in f.name:
            continue
        df = pd.read_csv(f)
        if "313t" in f.name or (df.get("gil_activo") == False).any():  # noqa: E712
            df["variante"] = df["variante"].replace({"hilos": "hilos (sin GIL)", "secuencial": "secuencial (3.13t)"})
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def resumen_datos(df):
    cols = ["F0_pool"] + FASES_DATOS + ["total_s", "total_sin_pool_s"]
    # Medida central: MEDIANA de las repeticiones. F1 lee 87,000 archivos y, con la RAM
    # casi llena, algunas corridas encuentran parte de ellos fuera de la caché de disco
    # (picos de 17–30 s frente a 6–7 s, en cualquier variante); la mediana no se deja
    # arrastrar por esos picos. La media y la desviación se guardan también.
    g = df.groupby(["variante", "trabajadores"])[cols]
    med, media, desv = g.median(), g.mean(), g.std(ddof=1)
    res = (med.add_suffix("_mediana").join(media[["total_s"]].add_suffix("_media"))
           .join(desv[["total_s"]].add_suffix("_desv")).reset_index())
    base = res.loc[res["variante"] == "secuencial"].iloc[0]
    res["aceleracion_total"] = base["total_s_mediana"] / res["total_s_mediana"]
    res["aceleracion_sin_pool"] = base["total_sin_pool_s_mediana"] / res["total_sin_pool_s_mediana"]
    for f in FASES_DATOS:
        res[f"aceleracion_{f}"] = base[f"{f}_mediana"] / res[f"{f}_mediana"]
    if "secuencial (3.13t)" in set(res["variante"]):   # sin GIL: también frente a su propio secuencial
        b2 = res.loc[res["variante"] == "secuencial (3.13t)"].iloc[0]
        m = res["variante"] == "hilos (sin GIL)"
        res.loc[m, "aceleracion_vs_secuencial_313t"] = b2["total_s_mediana"] / res.loc[m, "total_s_mediana"]
    return res


def grafica_barrido(res, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .graficas import REJILLA, TINTA, TINTA_2
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, col, titulo in ((axs[0], "aceleracion_total", "Fases de datos completas (incluye crear el pool)"),
                            (axs[1], "aceleracion_F2_limpieza", "Aceleración de F2 (perceptual hash)")):
        maximo, finales = 1, []
        for v, color in COLORES.items():
            s = res[res["variante"] == v].sort_values("trabajadores")
            if s.empty:
                continue
            ax.plot(s["trabajadores"], s[col], color=color, linewidth=2, marker="o", markersize=5, label=ETIQUETAS[v])
            finales.append([s.iloc[-1][col], s.iloc[-1]["trabajadores"], color])
            maximo = max(maximo, s[col].max())
        # Etiquetas al final de cada línea, separadas si quedan muy cerca
        finales.sort()
        hueco = maximo * 1.15 * 0.06
        pos = []
        for y, _, _ in finales:
            pos.append(max(y, pos[-1] + hueco) if pos else y)
        for (y, x, color), yp in zip(finales, pos):
            ax.annotate(f"{y:.1f}×", (x, y), xytext=(x + 0.5, yp), va="center", fontsize=8, color=TINTA)
        ax.axhline(1, color=TINTA_2, linewidth=0.8, linestyle=":")
        ax.text(1, 1, " secuencial", va="bottom", fontsize=8, color=TINTA_2)
        ax.set_xticks([1, 2, 4, 8, 12, 16])
        ax.set_xlim(0.5, 18)
        ax.set_ylim(0, maximo * 1.15)
        ax.set_xlabel("Trabajadores", color=TINTA_2)
        ax.set_ylabel("Aceleración (×), mediana de 3 corridas", color=TINTA_2)
        ax.set_title(titulo, color=TINTA, loc="left", fontsize=10.5)
        ax.grid(axis="y", color=REJILLA, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(REJILLA)
        ax.tick_params(colors=TINTA_2, labelsize=9)
    axs[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def comparar_datos():
    DIR.mkdir(parents=True, exist_ok=True)
    df = cargar_barrido()
    if "verificado" in df:
        malos = df[df["verificado"] == False]  # noqa: E712
        print("Corridas con datos distintos al secuencial:", len(malos))
    res = resumen_datos(df)
    res.round(4).to_csv(DIR / "datos_resumen.csv", index=False)
    grafica_barrido(res, DIR / "datos_aceleracion.png")
    cols = ["variante", "trabajadores", "total_s_mediana", "total_s_media", "total_s_desv", "F0_pool_mediana",
            "aceleracion_total",
            "aceleracion_sin_pool"] + [f"aceleracion_{f}" for f in FASES_DATOS]
    print(res[cols].round(2).to_string(index=False))


# ----------------------------------------------------------------------------
def comparar_total():
    DIR.mkdir(parents=True, exist_ok=True)
    sec = pd.read_csv(config.DIR_BENCHMARKS / "secuencial" / "resumen_tiempos.csv").set_index("fase")
    filas = []
    for d in sorted((config.DIR_BENCHMARKS / "paralelo").glob("*/resumen_tiempos.csv")):
        par = pd.read_csv(d).set_index("fase")
        acc = json.loads((d.parent / "resumen_exactitud.json").read_text())
        conf = d.parent.name
        for fase in par.index:
            filas.append({"configuracion": conf, "fase": fase, "paralelo_media_s": par.loc[fase, "media_s"],
                          "paralelo_desv_s": par.loc[fase, "desv_s"],
                          "secuencial_media_s": sec.loc[fase, "media_s"] if fase in sec.index else np.nan,
                          "exactitud_paralelo": acc.get("test_accuracy_media")})
    res = pd.DataFrame(filas)
    res["aceleracion"] = res["secuencial_media_s"] / res["paralelo_media_s"]
    res.round(4).to_csv(DIR / "total_por_fase.csv", index=False)
    print(res.round(3).to_string(index=False))
    grafica_composicion(sec, res, DIR / "total_composicion.png")
    grafica_aceleracion_fases(res, DIR / "total_aceleracion_fases.png")
    grafica_kernels(DIR / "kernels_por_capa.png")


GRUPOS = [("Preparación de datos (CPU)", ["F0_pool"] + FASES_DATOS, "#2a78d6"),
          ("Inicio de CUDA + copia a GPU", ["G0_init_cuda", "G1_transferencia"], "#eda100"),
          ("F4 Entrenamiento", ["F4_entrenamiento"], "#eb6834"),
          ("F5 Evaluación", ["F5_evaluacion"], "#1baf7a")]
NOMBRES_CONF = {"procesos_8_t2": "Paralela T2\n(port directo, FP32)", "procesos_8_t1": "Paralela T1\n(GPU, FP32)",
                "procesos_8_t3": "Paralela T3\n(GPU, precisión mixta)"}


def _estilo(ax):
    from .graficas import REJILLA, TINTA_2
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.set_axisbelow(True)


def grafica_composicion(sec, res, ruta):
    """Barra apilada por configuración: en qué se va el tiempo total."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .graficas import REJILLA, TINTA, TINTA_2
    confs = [("Secuencial\n(CPU)", {f: sec.loc[f, "media_s"] for f in sec.index})]
    for c in ("procesos_8_t2", "procesos_8_t1", "procesos_8_t3"):
        r = res[res["configuracion"] == c]
        if len(r):
            confs.append((NOMBRES_CONF[c], dict(zip(r["fase"], r["paralelo_media_s"]))))
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, porcentaje in ((axs[0], False), (axs[1], True)):
        for y, (nombre, t) in enumerate(confs[::-1]):
            total = sum(t.get(f, 0) for _, fs, _ in GRUPOS for f in fs)
            izq = 0
            for etiqueta, fs, color in GRUPOS:
                v = sum(t.get(f, 0) for f in fs)
                w = 100 * v / total if porcentaje else v
                ax.barh(y, w, left=izq, color=color, height=0.62, edgecolor="white", linewidth=1.5,
                        label=etiqueta if y == 0 else None)
                if porcentaje and w >= 7:
                    ax.text(izq + w / 2, y, f"{w:.0f} %", ha="center", va="center", fontsize=8, color="white")
                izq += w
            if not porcentaje:
                ax.text(izq + 8, y, f"{total:.1f} s", va="center", fontsize=9, color=TINTA)
        ax.set_yticks(range(len(confs)))
        ax.set_yticklabels([n for n, _ in confs[::-1]], fontsize=8.5)
        ax.grid(axis="x", color=REJILLA, linewidth=0.8)
        _estilo(ax)
    axs[0].set_xlim(0, max(sum(t.get(f, 0) for _, fs, _ in GRUPOS for f in fs) for _, t in confs) * 1.15)
    axs[0].set_xlabel("Tiempo total (s), media de 5 corridas", color=TINTA_2)
    axs[0].set_title("Tiempo total por versión", color=TINTA, loc="left", fontsize=11)
    axs[1].set_xlim(0, 100)
    axs[1].set_yticklabels([])
    axs[1].set_xlabel("% del tiempo total", color=TINTA_2)
    axs[1].set_title("Composición: la preparación de datos pasa a pesar más", color=TINTA, loc="left", fontsize=11)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafica_aceleracion_fases(res, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .graficas import NOMBRES_FASE, REJILLA, TINTA, TINTA_2
    fases = FASES_DATOS + ["F4_entrenamiento", "F5_evaluacion", "total_fases_s"]
    nombres = ["F1\nIngesta", "EDA\nImagen promedio", "F2\nLimpieza (hash)", "EDA\nBrillo", "F3\nCarga", "F4\nEntrenamiento", "F5\nEvaluación", "Total\ndel programa"]
    confs = [("procesos_8_t1", "T1 GPU FP32", "#2a78d6"), ("procesos_8_t3", "T3 GPU precisión mixta", "#eb6834")]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ancho = 0.38
    for k, (c, etq, color) in enumerate(confs):
        r = res[res["configuracion"] == c].set_index("fase")
        vals = [r.loc[f, "aceleracion"] if f in r.index else np.nan for f in fases]
        xs = np.arange(len(fases)) + (k - 0.5) * ancho
        ax.bar(xs, vals, width=ancho, color=color, label=etq, edgecolor="white", linewidth=1.5)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.4, f"{v:.1f}×", ha="center", va="bottom", fontsize=7.5, color=TINTA)
    ax.axhline(1, color=TINTA_2, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(fases)))
    ax.set_xticklabels(nombres, fontsize=8.5)
    ax.set_ylabel("Aceleración (×) = T secuencial / T paralelo", color=TINTA_2)
    ax.set_title("Aceleración por fase (procesos ×8 en la CPU + GPU)", color=TINTA, loc="left", fontsize=11)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def grafica_kernels(ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .graficas import REJILLA, TINTA, TINTA_2
    k = pd.read_csv(config.DIR_BENCHMARKS / "kernels" / "kernels_capas.csv")
    k = k[k["capa"].str.startswith("Conv")]
    impls = [("Numba v1, compartida", "16x16", "Numba v1 (memoria compartida, 16×16)", "#2a78d6"),
             ("Numba, sin compartida", "16x16", "Numba sin memoria compartida (16×16)", "#1baf7a"),
             ("Numba v2, 8 filtros por hilo", "16x16", "Numba v2 (8 filtros por hilo)", "#eda100"),
             ("PyTorch GPU (cuDNN)", "-", "PyTorch GPU (cuDNN)", "#eb6834"),
             ("PyTorch CPU", "-", "PyTorch CPU (10 hilos)", "#e87ba4")]
    capas = list(dict.fromkeys(k["capa"]))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ancho = 0.16
    for j, (impl, bloque, etq, color) in enumerate(impls):
        vals = [k[(k["capa"] == c) & (k["implementacion"] == impl) & (k["bloque"] == bloque)]["ms"].mean()
                for c in capas]
        xs = np.arange(len(capas)) + (j - 2) * ancho
        ax.bar(xs, vals, width=ancho, color=color, label=etq, edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(capas)))
    ax.set_xticklabels(capas, fontsize=9)
    ax.set_ylabel("ms por lote de 64 imágenes (mediana)", color=TINTA_2)
    ax.set_title("Convolución + ReLU: kernels propios frente a PyTorch", color=TINTA, loc="left", fontsize=11)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("que", choices=["datos", "total"])
    a = p.parse_args()
    comparar_datos() if a.que == "datos" else comparar_total()


if __name__ == "__main__":
    main()
