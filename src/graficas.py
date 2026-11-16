"""Gráficas de evidencia (matplotlib, PNG para el informe).

Paleta: azul = serie principal / CPU, naranja = segunda serie (validación o GPU).
Texto en tinta neutra; rejilla y ejes recesivos; una sola escala por eje.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

NOMBRES_FASE = {
    "F0_pool": "F0 Creación del pool de CPU",
    "F1_ingesta": "F1 Ingesta (metadatos)",
    "EDA_promedios": "EDA Imagen promedio + distancias",
    "F2_limpieza": "F2 Submuestreo + hash + duplicados",
    "EDA_brillo": "EDA Brillo por clase",
    "F3_preprocesamiento": "F3 Split + carga + resize + normalización",
    "G0_init_cuda": "G0 Inicialización de CUDA",
    "G1_transferencia": "G1 Transferencia a GPU + normalización",
    "F4_entrenamiento": "F4 Entrenamiento",
    "F5_evaluacion": "F5 Evaluación en prueba",
}


def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(REJILLA)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def tiempos_por_fase(resumen, ruta, titulo):
    """Barras horizontales: media ± desviación estándar por fase."""
    df = resumen[resumen["fase"].isin(NOMBRES_FASE)].copy()
    df["etiqueta"] = df["fase"].map(NOMBRES_FASE)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 1.4))
    ax.barh(df["etiqueta"], df["media_s"], xerr=df["desv_s"], color=AZUL, height=0.6,
            error_kw={"ecolor": TINTA_2, "capsize": 3, "elinewidth": 1})
    for y, (m, s) in enumerate(zip(df["media_s"], df["desv_s"])):
        ax.text(m + (s if pd.notna(s) else 0) + df["media_s"].max() * 0.01, y,
                f"{m:.1f} s", va="center", fontsize=9, color=TINTA)
    ax.set_xlabel("Tiempo (s) — media ± desviación estándar", color=TINTA_2)
    ax.set_title(titulo, color=TINTA, loc="left", fontsize=12)
    ax.set_xlim(0, df["media_s"].max() * 1.18)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def recursos_corrida(recursos, ruta, titulo, t_max=None):
    """Mapa de calor de uso por núcleo + uso de GPU, con las fases marcadas.
    t_max: si se indica, muestra solo los primeros t_max segundos (zoom)."""
    if t_max is not None:
        recursos = recursos[recursos["t_s"] <= t_max]
    cols = [c for c in recursos.columns if c.startswith("cpu") and c[3:-4].isdigit()]
    t = recursos["t_s"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    mat = recursos[cols].to_numpy().T
    paso = np.median(np.diff(t)) if len(t) > 1 else 1
    im = ax1.imshow(mat, aspect="auto", cmap=SECUENCIAL, vmin=0, vmax=100, interpolation="nearest",
                    extent=[t[0] - paso / 2, t[-1] + paso / 2, len(cols) - 0.5, -0.5])
    ax1.set_ylabel("Núcleo lógico", color=TINTA_2)
    ax1.set_yticks(range(len(cols)))
    ax1.set_yticklabels([c[3:-4] for c in cols], fontsize=7)
    cb = fig.colorbar(im, ax=[ax1, ax2], fraction=0.025, pad=0.01)
    cb.set_label("Uso de CPU (%)", color=TINTA_2)

    # NVML no responde cuando la GPU de la laptop está en reposo: se grafica como 0 %
    gpu = recursos["gpu_util_pct"].astype(float)
    sin_lectura = gpu.isna()
    ax2.plot(t, gpu.fillna(0), color=NARANJA, linewidth=2)
    if gpu.fillna(0).max() < 1:
        ax2.text(0.5, 0.5, "GPU sin uso durante toda la corrida"
                 + (" (en reposo, sin lectura NVML)" if sin_lectura.mean() > 0.5 else ""),
                 transform=ax2.transAxes, ha="center", va="center", fontsize=9, color=TINTA_2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("GPU (%)", color=TINTA_2)
    ax2.set_xlabel("Tiempo desde el inicio de la corrida (s)", color=TINTA_2)
    ax2.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(colors=TINTA_2, labelsize=9)

    # Bandas de fase
    fases = recursos["fase"].to_numpy()
    inicio = 0
    for i in range(1, len(fases) + 1):
        if i == len(fases) or fases[i] != fases[inicio]:
            f = fases[inicio]
            if f in NOMBRES_FASE:
                x0, x1 = t[inicio] - paso / 2, t[i - 1] + paso / 2
                for ax in (ax1, ax2):
                    ax.axvline(x0, color=TINTA_2, linewidth=0.6, linestyle=":")
                if x1 - x0 > (t[-1] - t[0]) * 0.04:
                    etiqueta = "EDA" if f.startswith("EDA") else f.split("_")[0]
                    ax1.text((x0 + x1) / 2, -0.9, etiqueta, ha="center", va="bottom",
                             fontsize=8, color=TINTA)
            inicio = i
    ax1.set_title(titulo, color=TINTA, loc="left", fontsize=12, pad=24)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficas_de_corrida(dir_corrida, version):
    """(Re)genera las gráficas de recursos de una corrida a partir de recursos.csv."""
    from pathlib import Path
    dir_corrida = Path(dir_corrida)
    recursos = pd.read_csv(dir_corrida / "recursos.csv")
    nombre = dir_corrida.name
    recursos_corrida(recursos, dir_corrida / "uso_recursos.png",
                     f"Versión {version} — uso de CPU por núcleo y de GPU ({nombre})")
    datos = recursos[recursos["fase"].isin(["F0_pool", "F1_ingesta", "EDA_promedios", "F2_limpieza",
                                            "EDA_brillo", "F3_preprocesamiento"])]
    if len(datos):
        recursos_corrida(recursos, dir_corrida / "uso_recursos_fases_datos.png",
                         f"Versión {version} — fases de preparación de datos ({nombre})",
                         t_max=float(datos["t_s"].max()) + 10)


def curvas_entrenamiento(historia, ruta, titulo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    e = historia["epoca"]
    for ax, k, nombre in ((ax1, "loss", "Pérdida"), (ax2, "accuracy", "Exactitud")):
        ax.plot(e, historia[k], color=AZUL, linewidth=2, marker="o", markersize=4, label="Entrenamiento")
        ax.plot(e, historia[f"val_{k}"], color=NARANJA, linewidth=2, marker="o", markersize=4,
                label="Validación")
        ax.set_title(nombre, color=TINTA, loc="left", fontsize=11)
        ax.set_xlabel("Época", color=TINTA_2)
        ax.grid(axis="y", color=REJILLA, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=TINTA_2, labelsize=9)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(titulo, color=TINTA, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
