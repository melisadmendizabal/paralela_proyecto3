"""PASO 4 - Diagramas del diseño paralelo para el informe.

Uso:    python -m src.diagramas
Salida: benchmarks/diseno/pipeline.png, benchmarks/diseno/mapeo_conv.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from . import config  # noqa: E402
from .graficas import AZUL, NARANJA, TINTA, TINTA_2, REJILLA  # noqa: E402

DIR = config.DIR_BENCHMARKS / "diseno"
AZUL_CLARO, NARANJA_CLARO = "#cde2fb", "#fbd9c9"


def _caja(ax, x, y, w, h, titulo, texto, borde, fondo):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=borde, facecolor=fondo))
    ax.text(x + w / 2, y + h - 0.16, titulo, ha="center", va="top", fontsize=9.5, weight="bold", color=TINTA)
    ax.text(x + w / 2, y + h - 0.5, texto, ha="center", va="top", fontsize=8, color=TINTA_2, linespacing=1.35)


def _flecha(ax, x0, y0, x1, y1, texto=None, color=TINTA_2, dy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color=color))
    if texto:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, texto, ha="center", va="bottom", fontsize=7.5, color=TINTA)


def pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # Fondos de los dos pipelines
    ax.add_patch(Rectangle((0.05, 0.2), 8.3, 4.0, facecolor="#f4f8fd", edgecolor=REJILLA))
    ax.add_patch(Rectangle((9.1, 0.2), 4.35, 4.0, facecolor="#fdf6f2", edgecolor=REJILLA))
    ax.text(0.2, 4.05, "Pipeline de CPU: preparación de imágenes", fontsize=11, weight="bold", color=AZUL, va="top")
    ax.text(9.25, 4.05, "Pipeline de GPU (CUDA): modelo", fontsize=11, weight="bold", color=NARANJA, va="top")

    w, h, y = 1.42, 1.6, 1.95
    cajas = [
        ("F1 Ingesta", "87,000 archivos\nbloques de rutas\n→ metadatos"),
        ("EDA promedio", "5,800 imágenes\nsumas parciales\npor clase → Σ"),
        ("F2 Limpieza", "17,400 hashes\nbarrera →\nduplicados (sec.)"),
        ("EDA brillo", "17,391 imágenes\n→ brillo\n→ agregación"),
        ("F3 Carga", "17,391 imágenes\n→ uint8 64×64×3\narreglo común"),
    ]
    xs = [0.25 + i * 1.62 for i in range(len(cajas))]
    for x, (t, s) in zip(xs, cajas):
        _caja(ax, x, y, w, h, t, s, AZUL, AZUL_CLARO)
    for a, b in zip(xs[:-1], xs[1:]):
        _flecha(ax, a + w + 0.03, y + h / 2, b - 0.03, y + h / 2)

    # Trabajadores
    ax.text(4.2, 1.45, "Cada fase reparte bloques de imágenes entre N trabajadores (hilos o procesos)",
            ha="center", fontsize=8.5, color=TINTA)
    for i in range(8):
        ax.add_patch(FancyBboxPatch((0.75 + i * 0.88, 0.62), 0.66, 0.46, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    linewidth=1, edgecolor=AZUL, facecolor="white"))
        ax.text(1.08 + i * 0.88, 0.85, f"T{i + 1}" if i < 7 else "…", ha="center", va="center", fontsize=8, color=AZUL)
    ax.text(4.2, 0.32, "A secuencial · B hilos · C procesos · D hilos + Numba · E Python sin GIL",
            ha="center", fontsize=8, color=TINTA_2)

    # Transferencia
    _flecha(ax, xs[-1] + w + 0.03, y + h / 2, 9.3, y + h / 2, "1 copia a la GPU\nuint8, 214 MB", color=TINTA)

    w2 = 1.25
    gx = [9.3, 10.65, 12.0]
    gcajas = [("Normalizar", "÷255 y NCHW\nen la GPU\n(1 kernel)"),
              ("F4 Entrenar", "2,865 pasos\nadelante · atrás\n· Adam"),
              ("F5 Evaluar", "2,609 imágenes\ncuDNN o\nkernels Numba")]
    for x, (t, s) in zip(gx, gcajas):
        _caja(ax, x, y, w2, h, t, s, NARANJA, NARANJA_CLARO)
    for a, b in zip(gx[:-1], gx[1:]):
        _flecha(ax, a + w2 + 0.03, y + h / 2, b - 0.03, y + h / 2)
    ax.text(11.27, 1.45, "Cada capa: un hilo CUDA por valor de salida", ha="center", fontsize=8.5, color=TINTA)
    ax.text(11.27, 1.0, "T1 PyTorch FP32 · T2 datos en GPU", ha="center", fontsize=8, color=TINTA_2)
    ax.text(11.27, 0.68, "T3 precisión mixta · T4 kernels Numba", ha="center", fontsize=8, color=TINTA_2)
    ax.text(11.27, 0.36, "vs. T0 CPU (línea base)", ha="center", fontsize=8, color=TINTA_2)
    fig.savefig(DIR / "pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _rejilla(ax, x0, y0, n, color=REJILLA, lw=0.7):
    for i in range(n + 1):
        ax.plot([x0 + i, x0 + i], [y0, y0 + n], color=color, linewidth=lw)
        ax.plot([x0, x0 + n], [y0 + i, y0 + i], color=color, linewidth=lw)


def mapeo_conv():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")

    # (a) Grid: mapas de salida (N·C_out) de 64×64 divididos en bloques 16×16
    ax = axs[0]
    for k in range(3, 0, -1):
        ax.add_patch(Rectangle((k * 5, k * 5), 64, 64, facecolor="white", edgecolor=REJILLA, linewidth=1))
    ax.add_patch(Rectangle((0, 0), 64, 64, facecolor="white", edgecolor=TINTA_2, linewidth=1.2))
    for i in range(1, 4):
        ax.plot([i * 16, i * 16], [0, 64], color=AZUL, linewidth=1)
        ax.plot([0, 64], [i * 16, i * 16], color=AZUL, linewidth=1)
    ax.add_patch(Rectangle((16, 32), 16, 16, facecolor=NARANJA_CLARO, edgecolor=NARANJA, linewidth=2))
    ax.text(24, 40, "bloque\n(1, 1, z)", ha="center", va="center", fontsize=8, color=TINTA)
    ax.set_xlim(-6, 86)
    ax.set_ylim(-14, 86)
    ax.text(32, -4, "grid.x = ⌈64/16⌉ = 4", ha="center", va="top", fontsize=8.5, color=TINTA)
    ax.text(-3, 32, "grid.y = 4", ha="right", va="center", fontsize=8.5, color=TINTA, rotation=90)
    ax.text(78, 78, "grid.z = N·C_out\n= 64·32 = 2,048", ha="center", va="bottom", fontsize=8.5, color=TINTA)
    ax.set_title("(a) Grid: un bloque por cada 16×16 salidas\nde cada mapa (imagen n, filtro c_out)",
                 fontsize=9.5, color=TINTA, loc="left")

    # (b) Bloque de 16×16 hilos (eje y hacia abajo, como las filas de la imagen)
    ax = axs[1]
    ax.add_patch(Rectangle((0, 0), 16, 2, facecolor=AZUL_CLARO, edgecolor="none"))
    _rejilla(ax, 0, 0, 16)
    ax.add_patch(Rectangle((0, 0), 16, 16, fill=False, edgecolor=NARANJA, linewidth=2))
    ax.add_patch(Rectangle((5, 9), 1, 1, facecolor=AZUL, edgecolor="none"))
    ax.annotate("hilo (tx=5, ty=9)\ncalcula 1 salida:\nΣ 3×3×C_in MAC\n+ sesgo, ReLU", xy=(6, 9.5), xytext=(9.5, 12.8),
                fontsize=8, color=TINTA, arrowprops={"arrowstyle": "-", "color": TINTA_2, "linewidth": 0.8})
    ax.annotate("filas 0–1 = 1 warp de 32 hilos\n(leen posiciones contiguas en x)", xy=(12, 1.5), xytext=(7.5, 5.5),
                fontsize=7.5, color=AZUL, arrowprops={"arrowstyle": "-", "color": AZUL, "linewidth": 0.8})
    ax.text(8, 16.6, "blockDim = (16, 16) = 256 hilos", ha="center", va="top", fontsize=8.5, color=TINTA)
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1.5, 19)
    ax.invert_yaxis()
    ax.set_title("(b) Bloque: un hilo por valor de salida", fontsize=9.5, color=TINTA, loc="left")

    # (c) Memoria compartida: tesela de entrada 18×18 (con halo)
    ax = axs[2]
    ax.add_patch(Rectangle((0, 0), 18, 18, facecolor=NARANJA_CLARO, edgecolor="none", alpha=0.6))
    ax.add_patch(Rectangle((1, 1), 16, 16, facecolor="white", edgecolor="none"))
    _rejilla(ax, 0, 0, 18, lw=0.6)
    ax.add_patch(Rectangle((1, 1), 16, 16, fill=False, edgecolor=NARANJA, linewidth=1.5))
    # El hilo (tx, ty) usa la tesela centrada en (tx+1, ty+1)
    ax.add_patch(Rectangle((5, 9), 3, 3, facecolor=AZUL_CLARO, edgecolor=AZUL, linewidth=1.5))
    ax.add_patch(Rectangle((6, 10), 1, 1, facecolor=AZUL, edgecolor="none"))
    ax.annotate("ventana 3×3 que lee\nel hilo (tx=5, ty=9)", xy=(8, 10.5), xytext=(10.5, 14.2), fontsize=8,
                color=TINTA, arrowprops={"arrowstyle": "-", "color": TINTA_2, "linewidth": 0.8})
    ax.text(9, 18.6, "halo de 1 píxel (padding “same”)", ha="center", va="top", fontsize=7.5, color=NARANJA)
    ax.text(9, 20.0, "tesela (16+2)×(16+2) por canal de entrada = 1,296 B", ha="center", va="top",
            fontsize=8.5, color=TINTA)
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1.5, 21)
    ax.invert_yaxis()
    ax.set_title("(c) Memoria compartida: cada píxel de entrada\nse lee de global 1 vez y se reutiliza ~9 veces",
                 fontsize=9.5, color=TINTA, loc="left")
    fig.tight_layout()
    fig.savefig(DIR / "mapeo_conv.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    pipeline()
    mapeo_conv()
    print("Guardado en", DIR)


if __name__ == "__main__":
    main()
