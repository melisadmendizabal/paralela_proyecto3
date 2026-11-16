"""PASO 5 - Versión paralela: pipeline de CPU (preparación) + pipeline de GPU (modelo).

Fases medidas (las de la versión secuencial + las nuevas del diseño, sección 5):
    F0  Creación del pool de trabajadores de CPU
    F1  Ingesta del dataset                      ┐
    EDA Imagen promedio + distancias             │ pipeline de CPU
    F2  Submuestreo + hash + duplicados          │ (src/datos_paralelo.py)
    EDA Brillo por clase                         │
    F3  Partición + carga (uint8) + etiquetas    ┘
    G0  Inicialización de CUDA                   ┐
    G1  Transferencia a la GPU + normalización   │ pipeline de GPU
    F4  Entrenamiento (15 épocas)                │
    F5  Evaluación en prueba                     ┘

Modos de GPU (Tabla 23 del informe):
    t1  datos en la GPU, FP32 (versión principal)
    t2  port directo: el código secuencial con device="cuda" (copia y 2 esperas por lote)
    t3  como t1, con precisión mixta (FP16)

Cada corrida se ejecuta en un PROCESO NUEVO, de modo que todas pagan la
inicialización de CUDA, igual que una ejecución real del programa.

Uso:
    python -m src.paralelo                                   # procesos ×8 + t1, 5 corridas
    python -m src.paralelo --gpu t3 --corridas 5
    python -m src.paralelo --corridas 1 --epochs 1 --salida benchmarks/prueba_paralelo   # prueba
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .energia import desactivar_ecoqos

FASES = ["F0_pool", "F1_ingesta", "EDA_promedios", "F2_limpieza", "EDA_brillo", "F3_preprocesamiento",
         "G0_init_cuda", "G1_transferencia", "F4_entrenamiento", "F5_evaluacion"]


# ============================================================================
# Una corrida (se ejecuta en un proceso hijo)
# ============================================================================
def ejecutar_corrida(nombre, dir_corrida, args, entrenar_modelo, verificar):
    import random

    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    from . import graficas
    from .bench_datos import huella
    from .datos_paralelo import PoolDatos, gil_activo, preparar_datos
    from .entrenamiento import (a_gpu_normalizado, a_tensores, entrenar, entrenar_gpu,
                                evaluar, evaluar_gpu)
    from .medicion import Cronometro, MonitorRecursos, guardar_json
    from .modelo import construir_modelo, contar_parametros

    def fijar_semillas(s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    dir_corrida.mkdir(parents=True, exist_ok=True)
    fijar_semillas(config.SEMILLA)
    torch.backends.cudnn.benchmark = True
    # FP32 estricto: PyTorch usa por defecto TF32 (menos bits de mantisa) en las
    # convoluciones de cuDNN en GPUs Ampere o posteriores; se desactiva para que
    # T1 calcule con la misma precisión que la CPU. T3 usa FP16 de forma explícita.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    monitor = MonitorRecursos(intervalo=args.intervalo)
    monitor.start()
    crono = Cronometro(monitor, sincronizar_cuda=False)   # se sincroniza a mano en las fases de GPU
    metricas = {"corrida": nombre, "inicio": datetime.now().isoformat(timespec="seconds"),
                "datos": args.datos, "trabajadores": args.trabajadores, "gpu": args.gpu}
    t_inicio = time.perf_counter()

    # ------------------------------------------------------- pipeline de CPU --
    with crono.fase("F0_pool"):
        pool = PoolDatos(args.datos, args.trabajadores)
        pool.iniciar()
    try:
        r = preparar_datos(pool, crono, con_eda=not args.sin_eda)
    finally:
        pool.cerrar()
    metricas["limpieza"] = r["info_limpieza"]
    metricas["particion"] = r["particion"]
    metricas["gil_activo"] = gil_activo()
    if not args.sin_eda:
        r["pares"].to_csv(dir_corrida / "eda_pares_distancias.csv", index=False)
        r["brillo_clase"].to_csv(dir_corrida / "eda_brillo_por_clase.csv")
    if verificar and not args.sin_eda:
        ref = json.loads((config.DIR_BENCHMARKS / "datos_paralelo" / "referencia.json").read_text())
        h = huella(r)
        metricas["verificacion_datos"] = {"identicos_al_secuencial": h == ref,
                                          "distintos": [k for k in ref if ref[k] != h.get(k)]}
        print("  verificación de datos:", metricas["verificacion_datos"], flush=True)

    # ------------------------------------------------------- pipeline de GPU --
    if entrenar_modelo:
        dev = torch.device("cuda")
        with crono.fase("G0_init_cuda"):
            torch.cuda.init()
            torch.zeros(1, device=dev)
            torch.cuda.synchronize()
        metricas["dispositivo"] = {"nombre": torch.cuda.get_device_name(0),
                                   "capacidad": ".".join(map(str, torch.cuda.get_device_capability(0))),
                                   "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}

        with crono.fase("G1_transferencia"):
            if args.gpu == "t2":     # port directo: tensores float32 en la CPU, como la línea base
                Xs = {k: a_tensores(r[f"X_{k}"].astype(np.float32) / np.float32(255), r[f"y_{k}"])
                      for k in ("train", "val", "test")}
            else:
                Xs = {k: a_gpu_normalizado(r[f"X_{k}"], r[f"y_{k}"], dev) for k in ("train", "val", "test")}
                torch.cuda.synchronize()
        if verificar and args.gpu != "t2":
            # La normalización en la GPU debe dar exactamente lo mismo que numpy en la CPU
            iguales = torch.equal(Xs["val"][0].cpu(),
                                  torch.from_numpy(r["X_val"].astype(np.float32) / np.float32(255))
                                  .permute(0, 3, 1, 2))
            metricas["verificacion_normalizacion_gpu"] = bool(iguales)
            print("  normalización en GPU idéntica a numpy:", iguales, flush=True)
        y_test = r["y_test"]
        codificador = r["codificador"]
        del r

        with crono.fase("F4_entrenamiento"):
            fijar_semillas(config.SEMILLA)
            modelo = construir_modelo(config.MODELO, dropout=config.DROPOUT)
            if args.gpu == "t2":
                historia = entrenar(modelo, *Xs["train"], *Xs["val"], epochs=args.epochs,
                                    batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
                                    device=dev, semilla=config.SEMILLA)
            else:
                historia = entrenar_gpu(modelo, *Xs["train"], *Xs["val"], epochs=args.epochs,
                                        batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
                                        semilla=config.SEMILLA, amp=(args.gpu == "t3"))
            torch.cuda.synchronize()
        pd.DataFrame(historia).to_csv(dir_corrida / "historia_entrenamiento.csv", index=False)
        metricas["modelo"] = {"nombre": config.MODELO, "parametros": contar_parametros(modelo),
                              "epochs": args.epochs, "batch_size": config.BATCH_SIZE,
                              "learning_rate": config.LEARNING_RATE, "dropout": config.DROPOUT,
                              "precision": "mixta (FP16)" if args.gpu == "t3" else "FP32"}

        with crono.fase("F5_evaluacion"):
            if args.gpu == "t2":
                test_loss, test_acc, preds = evaluar(modelo, *Xs["test"], config.BATCH_SIZE, dev)
            else:
                test_loss, test_acc, preds = evaluar_gpu(modelo, *Xs["test"], config.BATCH_SIZE,
                                                         amp=(args.gpu == "t3"))
            matriz = confusion_matrix(y_test, preds)
            reporte = classification_report(y_test, preds, target_names=codificador.classes_, digits=4)
        metricas["test"] = {"loss": test_loss, "accuracy": test_acc,
                            "val_loss_final": historia[-1]["val_loss"],
                            "val_accuracy_final": historia[-1]["val_accuracy"]}
        metricas["memoria_gpu_max_mb"] = torch.cuda.max_memory_allocated() / 2**20
        pd.DataFrame(matriz, index=codificador.classes_, columns=codificador.classes_) \
            .to_csv(dir_corrida / "matriz_confusion.csv")
        (dir_corrida / "reporte_clasificacion.txt").write_text(reporte, encoding="utf-8")
        if args.guardar_modelo:
            config.DIR_MODELOS.mkdir(exist_ok=True)
            torch.save(modelo.state_dict(), config.DIR_MODELOS / f"cnn2_paralelo_{args.gpu}_{args.epochs}ep.pt")
        graficas.curvas_entrenamiento(pd.DataFrame(historia), dir_corrida / "curvas_entrenamiento.png",
                                      f"CNN 2 en GPU ({args.gpu}) — {nombre}")
        print(f"  test_accuracy={test_acc:.4f}  test_loss={test_loss:.4f}", flush=True)

    tiempo_total = time.perf_counter() - t_inicio
    monitor.detener()
    metricas["tiempos_s"] = crono.tiempos
    metricas["tiempo_total_s"] = tiempo_total
    metricas["recursos_por_fase"] = monitor.resumen_por_fase()
    guardar_json(metricas, dir_corrida / "metricas.json")
    monitor.guardar_csv(dir_corrida / "recursos.csv")
    if monitor.muestras:
        graficas.graficas_de_corrida(dir_corrida, f"paralela ({args.datos} ×{args.trabajadores}, GPU {args.gpu})")

    fila = {"corrida": nombre, "tipo": "medicion" if entrenar_modelo else "calentamiento",
            "inicio": metricas["inicio"], "datos": args.datos, "trabajadores": args.trabajadores,
            "gpu": args.gpu}
    for f in FASES:
        fila[f] = round(crono.tiempos[f], 3) if f in crono.tiempos else None
    fila["total_fases_s"] = round(sum(crono.tiempos.values()), 3)
    fila["total_reloj_s"] = round(tiempo_total, 3)
    fila["test_accuracy"] = metricas.get("test", {}).get("accuracy")
    fila["test_loss"] = metricas.get("test", {}).get("loss")
    fila["epochs"] = args.epochs if entrenar_modelo else None
    return fila


# ============================================================================
# Orquestador
# ============================================================================
def resumir(tiempos, dir_salida, titulo):
    from . import graficas
    from .medicion import guardar_json
    med = tiempos[tiempos["tipo"] == "medicion"]
    columnas = [c for c in FASES + ["total_fases_s", "total_reloj_s"] if c in med and med[c].notna().any()]
    filas = []
    for c in columnas:
        v = med[c].dropna()
        filas.append({"fase": c, "n": len(v), "media_s": v.mean(), "desv_s": v.std(ddof=1) if len(v) > 1 else np.nan,
                      "min_s": v.min(), "max_s": v.max(),
                      "pct_del_total": 100 * v.mean() / med["total_fases_s"].mean() if c in FASES else 100.0})
    resumen = pd.DataFrame(filas).round(3)
    resumen.to_csv(dir_salida / "resumen_tiempos.csv", index=False)
    acc = med["test_accuracy"].dropna()
    guardar_json({"corridas": int(len(med)), "test_accuracy_media": float(acc.mean()) if len(acc) else None,
                  "test_accuracy_desv": float(acc.std(ddof=1)) if len(acc) > 1 else None},
                 dir_salida / "resumen_exactitud.json")
    graficas.tiempos_por_fase(resumen, dir_salida / "tiempos_por_fase.png", titulo)
    return resumen


def registrar_dispositivo(dir_salida):
    """Evidencia de que el hardware fue reconocido: Numba, PyTorch y nvidia-smi."""
    import contextlib
    import io

    import torch

    from . import numba_compat  # noqa: F401  (antes de numba.cuda)
    from numba import cuda
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cuda.detect()
    lineas = ["== numba.cuda.detect() ==", buf.getvalue(),
              "== PyTorch ==",
              f"torch {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}",
              f"dispositivo: {torch.cuda.get_device_name(0)}  capacidad {torch.cuda.get_device_capability(0)}",
              "", "== nvidia-smi ==",
              subprocess.run(["nvidia-smi"], capture_output=True, text=True, errors="replace").stdout]
    texto = "\n".join(lineas)
    (dir_salida / "dispositivo.txt").write_text(texto, encoding="utf-8")
    print(texto, flush=True)


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", default="procesos", choices=["secuencial", "hilos", "procesos", "numba"])
    p.add_argument("--trabajadores", type=int, default=8)
    p.add_argument("--gpu", default="t1", choices=["t1", "t2", "t3"])
    p.add_argument("--corridas", type=int, default=5)
    p.add_argument("--calentamiento", type=int, default=1)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--sin-eda", action="store_true")
    p.add_argument("--intervalo", type=float, default=1.0)
    p.add_argument("--salida", default=None, help="default: benchmarks/paralelo/<datos>_<N>_<gpu>")
    p.add_argument("--guardar-modelo", action="store_true", default=True)
    # uso interno: el orquestador lanza cada corrida con estos argumentos
    p.add_argument("--_corrida", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_tipo", default="medicion", help=argparse.SUPPRESS)
    p.add_argument("--_verificar", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = argumentos()
    ecoqos = desactivar_ecoqos()
    dir_salida = Path(args.salida or config.DIR_BENCHMARKS / "paralelo" / f"{args.datos}_{args.trabajadores}_{args.gpu}")

    if args._corrida:                                     # ---- proceso hijo: una corrida
        fila = ejecutar_corrida(args._corrida, dir_salida / args._corrida, args,
                                entrenar_modelo=(args._tipo == "medicion"), verificar=args._verificar)
        fila["ecoqos_desactivado"] = ecoqos
        (dir_salida / args._corrida / "fila.json").write_text(json.dumps(fila), encoding="utf-8")
        return

    # ---- orquestador
    from .medicion import guardar_json, info_entorno
    dir_salida.mkdir(parents=True, exist_ok=True)
    entorno = info_entorno()
    entorno["argumentos"] = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    entorno["ecoqos_desactivado"] = ecoqos
    guardar_json(entorno, dir_salida / "entorno.json")
    registrar_dispositivo(dir_salida)

    plan = [("calentamiento", i) for i in range(1, args.calentamiento + 1)] + \
           [("medicion", i) for i in range(1, args.corridas + 1)]
    filas = []
    base = [sys.executable, "-u", "-m", "src.paralelo", "--datos", args.datos,
            "--trabajadores", str(args.trabajadores), "--gpu", args.gpu, "--epochs", str(args.epochs),
            "--intervalo", str(args.intervalo), "--salida", str(dir_salida)]
    if args.sin_eda:
        base.append("--sin-eda")
    for tipo, i in plan:
        nombre = f"{tipo}_{i:02d}" if tipo == "calentamiento" else f"corrida_{i:02d}"
        print(f"\n===== {nombre} ({datetime.now():%H:%M:%S}) =====", flush=True)
        cmd = base + ["--_corrida", nombre, "--_tipo", tipo]
        if tipo == "medicion" and i == 1:
            cmd.append("--_verificar")
        subprocess.run(cmd, check=True, cwd=config.RAIZ)
        filas.append(json.loads((dir_salida / nombre / "fila.json").read_text(encoding="utf-8")))
        pd.DataFrame(filas).to_csv(dir_salida / "tiempos.csv", index=False)

    resumen = resumir(pd.DataFrame(filas), dir_salida,
                      f"Versión paralela ({args.datos} ×{args.trabajadores}, GPU {args.gpu}) — "
                      f"tiempo por fase ({args.corridas} corridas)")
    print("\nResumen (s):")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
