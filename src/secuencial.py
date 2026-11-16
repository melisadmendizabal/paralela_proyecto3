"""PASO 2 - Línea base secuencial: ejecución y medición.

Ejecuta el pipeline original del Lab 3 de punta a punta, en CPU:
    F1  Ingesta del dataset (metadatos de ~87,000 JPG)
    EDA Imagen promedio por clase + distancias entre clases
    F2  Submuestreo + perceptual hash + verificación de duplicados
    EDA Brillo por clase
    F3  Partición + carga/resize/normalización + codificación de etiquetas
    F4  Entrenamiento de la CNN 2 (15 épocas, lote 64) en CPU
    F5  Evaluación en el conjunto de prueba

Preparación de datos: bucles de Python en un único proceso (un núcleo).
Entrenamiento: PyTorch en CPU (sin GPU), con los hilos que PyTorch usa por
defecto, igual que TensorFlow en el Lab 3 original.

Uso (desde la raíz del proyecto):
    python -m src.secuencial --corridas 5
    python -m src.secuencial --corridas 1 --epochs 1 --salida benchmarks/prueba   # prueba rápida

Salidas en benchmarks/secuencial/ (ver README de esa carpeta).
"""
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from . import config, datos, eda, graficas
from .entrenamiento import a_tensores, entrenar, evaluar
from .medicion import Cronometro, MonitorRecursos, guardar_json, info_entorno
from .modelo import construir_modelo, contar_parametros

FASES = ["F1_ingesta", "EDA_promedios", "F2_limpieza", "EDA_brillo",
         "F3_preprocesamiento", "F4_entrenamiento", "F5_evaluacion"]


def fijar_semillas(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def ejecutar_pipeline(corrida, dir_corrida, args, entrenar_modelo=True):
    """Ejecuta el pipeline completo una vez y guarda todas sus métricas."""
    dir_corrida.mkdir(parents=True, exist_ok=True)
    fijar_semillas(config.SEMILLA)
    device = torch.device("cpu")

    monitor = MonitorRecursos(intervalo=args.intervalo)
    monitor.start()
    crono = Cronometro(monitor)
    metricas = {"corrida": corrida, "inicio": datetime.now().isoformat(timespec="seconds")}
    t_inicio = time.perf_counter()

    # ------------------------------------------------------------------ CPU --
    with crono.fase("F1_ingesta"):
        datos_img = datos.escanear_dataset()
    metricas["imagenes_dataset"] = len(datos_img)

    if not args.sin_eda:
        with crono.fase("EDA_promedios"):
            promedios = eda.promedios_por_clase(datos_img)
            _, pares = eda.distancias_entre_clases(promedios)
        pares.to_csv(dir_corrida / "eda_pares_distancias.csv", index=False)

    with crono.fase("F2_limpieza"):
        submuestra = datos.submuestrear(datos_img)
        submuestra["phash"] = datos.calcular_hashes(submuestra)
        limpia, info_dup = datos.limpiar_duplicados(submuestra)
    metricas["limpieza"] = info_dup

    if not args.sin_eda:
        with crono.fase("EDA_brillo"):
            brillos = eda.calcular_brillos(limpia)
            brillo_clase = eda.brillo_por_clase(limpia, brillos)
        brillo_clase.to_csv(dir_corrida / "eda_brillo_por_clase.csv")

    with crono.fase("F3_preprocesamiento"):
        train_df, val_df, test_df = datos.dividir_train_val_test(limpia)
        X_train, y_train_raw = datos.cargar_y_preprocesar(train_df)
        X_val, y_val_raw = datos.cargar_y_preprocesar(val_df)
        X_test, y_test_raw = datos.cargar_y_preprocesar(test_df)
        codificador, y_train, y_val, y_test = datos.codificar_etiquetas(y_train_raw, y_val_raw, y_test_raw)
        Xtr, ytr = a_tensores(X_train, y_train)
        Xva, yva = a_tensores(X_val, y_val)
        Xte, yte = a_tensores(X_test, y_test)
    metricas["particion"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    metricas["forma_X_train"] = list(X_train.shape)
    del X_train, X_val, X_test

    # --------------------------------------------------------------- MODELO --
    if entrenar_modelo:
        with crono.fase("F4_entrenamiento"):
            fijar_semillas(config.SEMILLA)
            modelo = construir_modelo(config.MODELO, dropout=config.DROPOUT)
            historia = entrenar(modelo, Xtr, ytr, Xva, yva, epochs=args.epochs,
                                batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
                                device=device, semilla=config.SEMILLA)
        pd.DataFrame(historia).to_csv(dir_corrida / "historia_entrenamiento.csv", index=False)
        metricas["modelo"] = {"nombre": config.MODELO, "parametros": contar_parametros(modelo),
                              "epochs": args.epochs, "batch_size": config.BATCH_SIZE,
                              "learning_rate": config.LEARNING_RATE, "dropout": config.DROPOUT,
                              "device": str(device), "hilos_torch": torch.get_num_threads()}

        with crono.fase("F5_evaluacion"):
            test_loss, test_acc, preds = evaluar(modelo, Xte, yte, config.BATCH_SIZE, device)
            matriz = confusion_matrix(y_test, preds)
            reporte = classification_report(y_test, preds, target_names=codificador.classes_, digits=4)
        metricas["test"] = {"loss": test_loss, "accuracy": test_acc,
                            "val_loss_final": historia[-1]["val_loss"],
                            "val_accuracy_final": historia[-1]["val_accuracy"]}
        pd.DataFrame(matriz, index=codificador.classes_, columns=codificador.classes_) \
            .to_csv(dir_corrida / "matriz_confusion.csv")
        (dir_corrida / "reporte_clasificacion.txt").write_text(reporte, encoding="utf-8")
        if args.guardar_modelo:
            config.DIR_MODELOS.mkdir(exist_ok=True)
            torch.save(modelo.state_dict(), config.DIR_MODELOS / "cnn2_secuencial_cpu.pt")
        graficas.curvas_entrenamiento(pd.DataFrame(historia), dir_corrida / "curvas_entrenamiento.png",
                                      f"CNN 2 en CPU — {corrida}")
        print(f"  test_accuracy={test_acc:.4f}  test_loss={test_loss:.4f}", flush=True)

    tiempo_total = time.perf_counter() - t_inicio
    monitor.detener()

    # ------------------------------------------------------------- GUARDAR --
    metricas["tiempos_s"] = crono.tiempos
    metricas["tiempo_total_s"] = tiempo_total
    metricas["recursos_por_fase"] = monitor.resumen_por_fase()
    guardar_json(metricas, dir_corrida / "metricas.json")
    monitor.guardar_csv(dir_corrida / "recursos.csv")
    if monitor.muestras:
        graficas.graficas_de_corrida(dir_corrida, "secuencial")

    fila = {"corrida": corrida,
            "tipo": "medicion" if entrenar_modelo else "calentamiento",
            "inicio": metricas["inicio"]}
    for f in FASES:
        fila[f] = round(crono.tiempos[f], 3) if f in crono.tiempos else None
    fila["total_fases_s"] = round(sum(crono.tiempos.values()), 3)
    fila["total_reloj_s"] = round(tiempo_total, 3)
    fila["test_accuracy"] = metricas.get("test", {}).get("accuracy")
    fila["test_loss"] = metricas.get("test", {}).get("loss")
    fila["epochs"] = args.epochs if entrenar_modelo else None
    fila["hilos_torch"] = torch.get_num_threads()
    return fila


def resumir(tiempos, dir_salida):
    """Media, desviación, mínimo y máximo por fase (solo corridas de medición)."""
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
    if len(med) and "test_accuracy" in med:
        acc = med["test_accuracy"].dropna()
        guardar_json({"corridas": int(len(med)), "test_accuracy_media": float(acc.mean()) if len(acc) else None,
                      "test_accuracy_desv": float(acc.std(ddof=1)) if len(acc) > 1 else None},
                     dir_salida / "resumen_exactitud.json")
    graficas.tiempos_por_fase(resumen, dir_salida / "tiempos_por_fase.png",
                              f"Versión secuencial — tiempo por fase ({len(med)} corridas)")
    return resumen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corridas", type=int, default=5, help="corridas medidas (default 5)")
    p.add_argument("--calentamiento", type=int, default=1,
                   help="corridas previas solo de datos para calentar la caché de disco (no se promedian)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--sin-eda", action="store_true", help="omitir las fases del EDA")
    p.add_argument("--hilos-torch", type=int, default=None, help="hilos de PyTorch en CPU (default: los de PyTorch)")
    p.add_argument("--intervalo", type=float, default=1.0, help="segundos entre muestras del monitor")
    p.add_argument("--salida", default=str(config.DIR_BENCHMARKS / "secuencial"))
    p.add_argument("--guardar-modelo", action="store_true", default=True)
    args = p.parse_args()

    if args.hilos_torch:
        torch.set_num_threads(args.hilos_torch)
    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    entorno = info_entorno()
    entorno["argumentos"] = vars(args)
    guardar_json(entorno, dir_salida / "entorno.json")
    print("Entorno:", {k: entorno[k] for k in ("cpu", "nucleos_fisicos", "nucleos_logicos", "torch_hilos_cpu", "gpu")
                       if k in entorno}, flush=True)

    ruta_tiempos = dir_salida / "tiempos.csv"
    filas = []
    plan = [("calentamiento", i) for i in range(1, args.calentamiento + 1)] + \
           [("medicion", i) for i in range(1, args.corridas + 1)]
    for tipo, i in plan:
        nombre = f"calentamiento_{i:02d}" if tipo == "calentamiento" else f"corrida_{i:02d}"
        print(f"\n===== {nombre} ({datetime.now():%H:%M:%S}) =====", flush=True)
        fila = ejecutar_pipeline(nombre, dir_salida / nombre, args, entrenar_modelo=(tipo == "medicion"))
        filas.append(fila)
        pd.DataFrame(filas).to_csv(ruta_tiempos, index=False)   # se guarda tras cada corrida

    resumen = resumir(pd.DataFrame(filas), dir_salida)
    print("\nResumen (s):")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
