"""Micro-benchmark: tiempo por lote de entrenamiento de la CNN 2 en CPU según
el número de hilos de PyTorch y el formato de memoria (NCHW vs channels_last).

Justifica la configuración de la línea base secuencial (hilos por defecto +
channels_last). Usa datos aleatorios con la forma real (lote 64, 3x64x64).

Uso:  python -m src.bench_hilos_cpu
Salida: benchmarks/secuencial/bench_hilos_cpu.csv
"""
import time

import pandas as pd
import torch
from torch import nn

from . import config
from .energia import desactivar_ecoqos
from .modelo import CNN2

LOTES_MEDIDOS = 15
LOTES_CALENTAMIENTO = 3


def medir(hilos, channels_last):
    torch.set_num_threads(hilos)
    torch.manual_seed(config.SEMILLA)
    total = LOTES_MEDIDOS + LOTES_CALENTAMIENTO
    X = torch.rand(config.BATCH_SIZE * total, 3, 64, 64)
    y = torch.randint(0, 29, (config.BATCH_SIZE * total,))
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    modelo = CNN2().to(memory_format=fmt)
    opt = torch.optim.Adam(modelo.parameters(), lr=config.LEARNING_RATE, eps=1e-7)
    crit = nn.CrossEntropyLoss()
    tiempos = []
    for i in range(total):
        xb = X[i * 64:(i + 1) * 64].contiguous(memory_format=fmt)
        yb = y[i * 64:(i + 1) * 64]
        t0 = time.perf_counter()
        opt.zero_grad()
        crit(modelo(xb), yb).backward()
        opt.step()
        if i >= LOTES_CALENTAMIENTO:
            tiempos.append(time.perf_counter() - t0)
    return sum(tiempos) / len(tiempos)


def main():
    print("EcoQoS desactivado:", desactivar_ecoqos())
    filas = []
    for hilos in (1, 2, 4, 6, 8, 10, 12, 16):
        for cl in (False, True):
            ms = medir(hilos, cl) * 1000
            filas.append({"hilos": hilos, "formato": "channels_last" if cl else "NCHW",
                          "ms_por_lote": round(ms, 1),
                          "s_por_epoca_estimado": round(ms * 191 / 1000, 1)})
            print(filas[-1], flush=True)
    salida = config.DIR_BENCHMARKS / "secuencial" / "bench_hilos_cpu.csv"
    pd.DataFrame(filas).to_csv(salida, index=False)
    print("guardado en", salida)


if __name__ == "__main__":
    main()
