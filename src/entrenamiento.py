"""Fase del modelo: entrenamiento y evaluación.

Las funciones reciben `device`, de modo que la MISMA lógica sirve para la
versión secuencial (device="cpu") y para la paralela (device="cuda").
Replica el comportamiento de model.fit de Keras: barajado por época,
Adam (eps=1e-7 como Keras), métricas de entrenamiento promediadas por lote
y validación al final de cada época.

Formato de memoria: se usa channels_last (NHWC en memoria, igual que Keras).
En la CPU de pruebas redujo el tiempo por lote ~22 % frente a NCHW, por lo que
se aplica a ambas versiones para que la línea base secuencial sea justa.
"""
import time

import numpy as np
import torch
from torch import nn


def a_tensores(X, y):
    """NHWC float32 (NumPy) -> NCHW float32 (torch), etiquetas int64."""
    X_t = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    return X_t, y_t


CL = torch.channels_last


def _sincronizar(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluar(modelo, X, y, batch_size, device):
    """Devuelve (loss, accuracy, predicciones) sobre un conjunto completo."""
    modelo.eval()
    criterio = nn.CrossEntropyLoss(reduction="sum")
    perdida_total, correctos, preds = 0.0, 0, []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device, memory_format=CL, non_blocking=True)
        yb = y[i:i + batch_size].to(device, non_blocking=True)
        logits = modelo(xb)
        perdida_total += criterio(logits, yb).item()
        pred = logits.argmax(dim=1)
        correctos += (pred == yb).sum().item()
        preds.append(pred.cpu())
    n = len(X)
    return perdida_total / n, correctos / n, torch.cat(preds).numpy()


def entrenar(modelo, X_train, y_train, X_val, y_val, *, epochs, batch_size, lr,
             device, semilla, verbose=True):
    """Entrena el modelo y devuelve el historial por época (lista de dicts)."""
    modelo.to(device, memory_format=CL)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, eps=1e-7)
    criterio = nn.CrossEntropyLoss()
    generador = torch.Generator().manual_seed(semilla)
    n = len(X_train)
    historia = []

    for epoca in range(1, epochs + 1):
        t0 = time.perf_counter()
        modelo.train()
        perm = torch.randperm(n, generator=generador)
        perdida_acum, correctos = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(device, memory_format=CL, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)

            optimizador.zero_grad(set_to_none=True)
            logits = modelo(xb)                 # propagación hacia adelante
            perdida = criterio(logits, yb)      # cálculo de la pérdida
            perdida.backward()                  # retropropagación
            optimizador.step()                  # actualización de pesos (Adam)

            perdida_acum += perdida.item() * len(idx)
            correctos += (logits.argmax(1) == yb).sum().item()
        _sincronizar(device)
        t_train = time.perf_counter() - t0

        val_loss, val_acc, _ = evaluar(modelo, X_val, y_val, batch_size, device)
        _sincronizar(device)
        t_total = time.perf_counter() - t0

        registro = {
            "epoca": epoca,
            "loss": perdida_acum / n,
            "accuracy": correctos / n,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "tiempo_entrenamiento_s": t_train,
            "tiempo_validacion_s": t_total - t_train,
            "tiempo_epoca_s": t_total,
        }
        historia.append(registro)
        if verbose:
            print(f"  época {epoca:2d}/{epochs}  loss={registro['loss']:.4f}  "
                  f"acc={registro['accuracy']:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  ({t_total:.1f} s)", flush=True)
    return historia
