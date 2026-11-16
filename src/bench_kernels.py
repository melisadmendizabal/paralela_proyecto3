"""PASO 5 - Kernels propios con Numba CUDA: corrección, configuración y fase F5.

1. Por capa (lote de 64, pesos reales del modelo entrenado):
   - error máximo frente a PyTorch (misma entrada, mismos pesos, FP32 estricto);
   - tiempo con bloques 8×8, 16×16 y 32×32 (con memoria compartida) y 16×16 sin
     memoria compartida; capa densa con teselas de 8, 16 y 32;
   - comparación con PyTorch en GPU (cuDNN / cuBLAS) y en CPU;
   - registros por hilo y memoria compartida de cada kernel -> ocupación real.
2. Fase F5 completa (2,609 imágenes de prueba): kernels propios frente a PyTorch
   en GPU y en CPU; exactitud y coincidencia de predicciones.

Uso:    python -m src.bench_kernels
Salida: benchmarks/kernels/
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from . import config
from .energia import desactivar_ecoqos

DIR = config.DIR_BENCHMARKS / "kernels"
SM, HILOS_SM, BLOQUES_SM, REGS_SM, SHARED_SM = 20, 1536, 24, 65536, 102400

CONVS = [("Conv 1", "features.0", 3, 32, 64), ("Conv 2", "features.2", 32, 32, 64),
         ("Conv 3", "features.5", 32, 64, 32), ("Conv 4", "features.7", 64, 64, 32),
         ("Conv 5", "features.10", 64, 128, 16)]


def ocupacion(hilos_bloque, regs, shared):
    """Ocupación teórica (capacidad 8.9) a partir de los recursos reales del kernel."""
    warps = math.ceil(hilos_bloque / 32)
    lim = [BLOQUES_SM, HILOS_SM // hilos_bloque]
    if regs:
        regs_warp = math.ceil(regs * 32 / 256) * 256          # asignación por warp en unidades de 256
        lim.append(REGS_SM // (regs_warp * warps))
    if shared is not None:
        lim.append(SHARED_SM // (shared + 1024))               # 1 KB reservado por bloque
    bloques = max(0, min(lim))
    return bloques, bloques * hilos_bloque / HILOS_SM


def cronometrar(fn, sincronizar, reps):
    """Mediana de reps ejecuciones (ms). La primera llamada (compilación JIT) se mide aparte."""
    t0 = time.perf_counter()
    fn()
    sincronizar()
    primera = (time.perf_counter() - t0) * 1000
    for _ in range(2):
        fn()
    sincronizar()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sincronizar()
        tiempos.append((time.perf_counter() - t0) * 1000)
    return float(np.median(tiempos)), primera


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeticiones", type=int, default=20)
    p.add_argument("--repeticiones-cpu", type=int, default=5)
    p.add_argument("--modelo", default=str(config.DIR_MODELOS / "cnn2_paralelo_t1_15ep.pt"),
                   help="pesos entrenados (15 épocas) para la F5")
    args = p.parse_args()
    ecoqos = desactivar_ecoqos()
    DIR.mkdir(parents=True, exist_ok=True)

    # kernels_numba ANTES que torch: conflicto de nvJitLink (ver src/numba_compat.py)
    from . import kernels_numba as K
    from numba import cuda
    import torch
    import torch.nn.functional as F

    from .modelo import CNN2

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    sync_gpu = torch.cuda.synchronize
    estado = torch.load(args.modelo, map_location="cpu")
    torch.manual_seed(config.SEMILLA)
    N = config.BATCH_SIZE
    filas, ocup = [], []

    # ------------------------------------------------------------ 1. por capa
    for nombre, clave, cin, cout, lado in CONVS:
        x = torch.rand(N, cin, lado, lado, device=dev)
        w = estado[f"{clave}.weight"].to(dev).contiguous()
        b = estado[f"{clave}.bias"].to(dev).contiguous()
        ref = F.relu(F.conv2d(x, w, b, padding=1))
        xn, wn, bn = cuda.as_cuda_array(x), cuda.as_cuda_array(w), cuda.as_cuda_array(b)
        y = cuda.device_array((N, cout, lado, lado), dtype=np.float32)
        configs = [("Numba v1, compartida", 8, True, None, 0), ("Numba v1, compartida", 16, True, None, 0),
                   ("Numba v1, compartida, máx. 40 registros", 16, True, 40, 0),
                   ("Numba v1, compartida", 32, True, None, 0),
                   ("Numba v1, compartida, máx. 64 registros", 32, True, 64, 0),
                   ("Numba, sin compartida", 16, False, None, 0),
                   ("Numba v2, 8 filtros por hilo", 16, True, None, 8)]
        for impl, B, compartida, max_reg, multi in configs:
            if multi:
                kern = K.kernel_conv_multi(B, multi)
            else:
                kern = K.kernel_conv(B, max_reg) if compartida else K.kernel_conv_global(B)
            grid = (math.ceil(lado / B), math.ceil(lado / B), N * cout // (multi or 1))
            try:
                ms, primera = cronometrar(lambda: K.lanzar_conv(xn, wn, bn, y, B, compartida, max_reg=max_reg,
                                                                multi=multi),
                                          cuda.synchronize, args.repeticiones)
                err = float((torch.as_tensor(y, device=dev) - ref).abs().max())
                estado_lanz = "ok"
            except Exception as e:                       # p. ej. CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
                ms, primera, err = float("nan"), float("nan"), float("nan")
                estado_lanz = type(e).__name__ + ": " + str(e).split(":")[0][:60]
            regs, sh = K.registros_por_hilo(kern), K.memoria_compartida(kern)
            bl, occ = ocupacion(B * B, regs, sh)
            filas.append({"capa": nombre, "implementacion": impl, "bloque": f"{B}x{B}", "ms": ms,
                          "primera_llamada_ms": primera, "error_max": err, "lanzamiento": estado_lanz})
            ocup.append({"capa": nombre, "implementacion": impl, "bloque": f"{B}x{B}", "grid": str(grid),
                         "bloques_lanzados": int(np.prod(grid)), "registros_por_hilo": regs,
                         "memoria_compartida_B": sh, "bloques_por_SM": bl, "ocupacion": occ,
                         "lanzamiento": estado_lanz})
            print(f"{nombre} {impl:40s} {B:2d}x{B:<2d} {ms:7.3f} ms  err={err:.2e}  regs={regs} "
                  f"shared={sh}  ocupación={occ:.0%}  {estado_lanz}", flush=True)
        ms, _ = cronometrar(lambda: F.relu(F.conv2d(x, w, b, padding=1)), sync_gpu, args.repeticiones)
        filas.append({"capa": nombre, "implementacion": "PyTorch GPU (cuDNN)", "bloque": "-", "ms": ms})
        xc = x.cpu().contiguous(memory_format=torch.channels_last)
        wc, bc = w.cpu().contiguous(memory_format=torch.channels_last), b.cpu()
        ms_cpu, _ = cronometrar(lambda: F.relu(F.conv2d(xc, wc, bc, padding=1)), lambda: None,
                                args.repeticiones_cpu)
        filas.append({"capa": nombre, "implementacion": "PyTorch CPU", "bloque": "-", "ms": ms_cpu})
        print(f"{nombre} PyTorch GPU {ms:7.3f} ms | PyTorch CPU {ms_cpu:8.2f} ms", flush=True)

    # Capa densa 1 (8,192 -> 256)
    for lote in (64, 256):
        a = torch.rand(lote, 8192, device=dev)
        w = estado["classifier.1.weight"].to(dev).contiguous()
        b = estado["classifier.1.bias"].to(dev).contiguous()
        ref = F.relu(F.linear(a, w, b))
        an, wn, bn = cuda.as_cuda_array(a), cuda.as_cuda_array(w), cuda.as_cuda_array(b)
        y = cuda.device_array((lote, 256), dtype=np.float32)
        nombre = f"Densa 1 (lote {lote})"
        for TS in (8, 16, 32):
            ms, primera = cronometrar(lambda: K.lanzar_densa(an, wn, bn, y, TS, relu=True), cuda.synchronize,
                                      args.repeticiones)
            err = float((torch.as_tensor(y, device=dev) - ref).abs().max())
            kern = K.kernel_densa(TS)
            regs, sh = K.registros_por_hilo(kern), K.memoria_compartida(kern)
            bl, occ = ocupacion(TS * TS, regs, sh)
            grid = (math.ceil(256 / TS), math.ceil(lote / TS))
            filas.append({"capa": nombre, "implementacion": "Numba, teselas", "bloque": f"{TS}x{TS}", "ms": ms,
                          "primera_llamada_ms": primera, "error_max": err})
            ocup.append({"capa": nombre, "implementacion": "Numba, teselas", "bloque": f"{TS}x{TS}",
                         "grid": str(grid), "bloques_lanzados": int(np.prod(grid)), "registros_por_hilo": regs,
                         "memoria_compartida_B": sh, "bloques_por_SM": bl, "ocupacion": occ})
            print(f"{nombre} teselas {TS:2d}  {ms:7.3f} ms  err={err:.2e}  grid={grid}", flush=True)
        ms, _ = cronometrar(lambda: F.relu(F.linear(a, w, b)), sync_gpu, args.repeticiones)
        filas.append({"capa": nombre, "implementacion": "PyTorch GPU (cuBLAS)", "bloque": "-", "ms": ms})
        ac, wc, bc = a.cpu(), w.cpu(), b.cpu()
        ms_cpu, _ = cronometrar(lambda: F.relu(F.linear(ac, wc, bc)), lambda: None, args.repeticiones_cpu)
        filas.append({"capa": nombre, "implementacion": "PyTorch CPU", "bloque": "-", "ms": ms_cpu})

    pd.DataFrame(filas).to_csv(DIR / "kernels_capas.csv", index=False)
    pd.DataFrame(ocup).to_csv(DIR / "kernels_ocupacion.csv", index=False)

    # ------------------------------------------------------------ 2. F5 completa
    print("\nPreparando el conjunto de prueba (pipeline de CPU con procesos)...", flush=True)
    from .datos_paralelo import PoolDatos, preparar_datos
    from .bench_datos import Crono
    with PoolDatos("procesos", 8) as pool:
        pool.iniciar()
        r = preparar_datos(pool, Crono(), con_eda=False)
    X_u8, y_test = r["X_test"], np.asarray(r["y_test"])
    del r
    X = torch.from_numpy(X_u8).to(dev).float().div_(torch.tensor(255.0, device=dev)).permute(0, 3, 1, 2)
    X_nchw = X.contiguous()                         # los kernels propios usan NCHW contiguo
    Xn = cuda.as_cuda_array(X_nchw)

    modelo = CNN2(dropout=config.DROPOUT)
    modelo.load_state_dict(estado)
    modelo.eval()
    modelo_gpu = CNN2(dropout=config.DROPOUT)
    modelo_gpu.load_state_dict(estado)
    modelo_gpu = modelo_gpu.to(dev, memory_format=torch.channels_last).eval()
    f5 = []

    @torch.no_grad()
    def torch_pred(m, Xt, lote):
        return torch.cat([m(Xt[i:i + lote]).argmax(1) for i in range(0, len(Xt), lote)])

    for lote in (64, 256):
        ms, primera = cronometrar(lambda: torch_pred(modelo_gpu, X, lote), sync_gpu, 10)
        pred_ref = torch_pred(modelo_gpu, X, lote).cpu().numpy()
        f5.append({"implementacion": "PyTorch GPU", "lote": lote, "ms": ms, "primera_llamada_ms": primera,
                   "exactitud": float((pred_ref == y_test).mean()), "coincide_con_pytorch_gpu": 1.0})
        red = K.CNN2Numba(estado, B=16, TS=16, lote_max=lote)
        ms, primera = cronometrar(lambda: red.predecir(Xn), cuda.synchronize, 10)
        logits, pred = red.predecir(Xn)
        with torch.no_grad():
            logits_ref = torch.cat([modelo_gpu(X[i:i + lote]) for i in range(0, len(X), lote)]).cpu().numpy()
        f5.append({"implementacion": "Numba (kernels propios)", "lote": lote, "ms": ms, "primera_llamada_ms": primera,
                   "exactitud": float((pred == y_test).mean()),
                   "coincide_con_pytorch_gpu": float((pred == pred_ref).mean()),
                   "error_max_logits": float(np.abs(logits - logits_ref).max())})
        print(f"F5 lote {lote}: PyTorch GPU {f5[-2]['ms']:.1f} ms acc={f5[-2]['exactitud']:.4f} | "
              f"Numba {ms:.1f} ms acc={f5[-1]['exactitud']:.4f} coincide={f5[-1]['coincide_con_pytorch_gpu']:.4f} "
              f"err={f5[-1]['error_max_logits']:.2e}", flush=True)
    Xc = X.cpu().contiguous(memory_format=torch.channels_last)
    modelo_cpu = modelo.to(memory_format=torch.channels_last)
    ms, _ = cronometrar(lambda: torch_pred(modelo_cpu, Xc, 64), lambda: None, 3)
    pred_cpu = torch_pred(modelo_cpu, Xc, 64).numpy()
    f5.append({"implementacion": "PyTorch CPU", "lote": 64, "ms": ms,
               "exactitud": float((pred_cpu == y_test).mean()),
               "coincide_con_pytorch_gpu": float((pred_cpu == pred_ref).mean())})
    print(f"F5 PyTorch CPU {ms:.1f} ms acc={f5[-1]['exactitud']:.4f}", flush=True)
    pd.DataFrame(f5).to_csv(DIR / "kernels_f5.csv", index=False)
    (DIR / "entorno.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "numba_cuda_detectado": cuda.is_available(), "ecoqos_desactivado": ecoqos,
        "repeticiones": args.repeticiones, "tf32": False, "modelo": args.modelo}, indent=2), encoding="utf-8")
    print("Guardado en", DIR)


if __name__ == "__main__":
    main()
