"""Utilidades de medición: cronómetro por fase, monitor de recursos y entorno.

- Cronometro: mide cada fase con time.perf_counter() (sincroniza CUDA si aplica).
- MonitorRecursos: hilo que cada `intervalo` segundos registra el uso de CPU
  por núcleo, la memoria y el estado de la GPU (NVML), etiquetando cada
  muestra con la fase que se está ejecutando. Es la evidencia de qué hardware
  se usó en cada fase.
- info_entorno: versiones y hardware, para documentar las condiciones.
"""
import csv
import json
import os
import platform
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # sin GPU NVIDIA o sin nvidia-ml-py
    pynvml, _GPU = None, None


# ----------------------------------------------------------------------------
# Monitor de recursos
# ----------------------------------------------------------------------------
class MonitorRecursos(threading.Thread):
    def __init__(self, intervalo=1.0):
        super().__init__(daemon=True)
        self.intervalo = intervalo
        self.fase = "inicio"
        self.muestras = []
        self._parar = threading.Event()
        self._proc = psutil.Process(os.getpid())
        self.n_nucleos = psutil.cpu_count(logical=True)

    def _muestra_gpu(self):
        """Lee el estado de la GPU. En laptops NVML falla a veces cuando la GPU
        está en reposo/ahorro de energía; ese valor se registra como vacío."""
        fila = {"gpu_util_pct": None, "gpu_mem_util_pct": None, "gpu_mem_usada_mb": None,
                "gpu_temp_c": None, "gpu_potencia_w": None}
        if _GPU is None:
            return fila

        def leer(fn):
            try:
                return fn()
            except Exception:
                return None
        util = leer(lambda: pynvml.nvmlDeviceGetUtilizationRates(_GPU))
        mem = leer(lambda: pynvml.nvmlDeviceGetMemoryInfo(_GPU))
        if util is not None:
            fila["gpu_util_pct"], fila["gpu_mem_util_pct"] = util.gpu, util.memory
        if mem is not None:
            fila["gpu_mem_usada_mb"] = round(mem.used / 2**20, 1)
        fila["gpu_temp_c"] = leer(lambda: pynvml.nvmlDeviceGetTemperature(_GPU, pynvml.NVML_TEMPERATURE_GPU))
        potencia = leer(lambda: pynvml.nvmlDeviceGetPowerUsage(_GPU))
        fila["gpu_potencia_w"] = potencia / 1000.0 if potencia is not None else None
        return fila

    def _muestrear(self, t0):
        nucleos = psutil.cpu_percent(interval=None, percpu=True)
        fila = {
            "t_s": round(time.perf_counter() - t0, 2),
            "fase": self.fase,
            "cpu_total_pct": round(sum(nucleos) / len(nucleos), 1),
            "cpu_proceso_pct": self._proc.cpu_percent(interval=None),
            "hilos_proceso": self._proc.num_threads(),
            "ram_proceso_mb": round(self._proc.memory_info().rss / 2**20, 1),
            "ram_sistema_pct": psutil.virtual_memory().percent,
        }
        # Procesos hijo (la versión paralela usa varios procesos)
        try:
            hijos = self._proc.children(recursive=True)
            fila["procesos_hijo"] = len(hijos)
        except psutil.Error:
            fila["procesos_hijo"] = None
        fila.update(self._muestra_gpu())
        for i, v in enumerate(nucleos):
            fila[f"cpu{i}_pct"] = v
        self.muestras.append(fila)

    def run(self):
        t0 = time.perf_counter()
        psutil.cpu_percent(interval=None, percpu=True)  # primera llamada = referencia
        self._proc.cpu_percent(interval=None)
        while not self._parar.wait(self.intervalo):
            try:
                self._muestrear(t0)
            except Exception as e:  # el monitor nunca debe detener la medición
                print(f"[monitor] muestra omitida: {e}", flush=True)

    def detener(self):
        self._parar.set()
        self.join(timeout=5)

    def guardar_csv(self, ruta):
        if not self.muestras:
            return
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.muestras[0].keys()))
            w.writeheader()
            w.writerows(self.muestras)

    def resumen_por_fase(self):
        """Promedios de CPU/GPU por fase (para la tabla de evidencia)."""
        fases = {}
        for m in self.muestras:
            fases.setdefault(m["fase"], []).append(m)
        resumen = {}
        for fase, filas in fases.items():
            def prom(k):
                vals = [f[k] for f in filas if f[k] is not None]
                return round(sum(vals) / len(vals), 1) if vals else None

            def maximo(k):
                vals = [f[k] for f in filas if f[k] is not None]
                return max(vals) if vals else None
            nucleos_activos = [sum(1 for i in range(self.n_nucleos) if f[f"cpu{i}_pct"] >= 50)
                               for f in filas]
            resumen[fase] = {
                "muestras": len(filas),
                "cpu_total_pct_prom": prom("cpu_total_pct"),
                "cpu_proceso_pct_prom": prom("cpu_proceso_pct"),
                "nucleos_sobre_50pct_prom": round(sum(nucleos_activos) / len(nucleos_activos), 1),
                "ram_proceso_mb_max": maximo("ram_proceso_mb"),
                "gpu_util_pct_prom": prom("gpu_util_pct"),
                "gpu_util_pct_max": maximo("gpu_util_pct"),
                "gpu_mem_usada_mb_max": maximo("gpu_mem_usada_mb"),
            }
        return resumen


# ----------------------------------------------------------------------------
# Cronómetro por fase
# ----------------------------------------------------------------------------
class Cronometro:
    def __init__(self, monitor=None, sincronizar_cuda=False):
        self.tiempos = {}
        self.monitor = monitor
        self.sincronizar_cuda = sincronizar_cuda

    def _sync(self):
        if self.sincronizar_cuda:
            import torch
            torch.cuda.synchronize()

    @contextmanager
    def fase(self, nombre):
        if self.monitor is not None:
            self.monitor.fase = nombre
        self._sync()
        t0 = time.perf_counter()
        print(f"[{datetime.now():%H:%M:%S}] -> {nombre}", flush=True)
        try:
            yield
        finally:
            self._sync()
            self.tiempos[nombre] = time.perf_counter() - t0
            print(f"[{datetime.now():%H:%M:%S}]    {nombre}: {self.tiempos[nombre]:.2f} s", flush=True)
            if self.monitor is not None:
                self.monitor.fase = "entre_fases"


# ----------------------------------------------------------------------------
# Entorno
# ----------------------------------------------------------------------------
def _nombre_cpu():
    if platform.system() == "Windows":
        try:
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                   r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def info_entorno():
    import numpy, pandas, PIL, sklearn, imagehash, torch
    info = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "sistema": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": _nombre_cpu(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "torch_hilos_cpu": torch.get_num_threads(),
        "torch_cuda_compilado": torch.version.cuda,
        "cuda_disponible": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
        "imagehash": imagehash.__version__,
    }
    if _GPU is not None:
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU)
        info.update({
            "gpu": pynvml.nvmlDeviceGetName(_GPU),
            "gpu_memoria_gb": round(mem.total / 2**30, 2),
            "driver_nvidia": pynvml.nvmlSystemGetDriverVersion(),
        })
    try:
        info["plan_energia"] = os.popen("powercfg /getactivescheme").read().strip()
    except Exception:
        pass
    try:
        bateria = psutil.sensors_battery()
        info["conectado_a_corriente"] = bateria.power_plugged if bateria else None
    except Exception:
        pass
    return info


def guardar_json(obj, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
