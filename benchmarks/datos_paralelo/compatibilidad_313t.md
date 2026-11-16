# Experimento E: compatibilidad con Python sin GIL (3.13t)

Intérprete: **Python 3.13.17 experimental free-threading build** (instalado con `pymanager install 3.13t`),
en un entorno virtual aparte (`.venv313t/`, ignorado por git). El entorno principal usa Python 3.13.16 con GIL.

## Instalación de librerías (Windows, 64 bits)

| Librería | Versión en el entorno principal | Versión en 3.13t | Resultado |
|---|---|---|---|
| numpy | 2.5.4 | 2.4.6 | `pip install numpy` intentó compilar desde el código fuente y falló (no hay compilador de C). Con `--only-binary=:all:` se instaló la última versión con paquete precompilado para 3.13t. |
| pandas | 2.3.3 | **3.0.3** | Igual que numpy. **No existe ninguna versión 2.x precompilada para 3.13t**: solo 3.0.0 en adelante. Pandas 3 cambia algunos comportamientos (tipo de texto por defecto), por eso la verificación por huellas puede diferir aunque los datos sean iguales. |
| scipy | 1.18.1 | 1.17.1 | Falló al compilar desde el código fuente; se instaló la última versión precompilada. |
| PyWavelets (dependencia de imagehash) | 1.10.0 | 1.9.0 | Falló al compilar desde el código fuente; se instaló la última versión precompilada. |
| scikit-learn | — | 1.8.0 | Instalado con paquete precompilado. |
| Pillow | 12.3.0 | 12.2.0 | Instalado con paquete precompilado. |
| imagehash | 4.3.2 | 4.3.2 | Python puro: sin problemas. |
| psutil | — | 7.2.2 | Instalado con paquete precompilado. |
| numba | 0.68.1 | — | **No compatible** con Python sin GIL: la variante D (hilos + Numba) no se puede ejecutar en este experimento. |
| torch | 2.6.0+cu124 | — | No se instaló: el experimento E solo mide el pipeline de CPU. |

Comando que funcionó: `.venv313t/Scripts/python.exe -m pip install --only-binary=:all: numpy pandas scipy PyWavelets scikit-learn psutil pillow imagehash`

## ¿Alguna librería reactiva el GIL?

Si se importa una extensión en C que no declara compatibilidad con Python sin GIL, el intérprete vuelve a
activar el GIL. Se importó cada módulo del pipeline y se consultó `sys._is_gil_enabled()` después de cada uno:

| Módulo importado | GIL activo después |
|---|---|
| (al iniciar) | False |
| numpy, PIL.Image, scipy.fftpack, scipy.spatial.distance, pywt, imagehash, pandas, sklearn, psutil | False |
| src.datos, src.eda, src.datos_paralelo | False |

**Ninguna librería reactivó el GIL.** Además, `bench_datos.py` registra el estado del GIL en cada corrida
(columna `gil_activo` de `bench_datos_313t.csv`).
