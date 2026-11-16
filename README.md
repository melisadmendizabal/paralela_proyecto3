# Proyecto 3 — Mejorando algoritmos con CUDA

CC3069 Computación Paralela y Distribuida, Universidad del Valle de Guatemala.
Belén Monterroso 231497 · Melisa Mendizabal 23778 · Renato Rojas 23813

Aceleración del clasificador del alfabeto de señas ASL del Laboratorio 3 de Data Science
(dataset *ASL Alphabet* de Kaggle, 29 clases, CNN de 2.2 M de parámetros):

- **Pipeline de CPU** (preparación de imágenes): repartido entre varios núcleos con hilos, procesos,
  Numba o Python sin GIL.
- **Pipeline de CUDA** (entrenamiento y evaluación): PyTorch en la GPU y kernels propios con Numba CUDA.

## Dónde está cada versión

| Versión | Código |
|---|---|
| Original del laboratorio (TensorFlow, notebooks) | `original/` |
| **Secuencial** (línea base, mismo pipeline portado a PyTorch en CPU) | `src/datos.py`, `src/eda.py`, `src/entrenamiento.py` (`entrenar`, `evaluar`), `src/secuencial.py` |
| **Paralela — CPU** (preparación de datos) | `src/datos_paralelo.py`, `src/numba_cpu.py` |
| **Paralela — CUDA** (modelo) | `src/entrenamiento.py` (`entrenar_gpu`, `evaluar_gpu`, `a_gpu_normalizado`), `src/kernels_numba.py` |
| **Paralela completa** | `src/paralelo.py` |
| Compartido por ambas versiones | `src/config.py` (hiperparámetros), `src/modelo.py` (CNN 2), `src/medicion.py` (cronómetro y monitor) |

## Requisitos

- Windows 11, Python 3.13, GPU NVIDIA con driver reciente (probado con una RTX 4050 Laptop).
- `pip install -r requirements.txt`
- Dataset de Kaggle descomprimido en `archive/asl_alphabet_train/asl_alphabet_train/` (29 carpetas).

## Cómo ejecutar (desde la raíz del proyecto)

```bash
# Versión secuencial (línea base): 1 calentamiento + 5 corridas, ~45 min
python -m src.secuencial

# Versión paralela: procesos ×8 en la CPU + GPU (t1 = FP32, t3 = precisión mixta, t2 = port directo)
python -m src.paralelo --gpu t1
python -m src.paralelo --gpu t3

# Experimentos
python -m src.perfil            # Paso 3: perfilado de la versión secuencial
python -m src.bench_datos       # variantes del pipeline de CPU × número de trabajadores
python -m src.bench_kernels     # kernels propios de Numba CUDA
python -m src.comparar datos    # aceleración del pipeline de CPU
python -m src.comparar total    # aceleración total por fase

# Variante E (Python sin GIL), en un entorno aparte
pymanager install 3.13t
python3.13t -m venv .venv313t
.venv313t\Scripts\python -m pip install --only-binary=:all: numpy pandas scipy PyWavelets scikit-learn psutil pillow imagehash
.venv313t\Scripts\python -m src.bench_datos --variantes secuencial hilos --etiqueta 313t
```

Prueba rápida de la versión paralela: `python -m src.paralelo --corridas 1 --calentamiento 0 --epochs 1 --salida benchmarks/prueba`.

## Resultados

Todo lo medido queda en `benchmarks/` (cada carpeta tiene los CSV, JSON, gráficas y logs de su experimento):
`secuencial/`, `perfil/`, `datos_paralelo/`, `paralelo/`, `kernels/`, `comparacion/`.

## Notas del entorno

- **EcoQoS (Windows 11):** los programas de medición desactivan la limitación de energía de Windows
  (`src/energia.py`); sin esto, un proceso sin ventana activa puede correr hasta 11 veces más lento.
- **Numba y PyTorch en el mismo proceso:** `src/kernels_numba.py` debe importarse **antes** que `torch`
  (conflicto de versiones de nvJitLink, ver `src/numba_compat.py`).
