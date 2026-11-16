# Métricas de la versión secuencial (Paso 2)

Generado por `python -m src.secuencial --corridas 5 --calentamiento 1` desde la raíz del proyecto.

## Protocolo

- **Calentamiento:** 1 corrida solo de las fases de datos (F1–F3) para cargar el dataset en la caché de disco del sistema operativo. **No se promedia.**
- **Medición:** 5 corridas completas e idénticas, todas con semilla 42, los mismos datos y los mismos hiperparámetros (CNN 2, configuración 2: lote 64, lr 0.001, dropout 0.5, 15 épocas).
- **Preparación de datos (F1–F3 y EDA):** bucles de Python en un único proceso, igual que en el Lab 3.
- **Entrenamiento y evaluación (F4–F5):** PyTorch en **CPU** con los hilos por defecto, sin GPU.
- **Monitor de recursos:** cada 1 s registra el uso de cada núcleo lógico, la RAM y la GPU (vía NVML), etiquetado con la fase en curso.

## Archivos generales

| Archivo | Contenido |
|---|---|
| `entorno.json` | Hardware y versiones: CPU, núcleos, RAM, GPU, driver, librerías, plan de energía, si estaba conectado a corriente |
| `tiempos.csv` | Una fila por corrida: segundos de cada fase, total, exactitud y pérdida en prueba |
| `resumen_tiempos.csv` | Por fase: n, media, desviación estándar, mínimo, máximo y % del tiempo total (solo corridas de medición) |
| `resumen_exactitud.json` | Exactitud media en prueba entre corridas |
| `tiempos_por_fase.png` | Gráfica de media ± desviación por fase |
| `log_ejecucion.txt` | Salida completa de la consola |

## Archivos por corrida (`corrida_XX/`, `calentamiento_XX/`)

| Archivo | Contenido |
|---|---|
| `metricas.json` | Tiempos por fase, conteos de datos (imágenes, duplicados, partición), métricas de prueba y uso promedio de CPU/GPU por fase |
| `recursos.csv` | Serie temporal (1 s): fase, CPU total, CPU por núcleo (`cpuN_pct`), CPU y RAM del proceso, hilos, GPU (uso, memoria, temperatura, potencia) |
| `uso_recursos.png` | Mapa de calor del uso por núcleo + uso de GPU, con las fases marcadas |
| `uso_recursos_fases_datos.png` | Igual, ampliado a las fases de preparación de datos (F1–F3 y EDA) |
| `historia_entrenamiento.csv` | Por época: loss, accuracy, val_loss, val_accuracy y tiempos de entrenamiento/validación |
| `curvas_entrenamiento.png` | Curvas de pérdida y exactitud |
| `reporte_clasificacion.txt` | Precisión, recall y F1 por clase en prueba |
| `matriz_confusion.csv` | Matriz de confusión 29×29 en prueba |
| `eda_pares_distancias.csv` | Distancias entre imágenes promedio de cada par de clases (EDA) |
| `eda_brillo_por_clase.csv` | Brillo medio y desviación por clase (EDA) |

## Fases medidas

| Fase | Operaciones (Tabla 2 del informe) |
|---|---|
| F1_ingesta | 1. Recorrido del dataset (~87,000 JPG) |
| EDA_promedios | 3–4. Imagen promedio por clase (29 × 200) + distancias entre clases |
| F2_limpieza | 5–7. Submuestreo + perceptual hash (17,400) + verificación de duplicados |
| EDA_brillo | 8. Brillo de las 17,391 imágenes limpias |
| F3_preprocesamiento | 9–11. Partición + carga/resize/normalización (17,391) + etiquetas + tensores |
| F4_entrenamiento | 12–17. Construcción del modelo + 15 épocas (incluye validación por época) |
| F5_evaluacion | 18. Predicción en prueba + matriz de confusión + reporte |

> Nota: en una laptop, NVML no responde cuando la GPU está en reposo. Esas muestras quedan vacías en `recursos.csv` y la gráfica las muestra como 0 %, ya que la versión secuencial no usa la GPU.
