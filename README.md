# PENGWIN · Detección, segmentación y medición de fragmentos pélvicos en CT

Proyecto Integrador Corte 2 · Universidad Autónoma de Occidente · 2026-2
Samuel Uribe · Emilio Marquez · Juan Pablo Lopez

> 🎥 **Video de respaldo de la demo** (dashboard en Docker + túnel de Cloudflare): https://www.youtube.com/watch?v=2TRgPktQ-YU

> ⚠ **Uso exclusivamente académico.** No es un dispositivo médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Sistema propio (sin YOLO, Detectron2 ni Mask R-CNN preentrenado) que, a partir de una CT pélvica del dataset [PENGWIN](https://pengwin.grand-challenge.org):

1. **Detecta** sacro, coxal izquierdo y coxal derecho con una grilla 9×9 propia y NMS propio.
2. **Segmenta** cada fragmento y lo colorea según el hueso al que pertenece (dos etapas: región → instancia).
3. **Mide en mm** la distancia mínima de cada conminuto al fragmento principal (`distance_transform_edt` con el espaciado del header).
4. Lo muestra en un **dashboard** con tres visualizadores, publicado con un túnel de Cloudflare.

## Resultados en test (15 pacientes)

| Tarea | Métrica | Test | Objetivo |
|---|---|---|---|
| Clasificación | F1 / AUC | 0.981 / 0.997 | ≥ 0.85 |
| Detección | IoU / mAP@0.5 / mAP@[0.5:0.95] | 0.806 / 0.896 / 0.581 | ≥ 0.65 / 0.65 / 0.40 |
| Segmentación de fragmento | Dice / IoU (fragmentwise 3D) | 0.627 / 0.564 | ≥ 0.85 / 0.70 |
| Referencia | Dice del principal / Dice 3D por región | 0.921 / 0.965 | — |

Modelo propio vs SAM ViT-B zero-shot con la misma caja como prompt: Dice 0.940 vs 0.857. Detalle, ablación, limitaciones y advertencia de uso en [`MODEL_CARD.md`](MODEL_CARD.md).

## Estructura
```
├── notebooks/
│   ├── PENGWIN_S8_S9_S10.ipynb        # datos, EDA, Visualizador 1, entrenamiento, métricas, SAM, ablación (con salidas)
│   ├── PENGWIN_S11_Dashboard.ipynb    # latencia, visualizadores 2 y 3, peores casos, model card, dashboard + túnel en Colab
│   └── ejecutados/                    # copia del notebook S11 con sus salidas
├── pengwin_core.py                    # arquitectura, NMS, post-proceso, distancias y visualizadores
├── app.py                             # dashboard Gradio
├── Dockerfile · docker-compose.yml
├── scripts/                           # preparar_datos.py · url_tunel.sh · url_tunel.ps1
├── tests/                             # pruebas de humo (corren en CI en cada PR)
├── weights/                           # s10_best.pth, s9_det_best.pth, abl_*.pth
├── data/                              # (no versionada) dashboard/ y images/ — ver data/README.md
├── results/                           # figuras y tablas del informe (copiadas de Drive: PENGWIN_outputs/informe y eda)
├── MODEL_CARD.md · IA_USAGE.md · GUIA_GITHUB.md
└── requirements.txt (Colab) · requirements-docker.txt (dashboard)
```

## Dashboard con Docker (recomendado para la demo)
Requisitos: Docker Desktop (Windows/macOS) o Docker Engine (Linux).

```bash
# 1. datos: predicciones de test + (opcional) CT crudos de test
python scripts/preparar_datos.py --outputs "<ruta a PENGWIN_outputs descargada de Drive>" \
       --zips PENGWIN_CT_train_images_part1.zip PENGWIN_CT_train_images_part2.zip

# 2. levantar app + túnel de Cloudflare
docker compose up -d --build

# 3. URL pública (computador y celular)
./scripts/url_tunel.sh            # Windows: powershell -ExecutionPolicy Bypass -File scripts\url_tunel.ps1

# apagar
docker compose down
```

- Local: http://localhost:7860 · Público: `https://….trycloudflare.com`. La URL cambia en cada arranque del túnel.
- **CPU por defecto.** Los casos precalculados cargan al instante; la inferencia en vivo de un volumen tarda ~2–3 min en CPU.
- La primera construcción descarga ~1.5 GB (PyTorch CPU + dependencias). Las siguientes usan la caché.

## Reproducir el entrenamiento (Colab)
1. Subir a Drive los zips de Zenodo (`PENGWIN_CT_train_images_part1.zip`, `..._part2.zip`, `PENGWIN_CT_train_labels.zip`) en `MyDrive/Datos_pengwin/`. La carpeta puede ser compartida de solo lectura.
2. Abrir `notebooks/PENGWIN_S8_S9_S10.ipynb` en Colab con GPU → *Ejecutar todas*. Semilla fija `SEED = 42`. Los splits por paciente quedan en `PENGWIN_outputs/eda/splits.json`.
3. Todas las salidas van a `MyDrive/PENGWIN_outputs/`: checkpoints, CSV, `dashboard/`. Entrenamiento y ablación se reanudan si Colab se reinicia.
4. `notebooks/PENGWIN_S11_Dashboard.ipynb` (sirve en CPU): latencia, galerías, `MODEL_CARD.md` y dashboard con túnel desde Colab (alternativa a Docker).

Tiempos aproximados en una T4: EDA ~15 min · semana 9 ~25 min · semana 10 ~1.5 h · SAM ~5 min · ablación ~16 min por configuración.

## Flujo de trabajo en GitHub
Ver [`GUIA_GITHUB.md`](GUIA_GITHUB.md): `main` protegida, un PR por tarea, revisión de otro integrante y CI en verde antes del merge.

## Uso de IA generativa
Ver [`IA_USAGE.md`](IA_USAGE.md).
