# Model card — PengwinNet10 (PENGWIN, Proyecto Integrador Corte 2, UAO 2026-2)

> ⚠ **USO EXCLUSIVAMENTE ACADÉMICO.** No es un dispositivo médico, no ha sido validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Descripción
Red multitarea 2.5D para CT pélvico: un backbone compartido y tres cabezas.
- **Entrada:** 5 cortes axiales vecinos (z−4, z−2, z, z+2, z+4 mm) a 256×256, ventana ósea 400/1800 HU.
- **Arquitectura:** Stem 3D (Conv3d 3×3×3 → Conv3d 5×1×1) → **FundidoraPC** (backbone del curso, 4 bloques conv-BN-ReLU-maxpool) → **CBAM** (canal + espacial) antes de la bifurcación.
- **Cabeza 1, clasificación:** presencia de sacro / coxal izq. / coxal der. en el corte (multietiqueta).
- **Cabeza 2, detección:** grilla propia 9×9×8 sin anchors + NMS propio (una caja por región).
- **Cabeza 3, segmentación:** decoder tipo U-Net con skips → 4 clases semánticas + mapa de fragmento principal + mapa de frontera de fractura.
- **Instancias en dos etapas:** primero la región dentro de su caja; luego principal y conminutos por componentes conexas 3D. Cada fragmento se colorea según su macro-hueso.
- **Distancia de separación:** `scipy.ndimage.distance_transform_edt` sobre las máscaras, con el espaciado físico del header (mm).
- **Tamaño:** 2.36 M parámetros. Sin YOLO, Detectron2 ni Mask R-CNN preentrenado; SAM solo como baseline.

## Entrenamiento
- **Datos:** PENGWIN 2024, Task 1 (CT), 100 casos con máscaras de fragmentos validadas por ortopedistas. Taxonomía de 3 regiones (sacro, coxal izquierdo, coxal derecho).
- **Splits:** fijos, por paciente y estratificados por patrón de fractura (semilla 42): 70 / 15 / 15.
- **Cortes:** todos los del volumen (≈ 24 % sin hueso).
- **Augmentation:** flip horizontal con intercambio izquierda/derecha, recorte reescalado, cutout, y contraste/brillo/gamma/ruido.
- **Optimización:** 12 épocas, AdamW + cosine annealing por iteración, precisión mixta (`torch.cuda.amp`), checkpoint de la mejor época de validación (11).
- **Pérdida compuesta:** 9 términos (detección, presencia, CE + Dice semánticos, principal, frontera). Los λ se calibran igualando la norma del gradiente de cada término en el tronco compartido; `img` y `sem_dice` llegan al tope de 20.

## Métricas en test (15 pacientes)
| Tarea | Métrica | Valor | Objetivo |
|---|---|---|---|
| Clasificación | F1 | 0.981 | ≥ 0.85 ✓ |
| Clasificación | AUC | 0.997 | ≥ 0.85 ✓ |
| Detección | IoU promedio | 0.806 | ≥ 0.65 ✓ |
| Detección | mAP@0.50 | 0.896 | ≥ 0.65 ✓ |
| Detección | mAP@[0.50:0.95] | 0.581 | ≥ 0.40 ✓ |
| Segmentación de fragmento | Dice (fragmentwise 3D) | 0.627 | ≥ 0.85 ✗ |
| Segmentación de fragmento | IoU (fragmentwise 3D) | 0.564 | ≥ 0.70 ✗ |
| Referencia | Dice del fragmento principal | 0.921 | — |
| Referencia | Dice 3D por región | 0.965 | — |

- **Por fragmento:** principal Dice 0.921 · conminutos Dice 0.288 (39 conminutos GT) · 23 falsos positivos.
- **Post-proceso calibrado en validación:** volumen mínimo 1.568 mm³ (percentil 5 de los conminutos de train), τ_principal 0.60, τ_frontera 0.50.
- **Distancia de separación:** se mide en el 38.5 % de los conminutos de test. Error medio 0.05 mm en los que están en contacto y 1.72 mm en los separados (n = 4).
- **SAM ViT-B zero-shot** con la misma caja como prompt: Dice 0.857 vs 0.940 del modelo propio. SAM tarda 495 ms por corte frente a 10 ms del modelo propio.
- **HD95 / ASSD por fragmento** (extensión opcional del §5; medianas en mm, fragmentos emparejados, resolución de trabajo): principal 3.61 / 0.52; conminutos 33.74 / 4.91 (21 de 39 emparejados); todos 11.70 / 1.42. Detalle en `results/hd95_assd.csv`, generado con `scripts/hd95_assd.py`.

### Ablación (validación; reducida: 4 épocas sobre 24 casos de train, misma semilla y λ)
| Configuración | F1 | AUC | IoU | mAP@0.5 | mAP@[0.5:0.95] | Dice semántico |
|---|---|---|---|---|---|---|
| Completo (CBAM, 2.5D N = 5) | 0.954 | 0.988 | 0.692 | 0.766 | 0.384 | 0.921 |
| Sin CBAM | 0.947 | 0.984 | 0.677 | 0.732 | 0.366 | 0.911 |

### Latencia
| Dispositivo | Por corte (3 cabezas + NMS) | Volumen de 401 cortes (red + instancias 3D) |
|---|---|---|
| GPU NVIDIA T4 | 10.2 ms | 9.6 s (8.3 + 1.3) |
| CPU (Colab, entorno sin GPU) | 175.7 ms | 160.7 s (151.6 + 4.3) |


## Otros checkpoints del repositorio
Los tres usan el mismo backbone (FundidoraPC + CBAM), splits por paciente, la semilla 42 y precisión mixta. Su alcance, sus limitaciones y la advertencia de uso no clínico son los mismos de este card.

| Checkpoint | Qué es | Entrenamiento | Métricas (validación) | Uso |
|---|---|---|---|---|
| `s9_det_best.pth` | Detector de la semana 9: entrada 2D de 1 corte, FundidoraPC + CBAM, grilla 9×9×8 sin anchors, NMS propio y cabeza de presencia por región. Sin cabeza de segmentación. | 15 épocas, AdamW + cosine. Se guarda la mejor época por IoU (9). Antes pasó la prueba de overfit sobre un batch de 16 cortes (IoU 0.93, mAP@0.50 1.0). | IoU 0.742 · mAP@0.50 0.827 · mAP@[0.50:0.95] 0.466 | Avance S9. Base de diseño de la cabeza de detección de PengwinNet10; no se usa en el dashboard. |
| `abl_*.pth` (completo) | PengwinNet10 con CBAM y entrada 2.5D N = 5. | Ablación reducida: 4 épocas sobre 24 casos de train, mismos λ. | Tabla de ablación: F1 0.954 · IoU 0.692 · mAP@0.50 0.766 · Dice semántico 0.921 | Solo para la ablación. |
| `abl_*.pth` (sin CBAM) | Igual al anterior, pero sin el bloque CBAM. | Igual al anterior. | F1 0.947 · IoU 0.677 · mAP@0.50 0.732 · Dice semántico 0.911 | Solo para la ablación. |

## Alcance
Demostrar, con fines académicos, un pipeline propio sobre CT pélvico del dataset PENGWIN: detección → NMS → segmentación de instancia → clasificación por pertenencia → medición en mm.

## Limitaciones conocidas
1. **Fragmentos en contacto en la superficie de fractura.** El 90.9 % de los conminutos GT toca a su fragmento principal (distancia ≈ 1 vóxel). Cuando la frontera predicha no los separa, **se fusionan con el principal** por error de segmentación. En test, 20 de 39 conminutos quedaron fusionados; esos no tienen distancia predicha y desaparecen del conteo. El caso más grave es un conminuto del sacro de 42.9 cm³ (caso 019).
2. **Distancias cercanas a cero.** Para fragmentos en contacto, la EDT mide de centro a centro de vóxel: la separación mínima reportable es ≈ 1 espaciado (0.6–1.3 mm), no 0.
3. **Resolución de 256 px.** El vóxel en plano pasa de ~0.8 mm a 1.2–1.95 mm. En los conminutos separados, la distancia predicha subestima la real en 1–2 píxeles.
4. **Fragmentos pequeños.** Los menores de ~1.6 cm³ se descartan por el umbral de tamaño calibrado y se reasignan al fragmento vecino.
5. **Sacro en sus extremos craneal y caudal.** Es la región con menor IoU de detección (0.663). Las 5 peores detecciones de test son cortes extremos del sacro, 3 de ellos sin caja predicha.
6. **Arquitectura 2.5D, no 3D completa.** La consistencia entre cortes se impone con contexto de ±4 mm, suavizado en z y componentes conexas 3D.
7. **Datos.** Son 100 casos de un solo challenge, con posibles sesgos de escáner y protocolo, y orientaciones de origen variables entre volúmenes. No se evaluó en otras instituciones.

## Uso no permitido
Diagnóstico, planeación quirúrgica o cualquier decisión clínica real.
