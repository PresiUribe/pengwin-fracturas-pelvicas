# weights/

| Archivo | Modelo | Origen |
|---|---|---|
| `s10_best.pth` | PengwinNet10 final (stem 3D + FundidoraPC + CBAM + 3 cabezas), mejor época de validación (11) | `MyDrive/PENGWIN_outputs/s10_best.pth` |
| `s9_det_best.pth` | Detector de la semana 9 (FundidoraPC + CBAM + grilla 9×9) | `MyDrive/PENGWIN_outputs/s9_det_best.pth` |
| `abl_*.pth` | Modelos de la ablación (4 épocas, 24 casos) | `MyDrive/PENGWIN_outputs/abl_*.pth` |

Cada checkpoint es un `dict` con `model` (state_dict), `lam` (λ calibrados), `n_ctx`, `use_cbam`, `epoch` y `val` (métricas de validación). Se cargan con `pengwin_core.load_model(ruta, device)`. Pesan menos de 15 MB cada uno, así que se versionan sin Git LFS.
