# data/ (no se versiona en git)

Estructura que espera el dashboard:

```
data/
├── dashboard/          # OBLIGATORIO · copiar de Drive: PENGWIN_outputs/dashboard/
│   ├── config.json     #   splits, post-proceso calibrado, metas, latencias
│   ├── 001.npz         #   volumen 256 px + instancias predichas + GT + cajas (15 casos de test)
│   └── 001.json        #   tabla de fragmentos con su distancia en mm
└── images/             # OPCIONAL · CT crudos de los 15 casos de test (Visualizador 1 en HU reales e inferencia en vivo)
    └── 001.mha
```

La forma rápida de llenarla:

```bash
python scripts/preparar_datos.py --outputs "<ruta a PENGWIN_outputs>" --zips PENGWIN_CT_train_images_part1.zip PENGWIN_CT_train_images_part2.zip
```

En el repositorio de GitHub, `data/dashboard/` se publica como asset de un Release (`dashboard_test.zip`), porque son decenas de MB de datos derivados del dataset.
