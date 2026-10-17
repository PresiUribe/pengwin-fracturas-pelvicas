# Guía para montar el repositorio en GitHub

La consigna (§7) pide un **historial de commits y Pull Requests por integrante**, con **`main` protegida** y **merge solo después de revisión por pares**. Por eso cada parte del proyecto entra por un PR hecho **desde la cuenta de quien lo trabajó** y lo aprueba otro integrante. No se deben subir commits a nombre de otra persona.

## 1. Crear el repositorio (lo hace una persona, el «dueño»)
1. GitHub → **New repository** → nombre `pengwin-fracturas-pelvicas` → Private (o Public) → **sin** README (ya viene en el zip).
2. **Settings → Collaborators** → invitar a los otros dos integrantes con rol **Write**.
3. Primer commit, solo con el esqueleto:

```bash
cd pengwin-fracturas-pelvicas
git init -b main
git add README.md .gitignore .gitattributes data/README.md data/dashboard/.gitkeep data/images/.gitkeep
git commit -m "Esqueleto del repositorio"
git remote add origin https://github.com/<dueño>/pengwin-fracturas-pelvicas.git
git push -u origin main
```

## 2. Proteger `main`
**Settings → Branches → Add branch ruleset** (o *Add classic branch protection rule*), sobre `main`:
- ✅ Require a pull request before merging → **Required approvals: 1** → ✅ Dismiss stale approvals when new commits are pushed
- ✅ Require status checks to pass → agregar **`pruebas (pytest, CPU)`** y **`build de la imagen`** (aparecen después del primer PR que corra CI)
- ✅ Block force pushes · ✅ Restrict deletions
- Aplicar también a administradores (para que el dueño no pueda saltarse la revisión)

## 3. Reparto sugerido (al menos un PR por integrante)
Ajusten los nombres según quién hizo realmente cada parte:

| Integrante | Rama | Archivos del zip que sube | Revisa |
|---|---|---|---|
| Samuel Uribe | `feat/datos-eda-entrenamiento` | `notebooks/PENGWIN_S8_S9_S10.ipynb`, `results/` (figuras y CSV del EDA) | Emilio |
| Emilio Marquez | `feat/modelo-core` | `pengwin_core.py`, `tests/`, `requirements.txt`, `weights/` | Juan Pablo |
| Juan Pablo Lopez | `feat/dashboard-docker` | `app.py`, `Dockerfile`, `docker-compose*.yml`, `.dockerignore`, `requirements-docker.txt`, `scripts/`, `.github/`, `notebooks/PENGWIN_S11_Dashboard.ipynb`, `notebooks/ejecutados/` | Samuel |
| Samuel Uribe | `docs/model-card-ia-usage` | `MODEL_CARD.md`, `IA_USAGE.md`, `GUIA_GITHUB.md`, `weights/README.md`, informe | Juan Pablo |

Orden recomendado de merge: **Emilio → Juan Pablo → Samuel (datos) → Samuel (docs)**. Cada PR lo revisa alguien distinto de su autor. CI necesita `pengwin_core.py` y `tests/` para correr. Antes de eso, el PR de Juan Pablo mostrará CI en rojo.

## 4. Flujo de cada integrante
```bash
git clone https://github.com/<dueño>/pengwin-fracturas-pelvicas.git && cd pengwin-fracturas-pelvicas
git checkout -b feat/modelo-core               # su rama
# copiar SUS archivos del zip a la carpeta del repo
git add pengwin_core.py tests/ requirements.txt weights/
git commit -m "Modelo: FundidoraPC + CBAM + 3 cabezas, NMS propio y post-proceso 3D"
git push -u origin feat/modelo-core
```
Luego, en GitHub: **Compare & pull request** → llenar la plantilla → asignar revisor. El revisor deja comentarios o aprueba (**Review changes → Approve**). Con CI en verde y la aprobación → **Squash and merge**.

Vale la pena hacer más de un commit por PR, uno por cambio lógico. Así el historial muestra el trabajo real, como pide la consigna.

## 5. Datos y pesos
- **Pesos** (`weights/*.pth`, < 15 MB cada uno): van en git dentro del PR del modelo. Copiarlos de `MyDrive/PENGWIN_outputs/`.
- **Predicciones de test para el dashboard** (`data/dashboard/`, decenas de MB): **no** van en git. Publíquenlas en **Releases → Draft a new release** → tag `v1.0` → adjuntar `dashboard_test.zip` (el contenido de `PENGWIN_outputs/dashboard`). El README de `data/` explica cómo usarlo.
- **CT crudos**: no se redistribuyen (licencia del dataset). Cada quien los descarga de Zenodo.

## 6. Checklist final de la consigna (§7)
- [ ] Historial de commits y PR por integrante, `main` protegida, merges con revisión
- [ ] Pesos `.pth` + pipeline reproducible (`requirements*.txt`, `SEED = 42`, README con comandos)
- [ ] Dashboard en vivo con túnel de Cloudflare (Docker o Colab), probado desde PC y celular
- [ ] Latencia GPU y CPU (`MODEL_CARD.md`, sección Latencia)
- [ ] `MODEL_CARD.md` con fusión de fragmentos en contacto y advertencia de uso no clínico
- [ ] `IA_USAGE.md` completo, con los prompts de **cada** integrante
- [ ] Informe mini-paper y pitch con video de respaldo