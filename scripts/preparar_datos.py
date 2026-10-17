"""Prepara la carpeta data/ para el dashboard en Docker (solo librería estándar de Python).

Uso (desde la raíz del repo):
  python scripts/preparar_datos.py --outputs "<carpeta PENGWIN_outputs descargada de Drive>" \
                                   [--zips <PENGWIN_CT_train_images_part1.zip> <..._part2.zip>]

Qué hace:
  1. Copia PENGWIN_outputs/dashboard/  → data/dashboard/   (predicciones de test, obligatorio)
  2. Copia PENGWIN_outputs/s10_best.pth → weights/           (si aún no está en el repo)
  3. Con --zips: extrae SOLO los CT de los casos de test → data/images/
     (opcional: los usa el Visualizador 1 y la inferencia en vivo; sin ellos el MIP usa el volumen a 256 px)
"""
import argparse, json, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", required=True, help="carpeta PENGWIN_outputs (o el .zip que descarga Drive)")
ap.add_argument("--zips", nargs="*", default=[], help="zips de imágenes de Zenodo (part1 y part2)")
a = ap.parse_args()

src = Path(a.outputs)
if src.suffix == ".zip":                                   # Drive descarga las carpetas como zip
    tmp = ROOT / "data" / "_outputs_tmp"
    with zipfile.ZipFile(src) as z: z.extractall(tmp)
    hits = list(tmp.rglob("dashboard/config.json"))
    if not hits: sys.exit("El zip no contiene dashboard/config.json")
    src = hits[0].parents[1]

dash = src / "dashboard"
if not (dash / "config.json").exists():
    sys.exit(f"No encontré {dash}/config.json. ¿Es la carpeta PENGWIN_outputs correcta?")

dst = ROOT / "data" / "dashboard"
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(dash, dst)
print(f"✓ dashboard: {len(list(dst.glob('*.npz')))} casos → {dst}")

w = ROOT / "weights" / "s10_best.pth"
if not w.exists():
    if (src / "s10_best.pth").exists():
        w.parent.mkdir(exist_ok=True); shutil.copy2(src / "s10_best.pth", w); print(f"✓ pesos → {w}")
    else:
        print("⚠ No encontré s10_best.pth: la inferencia en vivo no estará disponible (los casos precalculados sí).")
for extra in ("s9_det_best.pth",):
    if (src / extra).exists() and not (w.parent / extra).exists():
        shutil.copy2(src / extra, w.parent / extra); print(f"✓ {extra} → {w.parent}")

if a.zips:
    test_ids = set(json.load(open(dst / "config.json"))["splits"]["test"])
    out = ROOT / "data" / "images"; out.mkdir(parents=True, exist_ok=True); n = 0
    for zp in a.zips:
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                name = Path(m).name
                if name.split(".")[0] in test_ids and not (out / name).exists():
                    with z.open(m) as f, open(out / name, "wb") as g: shutil.copyfileobj(f, g)
                    n += 1
    print(f"✓ {n} CT de test → {out}")
print("Listo. Ahora: docker compose up -d --build")
