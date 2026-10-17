"""HD95 y ASSD por fragmento en test (extensión opcional del enunciado, §5).

Lee las predicciones exportadas por el notebook S10 (data/dashboard/*.npz + *.json), empareja cada fragmento GT con
uno predicho dentro de su región (húngaro sobre IoU, igual que el Dice por fragmento) y mide, en mm con el espaciado
remuestreado (resolución de trabajo, 256 px):
  - HD95: máximo de los percentiles 95 de las distancias superficie→superficie en ambos sentidos.
  - ASSD: promedio de todas las distancias superficie→superficie en ambos sentidos.
Los fragmentos GT sin pareja (fusionados con el principal o no detectados) no tienen HD95/ASSD y se cuentan aparte.

Uso:  python scripts/hd95_assd.py --export_dir data/dashboard --out results/hd95_assd.csv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

REGIONS = {1: "Sacro", 2: "Coxal izq.", 3: "Coxal der."}


def gt_info(gt, rlut):
    ids = np.unique(gt); ids = ids[ids > 0]
    vol = np.bincount(gt.ravel())
    info = pd.DataFrame(dict(id=ids.astype(int), region=rlut[ids].astype(int), vol=vol[ids]))
    info["is_main"] = info.groupby("region").vol.transform("max") == info.vol   # principal = el de mayor volumen
    return info


def surface_distances(a, b, sp, margin=3):
    """Distancias (mm) de la superficie de a a la de b y viceversa, en un recorte alrededor de ambas máscaras."""
    sl = ndi.find_objects((a | b).astype(np.uint8))[0]
    sl = tuple(slice(max(s.start - margin, 0), s.stop + margin) for s in sl)
    a, b = a[sl], b[sl]
    sa = a & ~ndi.binary_erosion(a)
    sb = b & ~ndi.binary_erosion(b)
    d_ab = ndi.distance_transform_edt(~sb, sampling=sp)[sa]
    d_ba = ndi.distance_transform_edt(~sa, sampling=sp)[sb]
    return d_ab, d_ba


def evaluate_case(npz, meta):
    z = np.load(npz)
    gt, pred, rlut, sp = z["gt"].astype(np.int32), z["inst"].astype(np.int32), z["rlut"], tuple(float(s) for s in z["sp_rs"])
    ginfo = gt_info(gt, rlut)
    pinfo = pd.DataFrame(meta["fragments"])
    rows = []
    for r in REGIONS:
        g_ids = ginfo[ginfo.region == r].id.values
        p_ids = pinfo[pinfo.region == r].id.values.astype(int) if len(pinfo) else np.array([], int)
        if not len(g_ids): continue
        I = np.array([[np.count_nonzero((gt == g) & (pred == p)) for p in p_ids] for g in g_ids]).reshape(len(g_ids), len(p_ids))
        sg = np.array([np.count_nonzero(gt == g) for g in g_ids]); sp_ = np.array([np.count_nonzero(pred == p) for p in p_ids])
        IoU = I / np.maximum(sg[:, None] + sp_[None] - I, 1) if len(p_ids) else I
        pair = {}
        if len(p_ids):
            gi, pj = linear_sum_assignment(-IoU)
            pair = {g: p for g, p in zip(gi, pj) if IoU[g, p] > 0}
        for k, g in enumerate(g_ids):
            row = dict(case_id=meta["case_id"], region=REGIONS[r], gt_id=int(g),
                       tipo="principal" if bool(ginfo.set_index("id").is_main[g]) else "conminuto",
                       vol_gt_cm3=round(sg[k] * np.prod(sp) / 1000, 2), emparejado=k in pair,
                       dice=np.nan, hd95_mm=np.nan, assd_mm=np.nan)
            if k in pair:
                p = p_ids[pair[k]]; a, b = gt == g, pred == p
                d_ab, d_ba = surface_distances(a, b, sp)
                row.update(dice=2 * I[k, pair[k]] / (sg[k] + sp_[pair[k]]),
                           hd95_mm=max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)),
                           assd_mm=np.concatenate([d_ab, d_ba]).mean())
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", default="data/dashboard")
    ap.add_argument("--out", default="results/hd95_assd.csv")
    a = ap.parse_args()
    files = sorted(Path(a.export_dir).glob("*.npz"))
    if not files: raise SystemExit(f"No hay .npz en {a.export_dir}")
    rows = []
    for f in files:
        meta = json.load(open(f.with_suffix(".json"), encoding="utf-8"))
        rows += evaluate_case(f, meta)
        print(f"caso {meta['case_id']} listo", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(a.out, index=False)

    m = df[df.emparejado]
    res = []
    for nombre, t in (("Principal", "principal"), ("Conminutos", "conminuto"), ("Todos", None)):
        g, mg = (df, m) if t is None else (df[df.tipo == t], m[m.tipo == t])
        res.append(dict(grupo=nombre, fragmentos_gt=len(g), emparejados=len(mg),
                        hd95_media=mg.hd95_mm.mean(), hd95_mediana=mg.hd95_mm.median(),
                        assd_media=mg.assd_mm.mean(), assd_mediana=mg.assd_mm.median()))
    res = pd.DataFrame(res)
    reg = m.groupby("region")[["hd95_mm", "assd_mm"]].median()
    print("\n=== HD95 / ASSD por fragmento (test, mm, resolución de trabajo) ===")
    print(res.round(2).to_string(index=False))
    print("\nMedianas por región (fragmentos emparejados):")
    print(reg.round(2).to_string())
    print(f"\nDetalle por fragmento: {a.out}")


if __name__ == "__main__":
    main()