"""PENGWIN — núcleo de inferencia y visualización (extraído del notebook S8–S10).

Misma arquitectura y post-proceso que el notebook: los nombres de las capas coinciden con
`s10_best.pth`, así que el checkpoint carga sin conversión. Uso académico, no clínico.
"""
import json, math, re, shutil, tempfile, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ constantes (idénticas al notebook)
REGIONS = {1: "Sacro", 2: "Coxal izq.", 3: "Coxal der."}
REGION_COLORS = {1: "#E4572E", 2: "#29AB87", 3: "#3F88C5"}
WINDOW = dict(level=400, width=1800)
HU_BONE_THR = 200
TARGET_SIZE, S, NC = 256, 9, 3
NOUT, NSEG = 5 + NC, NC + 1
N_CTX, CTX_MM = 5, 2.0
CONF_OP, GATE_MARGIN = 0.30, 6
DEFAULT_PP = dict(min_mm3=250.0, thr_main=128, thr_bnd=128)
DISCLAIMER = ("⚠ USO EXCLUSIVAMENTE ACADÉMICO. Este sistema no es un dispositivo médico, no ha sido validado "
              "clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.")


def hex_rgb(col):
    return np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def shade(col, k):
    """Tono k-ésimo del color de la región (principal = tono base, conminutos más oscuros/claros)."""
    rgb = hex_rgb(col) / 255
    f = [0, .55, -.40, .80, -.60, .30, -.20, .65, -.50, .45][k % 10]      # alterna aclarar/oscurecer
    rgb = rgb + (1 - rgb) * f if f >= 0 else rgb * (1 + f)
    return (rgb.clip(0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ carga y preprocesamiento
def load_volume(path, dtype=np.int16):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    if dtype is not None: arr = arr.astype(dtype, copy=False)
    return arr, dict(spacing_zyx=tuple(float(s) for s in img.GetSpacing()[::-1]), shape_zyx=arr.shape)


# ------------------------------------------------------------------ entrada flexible: un volumen, o sus cortes sueltos
VOL_EXT = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd", ".nhdr")
IMG8_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def _expand(paths):
    """Descomprime los .zip y devuelve la lista plana de archivos (sin carpetas ni archivos ocultos)."""
    out = []
    for p in map(Path, paths):
        if p.suffix.lower() == ".zip":
            d = Path(tempfile.mkdtemp(prefix="pengwin_zip_"))
            with zipfile.ZipFile(p) as z:
                for m in z.infolist():
                    name = Path(m.filename)
                    if m.is_dir() or name.name.startswith(".") or "__MACOSX" in name.parts or ".." in name.parts: continue
                    dst = d / name.name if not (d / name.name).exists() else d / f"{len(out)}_{name.name}"
                    with z.open(m) as src, open(dst, "wb") as f: shutil.copyfileobj(src, f)
                    out.append(dst)
        elif p.is_dir():
            out += [q for q in p.rglob("*") if q.is_file() and not q.name.startswith(".")]
        else:
            out.append(p)
    return out


def _is_dicom(p):
    try:
        with open(p, "rb") as f: f.seek(128); return f.read(4) == b"DICM"
    except OSError:
        return False


def load_input(paths, sp_default=(1.0, 0.8, 0.8)):
    """Carga un CT desde: un volumen 3D (.mha/.nii/.nii.gz/.nrrd), una serie DICOM, cortes 2D sueltos con header
    (.mha/.nii/.nrrd) o cortes PNG/JPG/TIF, sueltos o en un .zip. Cualquier número de cortes y cualquier tamaño en plano.
    Devuelve (vol[z,y,x], meta). meta['pre8'] = True si son imágenes de 8 bits (ya ventaneadas, sin HU)."""
    files = _expand([paths] if isinstance(paths, (str, Path)) else paths)
    if not files: raise ValueError("No se recibió ningún archivo.")
    lower = lambda p: str(p).lower()
    vols = [p for p in files if lower(p).endswith(VOL_EXT)]
    # 1) un único volumen 3D
    if len(files) == 1 and vols:
        img = sitk.ReadImage(str(files[0]))
        if img.GetDimension() == 3 and img.GetSize()[2] > 1:
            arr, meta = load_volume(files[0])
            return arr, dict(meta, pre8=False, fuente=f"volumen {Path(files[0]).name}")
    # 2) serie DICOM (se ordena por posición del paciente; aplica pendiente/intercepto → HU)
    dcm = [p for p in files if lower(p).endswith((".dcm", ".ima")) or _is_dicom(p)]
    if dcm:
        d = Path(tempfile.mkdtemp(prefix="pengwin_dcm_"))
        for i, p in enumerate(dcm): shutil.copy(p, d / f"{i:05d}.dcm")
        r = sitk.ImageSeriesReader()
        series = r.GetGDCMSeriesIDs(str(d)) or [""]
        names = max((r.GetGDCMSeriesFileNames(str(d), sid) for sid in series), key=len)
        if len(names) > 1 and len(names) >= 0.5 * len(dcm):
            r.SetFileNames(names); img = r.Execute()
            arr = sitk.GetArrayFromImage(img).astype(np.int16)
            sp = tuple(float(v) for v in img.GetSpacing()[::-1])
            return arr, dict(spacing_zyx=sp, shape_zyx=arr.shape, pre8=False, fuente=f"serie DICOM de {len(names)} cortes")
        files = [d / f"{i:05d}.dcm" for i in range(len(dcm))]       # serie incompleta o un solo corte: corte por corte
    # 3) cortes 2D sueltos (con header o imágenes de 8 bits)
    sl = [p for p in files if lower(p).endswith(VOL_EXT + IMG8_EXT + (".dcm",))]
    if not sl: raise ValueError("Formato no reconocido. Usen .mha, .nii(.gz), .nrrd, DICOM, PNG/JPG/TIF o un .zip con los cortes.")
    pre8 = all(lower(p).endswith(IMG8_EXT) for p in sl)
    if len(sl) == 1 and not pre8:                                 # un único archivo con header (2D o 3D de 1 corte)
        arr, meta = load_volume(sl[0])
        if arr.ndim == 2: arr = arr[None]
        sp = meta["spacing_zyx"] if len(meta["spacing_zyx"]) == 3 else (sp_default[0],) + meta["spacing_zyx"]
        return arr, dict(spacing_zyx=(sp_default[0],) + tuple(sp[1:]), shape_zyx=arr.shape, pre8=False, fuente="1 corte con header")
    imgs = []
    for p in sl:
        im = sitk.ReadImage(str(p))
        a = sitk.GetArrayFromImage(im)
        if im.GetNumberOfComponentsPerPixel() > 1: a = a[..., :3].mean(-1)       # RGB → gris
        a = np.squeeze(a)
        if a.ndim == 3: imgs += [(a[k], im, p) for k in range(len(a))]           # un sub-volumen dentro de la lista
        else: imgs.append((a, im, p))
    origins = [im.GetOrigin()[-1] if im.GetDimension() == 3 else None for _, im, _ in imgs]
    if not pre8 and all(o is not None for o in origins) and len(set(origins)) == len(origins):
        imgs = [t for _, t in sorted(zip(origins, imgs), key=lambda q: q[0])]
        oz = np.sort(np.array(origins, float)); dz = float(np.median(np.diff(oz))) if len(oz) > 1 else sp_default[0]
    else:
        imgs.sort(key=lambda t: _natural_key(t[2])); dz = sp_default[0]
        if not pre8 and imgs[0][1].GetDimension() == 3 and imgs[0][1].GetSize()[2] == 1 and len(imgs) == 1:
            dz = float(imgs[0][1].GetSpacing()[2])
    shape = max({a.shape for a, _, _ in imgs}, key=lambda sh: sum(a.shape == sh for a, _, _ in imgs))
    out = []
    for a, _, _ in imgs:                                                        # cortes de otro tamaño → al tamaño dominante
        if a.shape != shape:
            t = torch.from_numpy(a.astype(np.float32))[None, None]
            a = F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        out.append(a)
    arr = np.stack(out)
    im0 = imgs[0][1]
    if pre8:
        arr = np.clip(arr, 0, 255).astype(np.uint8); sp = (dz, sp_default[1], sp_default[2])
    else:
        arr = arr.astype(np.int16); sxy = im0.GetSpacing()[:2]; sp = (abs(dz), float(sxy[1]), float(sxy[0]))
    return arr, dict(spacing_zyx=tuple(float(v) for v in sp), shape_zyx=arr.shape, pre8=pre8,
                     fuente=f"{len(arr)} cortes {'de 8 bits (sin HU)' if pre8 else 'con header'}")


def window_hu(vol, level=400, width=1800):
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(vol, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def to_grid(img, size=TARGET_SIZE, chunk=48, pre8=False):
    """Ventana ósea → padding a cuadrado → resize bilineal a 256 (igual que prep_volume del notebook).
    pre8: cortes de 8 bits ya ventaneados (PNG/JPG), se usan tal cual en [0, 1]."""
    nz, ny, nx = img.shape; side = max(ny, nx)
    py, px = (side - ny) // 2, (side - nx) // 2
    pad = ((0, 0), (py, side - ny - py), (px, side - nx - px)); out = []
    for i in range(0, nz, chunk):
        a = img[i:i + chunk].astype(np.float32) / 255 if pre8 else window_hu(img[i:i + chunk], **WINDOW)
        t = torch.from_numpy(np.pad(a, pad))[:, None].float()
        out.append(F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0])
    x = (torch.cat(out).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    return x, size / side, (py, px)


def ctx_index(nz, z, sp_z, n=N_CTX, mm=CTX_MM):
    offs = np.round((np.arange(n) - n // 2) * mm / sp_z).astype(int)
    return np.clip(z + offs, 0, nz - 1)


# ------------------------------------------------------------------ modelo (mismos nombres de capas que el notebook)
def conv_bn(ci, co, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, p, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=False), nn.MaxPool2d(2, 2))


class FundidoraPC(nn.Module):
    def __init__(self, in_channels=3, widths=(32, 64, 128, 256)):
        super().__init__()
        self.f1 = conv_block(in_channels, widths[0]); self.f2 = conv_block(widths[0], widths[1])
        self.f3 = conv_block(widths[1], widths[2]); self.f4 = conv_block(widths[2], widths[3])
        self.gap = nn.AdaptiveAvgPool2d(1); self.out_features = widths[3]

    def forward(self, x):
        f = self.f4(self.f3(self.f2(self.f1(x))))
        return self.gap(f).flatten(1), f


class CBAM(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(c // r, c, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2, bias=False); self.last_sa = None

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        self.last_sa = sa.detach()
        return x * sa


class Stem3D(nn.Module):
    def __init__(self, n=N_CTX, mid=8):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(1, mid, 3, padding=1, bias=False), nn.BatchNorm3d(mid), nn.ReLU(inplace=True),
                                  nn.Conv3d(mid, 3, (n, 1, 1)))

    def forward(self, x):
        c = x.shape[1] // 2
        return x[:, c:c + 1].expand(-1, 3, -1, -1) + self.conv(x[:, None])[:, :, 0]


class Up(nn.Module):
    def __init__(self, ci, cs, co):
        super().__init__(); self.conv = nn.Sequential(conv_bn(ci + cs, co), conv_bn(co, co))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class PengwinNet10(nn.Module):
    def __init__(self, use_cbam=True, n_ctx=N_CTX):
        super().__init__()
        self.stem = Stem3D(n_ctx); self.backbone = FundidoraPC(3); c = self.backbone.out_features
        w = [32, 64, 128]
        self.cbam = CBAM(c) if use_cbam else nn.Identity()
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, NC))
        self.to_grid = conv_bn(c, 256, k=3, s=2, p=2)
        self.det_head = nn.Sequential(conv_bn(256, 256), nn.Conv2d(256, NOUT, 1))
        self.up3, self.up2 = Up(c, w[2], 128), Up(128, w[1], 64)
        self.up1, self.up0 = Up(64, w[0], 32), Up(32, 3, 16)
        self.seg_out = nn.Conv2d(16, NSEG + 2, 1)

    def forward(self, x):
        s = self.stem(x); b = self.backbone
        f1 = b.f1(s); f2 = b.f2(f1); f3 = b.f3(f2); f4 = self.cbam(b.f4(f3))
        seg = self.seg_out(self.up0(self.up1(self.up2(self.up3(f4, f3), f2), f1), s))
        return dict(det=self.det_head(self.to_grid(f4)), cls=self.cls_head(f4),
                    sem=seg[:, :NSEG], main=seg[:, NSEG], bnd=seg[:, NSEG + 1])


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = PengwinNet10(ck.get("use_cbam", True), ck.get("n_ctx", N_CTX))
    m.load_state_dict(ck["model"]); return m.to(device).eval(), ck


# ------------------------------------------------------------------ detección: decodificación + NMS propio
def box_iou(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda t: (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-7)


def cell_to_xyxy(xywh, size=TARGET_SIZE):
    S_ = xywh.shape[-1]; cell = size / S_
    gy, gx = torch.meshgrid(torch.arange(S_, device=xywh.device), torch.arange(S_, device=xywh.device), indexing="ij")
    cx, cy = (gx + xywh[:, 0]) * cell, (gy + xywh[:, 1]) * cell
    w, h = xywh[:, 2] * size, xywh[:, 3] * size
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def nms(boxes, scores, iou_thr=0.5):
    order, keep = scores.argsort(descending=True), []
    while order.numel():
        i = order[0]; keep.append(i.item())
        if order.numel() == 1: break
        iou = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def postprocess(det, conf_thr=CONF_OP, iou_thr=0.5, max_per_class=1):
    det = det.float()
    boxes = cell_to_xyxy(det[:, 1:5].sigmoid()).flatten(1, 2)
    obj = det[:, 0].sigmoid().flatten(1)
    cprob = det[:, 5:].softmax(1).flatten(2).transpose(1, 2)
    out = []
    for b in range(det.shape[0]):
        conf, cl = cprob[b].max(1); score = obj[b] * conf; res = []
        for c in range(NC):
            m = (cl == c) & (score >= conf_thr)
            if not m.any(): continue
            k = nms(boxes[b][m], score[m], iou_thr)[:max_per_class]
            res.append(torch.cat([boxes[b][m][k], score[m][k, None], torch.full((len(k), 1), float(c), device=det.device)], 1))
        out.append(torch.cat(res) if res else torch.zeros(0, 6, device=det.device))
    return out


def gate_sem(sem_lbl, boxes, margin=GATE_MARGIN):
    keep = torch.zeros_like(sem_lbl, dtype=torch.bool); H, W = sem_lbl.shape[-2:]
    for x0, y0, x1, y1, s, c in boxes.tolist():
        ys = slice(max(int(y0) - margin, 0), min(int(math.ceil(y1)) + margin, H))
        xs = slice(max(int(x0) - margin, 0), min(int(math.ceil(x1)) + margin, W))
        keep[ys, xs] |= sem_lbl[ys, xs] == int(c) + 1
    return torch.where(keep, sem_lbl, torch.zeros_like(sem_lbl))


# ------------------------------------------------------------------ inferencia volumétrica + instancias 3D
@torch.no_grad()
def infer_volume(model, x_u8, sp_z, device, bs=32, n_ctx=N_CTX):
    nz = len(x_u8); H = W = TARGET_SIZE; amp = device.type == "cuda"
    probs = torch.zeros(nz, NSEG, H, W, dtype=torch.float16)
    pm = torch.zeros(nz, H, W, dtype=torch.float16); pb = pm.clone(); boxes = []
    for i in range(0, nz, bs):
        zs = range(i, min(i + bs, nz))
        x = torch.from_numpy(np.stack([x_u8[ctx_index(nz, z, sp_z, n_ctx)] for z in zs])).float().div(255).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            o = model(x)
        probs[i:i + len(zs)] = o["sem"].float().softmax(1).half().cpu()
        pm[i:i + len(zs)] = o["main"].float().sigmoid().half().cpu()
        pb[i:i + len(zs)] = o["bnd"].float().sigmoid().half().cpu()
        boxes += [q.cpu() for q in postprocess(o["det"], CONF_OP)]

    def zsmooth(t):
        s = t.clone()
        if nz > 2: s[1:-1] = ((t[:-2].float() + 2 * t[1:-1].float() + t[2:].float()) / 4).half()
        return s
    sm, pm, pb = zsmooth(probs), zsmooth(pm), zsmooth(pb)
    sem = torch.stack([gate_sem(sm[z].float().argmax(0), boxes[z]) for z in range(nz)]).numpy().astype(np.uint8)
    u8 = lambda t: (t.float() * 255).round().to(torch.uint8).numpy()
    return dict(sem=sem, pmain=u8(pm), pbnd=u8(pb), boxes=[b.numpy() for b in boxes])


def instances_3d(sem, pmain, pbnd, sp, min_mm3=250.0, thr_main=128, thr_bnd=128):
    min_vox = max(1, int(min_mm3 / np.prod(sp)))
    inst = np.zeros(sem.shape, np.int32); info = []; nid = 0
    for r in REGIONS:
        R = sem == r
        if R.sum() < min_vox: continue
        sl = ndi.find_objects(R.astype(np.uint8))[0]; Rs = R[sl]; sub = np.zeros(Rs.shape, np.int32)
        is_main = pmain[sl] >= thr_main
        lm, n = ndi.label(Rs & is_main)
        if n:
            nid += 1; sub[lm == np.bincount(lm.ravel())[1:].argmax() + 1] = nid
            info.append(dict(id=nid, region=r, is_main=True))
        lr_, n = ndi.label(Rs & ~is_main & (pbnd[sl] < thr_bnd))
        if n:
            sizes = np.bincount(lr_.ravel())[1:]
            for j in np.argsort(-sizes):
                if sizes[j] < min_vox: break
                nid += 1; sub[lr_ == j + 1] = nid; info.append(dict(id=nid, region=r, is_main=False))
        if not (sub > 0).any(): continue
        if not any(d["is_main"] and d["region"] == r for d in info):
            ids = [d["id"] for d in info if d["region"] == r]
            big = ids[int(np.argmax([(sub == i).sum() for i in ids]))]
            next(d for d in info if d["id"] == big)["is_main"] = True
        un = Rs & (sub == 0)
        if un.any():
            idx = ndi.distance_transform_edt(sub == 0, sampling=sp, return_distances=False, return_indices=True)
            sub[un] = sub[tuple(a[un] for a in idx)]
        inst[sl] = np.where(Rs, sub, inst[sl])
    return inst, pd.DataFrame(info, columns=["id", "region", "is_main"]).astype({"id": int, "region": int, "is_main": bool})


def separation(inst, info, sp, margin=5):
    """Distancia mínima borde a borde [mm] de cada conminuto al principal de su región (EDT con espaciado físico)."""
    out = {}
    for r in REGIONS:
        ir = info[info.region == r]
        if len(ir) < 2 or not ir.is_main.any(): continue
        main = int(ir[ir.is_main].id.iloc[0]); rm = np.isin(inst, ir.id.values)
        sl = tuple(slice(max(s.start - margin, 0), s.stop + margin) for s in ndi.find_objects(rm.astype(np.uint8))[0])
        sub = inst[sl]; dt = ndi.distance_transform_edt(sub != main, sampling=sp)
        for i in ir[~ir.is_main].id.values:
            m = sub == i
            if m.any(): out[int(i)] = float(dt[m].min())
    return out


def gt_instances(gt, rlut):
    """GT a 256 px → (inst, info) con el principal = fragmento de mayor volumen de cada región."""
    ids = np.unique(gt); ids = ids[ids > 0]
    if not len(ids): return gt.astype(np.int32), pd.DataFrame(columns=["id", "region", "is_main"])
    vol = np.bincount(gt.ravel(), minlength=256)
    info = pd.DataFrame(dict(id=ids.astype(int), region=rlut[ids].astype(int), vol=vol[ids]))
    info["is_main"] = info.groupby("region").vol.transform("max") == info.vol
    return gt.astype(np.int32), info[["id", "region", "is_main"]]


def fragment_table(case):
    info, inst, sp = case["info"], case["inst"], case["sp_rs"]; vox = float(np.prod(sp))
    vol = np.bincount(inst.ravel(), minlength=int(inst.max()) + 1)
    rows = []
    for r in REGIONS:
        ir = info[info.region == r].reset_index(drop=True)
        for k, row in ir.iterrows():
            d = row.get("dist_mm", np.nan)
            rows.append({"Región": REGIONS[r], "Fragmento": "principal" if row.is_main else f"conminuto {k}",
                         "Volumen [cm³]": round(vol[int(row.id)] * vox / 1000, 1),
                         "Separación pred. [mm]": "—" if row.is_main or pd.isna(d) else round(float(d), 2)})
    df = pd.DataFrame(rows)
    gs = case.get("gt_sep")
    if gs is not None and len(df):
        df["Separación GT [mm] (región)"] = np.where(df["Fragmento"] == "principal", "—", df["Región"].map(gs).fillna("—"))
    return df


# ------------------------------------------------------------------ casos: precalculados o inferencia en vivo
def label_fragments(info):
    """Etiqueta legible por fragmento: 'Coxal izq. · conminuto 2'."""
    lab = {}
    for r in REGIONS:
        ir = info[info.region == r].reset_index(drop=True)
        for k, row in ir.iterrows():
            lab[int(row.id)] = f"{REGIONS[r]} · " + ("principal" if row.is_main else f"conminuto {k}")
    return lab


def _gt_sep_summary(gt, rlut, sp):
    gi, ginfo = gt_instances(gt, rlut)
    d = separation(gi, ginfo, sp)
    s = {}
    for r in REGIONS:
        v = [round(d[i], 2) for i in ginfo[(ginfo.region == r) & ~ginfo.is_main].id if i in d]
        if v: s[REGIONS[r]] = ", ".join(map(str, sorted(v)))
    return s


def _find_raw(case_id, raw, raw_dir):
    """Ruta del CT crudo: la guardada en Colab si existe; si no, el archivo del caso dentro de raw_dir (Docker)."""
    if raw and Path(raw).exists(): return raw
    if raw_dir and Path(raw_dir).exists():
        for p in sorted(Path(raw_dir).rglob(f"{case_id}.*")):
            if p.name.endswith((".mha", ".nii.gz", ".nii", ".nrrd")): return str(p)
    return raw


def load_exported(npz_path, raw_dir=None):
    z = np.load(npz_path); meta = json.load(open(Path(npz_path).with_suffix(".json")))
    meta["image"] = _find_raw(meta["case_id"], meta.get("image"), raw_dir)
    info = pd.DataFrame(meta["fragments"]); info = info if len(info) else pd.DataFrame(columns=["id", "region", "is_main", "dist_mm"])
    info["is_main"] = info.is_main.astype(bool); info["region"] = info.region.astype(int); info["id"] = info.id.astype(int)
    boxes = [b[~np.isnan(b[:, 0])] for b in z["boxes"]]
    case = dict(case_id=meta["case_id"], raw_path=meta.get("image"), x=z["x"], inst=z["inst"], info=info,
                boxes=boxes, sp_rs=tuple(z["sp_rs"]), sp=tuple(z["sp"]), gt=z["gt"], rlut=z["rlut"], fuente="precalculado (S10)")
    case["gt_sep"] = _gt_sep_summary(case["gt"], case["rlut"], case["sp_rs"])
    return case


def run_live(model, device, image_path, pp=DEFAULT_PP, case_id=None, sp_default=(1.0, 0.8, 0.8)):
    """image_path: un archivo o una lista (cortes, serie DICOM, .zip). Ver load_input."""
    t0 = time.perf_counter()
    img, meta = load_input(image_path, sp_default); sp = meta["spacing_zyx"]
    x, scale, pad = to_grid(img, pre8=meta["pre8"]); sp_rs = (sp[0], sp[1] / scale, sp[2] / scale)
    t1 = time.perf_counter()
    if device.type == "cuda": torch.cuda.synchronize()
    pr = infer_volume(model, x, sp[0], device)
    if device.type == "cuda": torch.cuda.synchronize()
    t2 = time.perf_counter()
    inst, info = instances_3d(pr["sem"], pr["pmain"], pr["pbnd"], sp_rs, **pp)
    d = separation(inst, info, sp_rs); info["dist_mm"] = info.id.map(d)
    t3 = time.perf_counter()
    lat = dict(dispositivo=device.type.upper(), cortes=len(x), carga_s=round(t1 - t0, 2), red_s=round(t2 - t1, 2),
               ms_por_corte=round((t2 - t1) / len(x) * 1000, 1), instancias_s=round(t3 - t2, 2), total_s=round(t3 - t0, 2))
    single = isinstance(image_path, (str, Path))
    return dict(case_id=case_id or (Path(image_path).name if single else f"{len(image_path)} archivos"),
                raw_path=str(image_path) if single else None, raw_vol=(img, sp, meta["pre8"]), x=x, inst=inst, info=info,
                boxes=pr["boxes"], sp_rs=sp_rs, sp=sp, gt=None, rlut=None,
                fuente=f"inferencia en vivo ({lat['dispositivo']}) · {meta['fuente']}", pre8=meta["pre8"], latencia=lat)


# ------------------------------------------------------------------ Visualizador 1: MIP rotacional del volumen crudo
def body_mask(iso, iso_mm=2.0, hu=-400, open_mm=6):
    """Máscara del paciente: todo lo que supera -400 HU, abierto morfológicamente (borra la camilla, que es una
    lámina delgada pegada a la espalda) y quedándose con la componente más grande (descarta el fantoma)."""
    st = ndi.generate_binary_structure(3, 1); it = max(1, int(round(open_mm / iso_mm)))
    m = ndi.binary_erosion(iso > hu, st, iterations=it)
    lab, n = ndi.label(m)
    if n: m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    m = ndi.binary_dilation(m, st, iterations=it) & (iso > hu)      # apertura exacta: no vuelve a crecer hacia la camilla
    for z in range(len(m)): m[z] = ndi.binary_fill_holes(m[z])
    return m


def _iso_volume(case, iso_mm=2.0):
    """Volumen crudo remuestreado a vóxel isotrópico + máscara de cuerpo (se calcula una vez por caso)."""
    if case.get("_iso") is not None: return case["_iso"]
    raw = case.get("raw_path")
    lo = WINDOW["level"] - WINDOW["width"] / 2
    if case.get("raw_vol") is not None:                         # inferencia en vivo: el volumen ya cargado
        vol, sp, pre8 = case["raw_vol"]; sp = np.array(sp)
        vol = vol.astype(np.float32) / 255 * WINDOW["width"] + lo if pre8 else vol
        fuente = "cortes de 8 bits (HU aproximadas)" if pre8 else "volumen crudo (HU del header)"
    elif raw and Path(raw).exists():
        vol, meta = load_volume(raw); sp = np.array(meta["spacing_zyx"]); fuente = "volumen crudo (HU del header)"
    else:   # respaldo: HU aproximadas reconstruidas desde la ventana ósea a 256 px
        lo = WINDOW["level"] - WINDOW["width"] / 2
        vol = case["x"].astype(np.float32) / 255 * WINDOW["width"] + lo; sp = np.array(case["sp_rs"])
        fuente = "respaldo: volumen a 256 px"
    iso = ndi.zoom(vol.astype(np.float32), sp / iso_mm, order=1)
    case["_iso"] = (iso, body_mask(iso, iso_mm), fuente)
    return case["_iso"]


def _mip_frames(case, thr=HU_BONE_THR, limpiar=True, iso_mm=2.0, step=10, hu_max=1500, min_mm3=2000):
    """Umbral HU sobre el volumen SIN procesar → proyecciones MIP girando alrededor del eje craneocaudal. No usa el modelo.
    Devuelve (ángulos, cuadros [fila 0 = caudal], fuente)."""
    iso, body, fuente = _iso_volume(case, iso_mm)
    m = iso >= thr
    if limpiar: m &= body
    lab, n = ndi.label(m)
    if n:
        sizes = np.bincount(lab.ravel())[1:] * iso_mm ** 3
        m = np.isin(lab, np.nonzero(sizes >= min_mm3)[0] + 1)
    b = np.where(m, 40 + 215 * (np.clip(iso, thr, hu_max) - thr) / max(hu_max - thr, 1), 0).astype(np.uint8)
    zz, yy, xx = np.nonzero(m)
    if len(zz):                                                 # recorte al hueso (+ margen) para no dejar espacio vacío
        c = (len(b[0]) // 2, b.shape[2] // 2); r = int(np.ceil(np.hypot(yy - c[0], xx - c[1]).max())) + 4
        b = b[max(zz.min() - 3, 0):zz.max() + 4, max(c[0] - r, 0):c[0] + r, max(c[1] - r, 0):c[1] + r]
    angs = list(range(0, 360, step)); frames = []
    for a in angs:
        rot = ndi.rotate(b, a, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0)
        frames.append(rot.max(axis=1))
    return angs, frames, fuente


def mip_gif(case, path, thr=HU_BONE_THR, limpiar=True, iso_mm=2.0, step=10, size=520, ms=90):
    """Misma proyección rotacional, guardada como GIF animado (Gradio no reproduce las animaciones de Plotly)."""
    angs, frames, _ = _mip_frames(case, thr, limpiar, iso_mm, step)
    h, w = frames[0].shape; k = size / max(h, w)
    ims = [Image.fromarray(np.flipud(f)).resize((max(int(w * k), 1), max(int(h * k), 1)), Image.BILINEAR) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0)
    return str(path)


def mip_rotacional(case, thr=HU_BONE_THR, limpiar=True, iso_mm=2.0, step=10, hu_max=1500, min_mm3=2000):
    """Versión Plotly (notebook/Colab): heatmap con botón ▶ Girar y slider de ángulo."""
    angs, frames, fuente = _mip_frames(case, thr, limpiar, iso_mm, step, hu_max, min_mm3)
    nz_, nx_ = frames[0].shape
    hm = lambda f: go.Heatmap(z=f, colorscale="gray", showscale=False, zmin=0, zmax=255,
                              x=np.arange(nx_) * iso_mm, y=np.arange(nz_) * iso_mm, hoverinfo="skip")
    fig = go.Figure(data=[hm(frames[0])], frames=[go.Frame(data=[hm(f)], name=str(a)) for a, f in zip(angs, frames)])
    fig.update_layout(
        title=dict(text=f"MIP rotacional · HU ≥ {thr} · {'sin camilla ni fantoma' if limpiar else 'sin limpieza'} · {fuente}",
                   font=dict(size=13)),
        xaxis=dict(title="mm", scaleanchor="y", constrain="domain", showgrid=False, zeroline=False),
        yaxis=dict(title="mm", constrain="domain", showgrid=False, zeroline=False),
        height=600, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="black",
        updatemenus=[dict(type="buttons", x=0, y=-0.06, xanchor="left", buttons=[
            dict(label="▶ Girar", method="animate", args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=True)]),
            dict(label="⏸", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(currentvalue=dict(prefix="ángulo: ", suffix="°"), pad=dict(t=30),
                      steps=[dict(label=str(a), method="animate",
                                  args=[[str(a)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for a in angs])])
    return fig


# ------------------------------------------------------------------ Visualizador 2: corte con cajas, máscaras y distancias
def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(f).exists(): return ImageFont.truetype(f, size)
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


VENTANAS = {"Ósea 400/1800 (entrada de la red)": None, "Tejido blando 40/400": (40, 400),
            "Cortical 700/1500": (700, 1500), "Personalizada": "custom"}


def display_slice(x2d, window=None):
    """Re-ventanea el corte para mostrarlo. La red siempre recibe la ventana ósea; esto es solo visualización.
    Las HU se reconstruyen de la ventana ósea guardada (rango −500…1300 HU, ~7 HU por nivel de gris)."""
    if window is None: return x2d.astype(np.float32)
    lo = WINDOW["level"] - WINDOW["width"] / 2
    hu = x2d.astype(np.float32) / 255 * WINDOW["width"] + lo
    return window_hu(hu, *window) * 255


def render_slice(case, z, show_gt=False, up=3, alpha=0.5, window=None):
    x, inst, info = case["x"], case["inst"], case["info"]; z = int(np.clip(z, 0, len(x) - 1))
    base = np.repeat(display_slice(x[z], window)[..., None], 3, -1).astype(np.float32)
    rank = {}
    for r in REGIONS:
        for k, i in enumerate(info[info.region == r].id): rank[int(i)] = k
    for _, row in info.iterrows():
        m = inst[z] == row.id
        if m.any(): base[m] = (1 - alpha) * base[m] + alpha * shade(REGION_COLORS[int(row.region)], rank[int(row.id)])
    if show_gt and case.get("gt") is not None:          # contorno del GT en blanco para comparar
        g = case["gt"][z].astype(np.int32)
        edge = (g > 0) & ((ndi.maximum_filter(g, 3) != g) | (ndi.minimum_filter(g, 3) != g))
        base[edge] = 255
    im = Image.fromarray(base.clip(0, 255).astype(np.uint8)).resize((x.shape[2] * up, x.shape[1] * up), Image.NEAREST)
    dr = ImageDraw.Draw(im); f_box, f_d = _font(13), _font(14)
    for x0, y0, x1, y1, s, c in case["boxes"][z].tolist():
        col = tuple(int(v) for v in hex_rgb(REGION_COLORS[int(c) + 1]))
        dr.rectangle([x0 * up, y0 * up, x1 * up, y1 * up], outline=col, width=2)
        t = f"{REGIONS[int(c) + 1]} {s:.2f}"; tb = dr.textbbox((x0 * up, y0 * up - 17), t, font=f_box)
        dr.rectangle(tb, fill=col); dr.text((x0 * up, y0 * up - 17), t, fill="white", font=f_box)
    for _, row in info[~info.is_main].iterrows():          # distancia en mm junto a cada conminuto visible
        d = row.get("dist_mm", np.nan)
        ys, xs = np.nonzero(inst[z] == row.id)
        if not len(ys) or pd.isna(d): continue
        col = tuple(int(v) for v in hex_rgb(REGION_COLORS[int(row.region)]))
        cx, cy = xs.mean() * up, ys.mean() * up; t = f"{float(d):.1f} mm"
        tb = dr.textbbox((cx, cy), t, font=f_d, anchor="mm"); tb = (tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2)
        dr.rectangle(tb, fill=(20, 20, 20), outline=col, width=2); dr.text((cx, cy), t, fill="white", font=f_d, anchor="mm")
    dr.text((8, 6), f"z = {z}  ·  {z * case['sp_rs'][0]:.1f} mm", fill="white", font=_font(14))
    return im


def best_slice(case):
    """Corte con más conminutos visibles (o más hueso si no hay conminutos)."""
    info, inst = case["info"], case["inst"]
    if not len(info): return len(inst) // 2                     # el modelo no encontró hueso: corte central
    con = info[~info.is_main.astype(bool)].id.values
    score = np.isin(inst, con).sum((1, 2)) if len(con) else (inst > 0).sum((1, 2))
    return int(score.argmax())


# ------------------------------------------------------------------ Visualizador 3: reconstrucción 3D (marching cubes + Mesh3d)
def mesh_figure(case, step=2, smooth=0.8, min_vox=30):
    inst, info, sp = case["inst"], case["info"], np.array(case["sp_rs"])
    labels = label_fragments(info); fig = go.Figure(); lx, ly, lz, lt, lc = [], [], [], [], []
    rank = {}
    for r in REGIONS:
        for k, i in enumerate(info[info.region == r].id): rank[int(i)] = k
    for _, row in info.iterrows():
        m = inst == row.id
        if m.sum() < min_vox: continue
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in ndi.find_objects(m.astype(np.uint8))[0])
        f = ndi.gaussian_filter(m[sl].astype(np.float32), smooth)
        try:
            v, fc, _, _ = marching_cubes(f, 0.5, spacing=tuple(sp), step_size=step)
        except (ValueError, RuntimeError):
            continue
        v += np.array([s.start for s in sl]) * sp
        rgb = shade(REGION_COLORS[int(row.region)], rank[int(row.id)])
        d = row.get("dist_mm", np.nan)
        name = labels[int(row.id)] + ("" if row.is_main or pd.isna(d) else f" · {float(d):.1f} mm")
        fig.add_trace(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=fc[:, 0], j=fc[:, 1], k=fc[:, 2],
                                color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", opacity=1.0 if row.is_main else 0.95,
                                name=name, showlegend=True, hovertext=name, hoverinfo="text",
                                lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
                                lightposition=dict(x=100, y=-200, z=300)))
        zz, yy, xx = np.nonzero(m)
        lx.append(xx.mean() * sp[2]); ly.append(yy.min() * sp[1] - 12); lz.append(zz.mean() * sp[0])
        lt.append(name.replace("Coxal ", "C. ")); lc.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=lt, textfont=dict(size=11, color=lc),
                               hoverinfo="skip", showlegend=False))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="white",
                      title=dict(text="Fragmentos coloreados por macro-hueso · distancia de separación en la etiqueta", font=dict(size=13)),
                      legend=dict(orientation="h", y=-0.02, font=dict(size=11)),
                      scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]",
                                 camera=dict(eye=dict(x=0, y=-2.0, z=0.3), up=dict(x=0, y=0, z=1))))
    return fig