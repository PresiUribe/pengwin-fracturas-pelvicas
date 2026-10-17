"""Pruebas de humo del pipeline con un CT sintético (sin datos reales ni pesos entrenados).
Corren en CI en cada Pull Request:  pytest -q tests/"""
import json, sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pengwin_core as pc


def _fake_ct(path):
    nz, ny, nx = 60, 200, 220
    z, y, x = np.mgrid[:nz, :ny, :nx]
    vol = np.full((nz, ny, nx), -1000, np.int16)
    body = ((y - 90) / 70) ** 2 + ((x - 110) / 95) ** 2 < 1
    vol[body] = 40
    vol[10:50, 60:120, 40:90] = 700                     # "coxal"
    vol[10:50, 60:120, 130:180] = 700                   # "coxal"
    vol[20:30, 70:85, 92:100] = 700                     # "conminuto"
    vol[:, 160:163, 10:210] = 400                       # camilla
    img = sitk.GetImageFromArray(vol); img.SetSpacing((0.8, 0.8, 1.0)); sitk.WriteImage(img, str(path))
    return vol


def test_nms_y_iou():
    bx = torch.tensor([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30.]]); sc = torch.tensor([.9, .8, .7])
    assert pc.nms(bx, sc).tolist() == [0, 2]
    assert abs(pc.box_iou(bx[:1], bx[:1]).item() - 1) < 1e-6


def test_modelo_formas():
    m = pc.PengwinNet10().eval()
    with torch.no_grad():
        o = m(torch.zeros(2, pc.N_CTX, 256, 256))
    assert o["det"].shape == (2, pc.NOUT, pc.S, pc.S) and o["sem"].shape == (2, pc.NSEG, 256, 256)


def test_pipeline_en_vivo_y_visualizadores(tmp_path):
    _fake_ct(tmp_path / "001.mha")
    torch.save(dict(model=pc.PengwinNet10().state_dict(), use_cbam=True, n_ctx=pc.N_CTX), tmp_path / "ck.pth")
    model, _ = pc.load_model(tmp_path / "ck.pth", torch.device("cpu"))
    case = pc.run_live(model, torch.device("cpu"), tmp_path / "001.mha", case_id="001")
    assert case["x"].shape[1:] == (256, 256) and case["latencia"]["cortes"] == 60
    # visualizadores sobre instancias conocidas (las de un modelo sin entrenar no significan nada)
    inst = np.zeros_like(case["x"], np.int32); inst[20:40, 80:120, 60:100] = 1; inst[20:40, 80:120, 150:190] = 2
    inst[25:30, 90:95, 102:106] = 3
    info = pc.pd.DataFrame(dict(id=[1, 2, 3], region=[2, 3, 2], is_main=[True, True, False]))
    d = pc.separation(inst, info, case["sp_rs"]); info["dist_mm"] = info.id.map(d)
    case.update(inst=inst, info=info)
    assert 3 in d and d[3] > 0
    assert pc.render_slice(case, 27, window=(40, 400)).size == (768, 768)
    assert len(pc.mesh_figure(case).data) >= 3
    fig = pc.mip_rotacional(case, thr=200, step=90)
    assert len(fig.frames) == 4
    assert Image.open(pc.mip_gif(case, tmp_path / "mip.gif", thr=200, step=90)).n_frames == 4
    iso, body, _ = case["_iso"]
    assert not body[:, int(161 * 0.8 / 2), int(20 * 0.8 / 2)].any()   # camilla (y=161, x=20 a 0.8 mm → vóxel de 2 mm) fuera del cuerpo


def test_app_construye(tmp_path):
    exp = tmp_path / "dashboard"; exp.mkdir()
    json.dump(dict(postproceso=pc.DEFAULT_PP, metas=[], latencia_ms={}), open(exp / "config.json", "w"))
    import app
    assert app.build(exp, None, "cpu") is not None


def test_carga_cortes_sueltos(tmp_path):
    """El profesor puede subir el volumen o sus cortes (desordenados, otro tamaño, PNG o .zip)."""
    import zipfile
    vol = _fake_ct(tmp_path / "vol.mha")
    for k in range(len(vol)):                                    # cortes 2D con origen, nombres desordenados
        s = sitk.GetImageFromArray(vol[k:k + 1]); s.SetSpacing((0.8, 0.8, 1.0)); s.SetOrigin((0, 0, float(k)))
        sitk.WriteImage(s, str(tmp_path / f"c{(k * 7) % len(vol):03d}_{k}.mha"))
    files = sorted(tmp_path.glob("c*.mha"), reverse=True)
    arr, meta = pc.load_input(files)
    assert np.array_equal(arr, vol) and meta["spacing_zyx"] == (1.0, 0.8, 0.8) and not meta["pre8"]
    png = (pc.window_hu(vol) * 255).astype(np.uint8)
    with zipfile.ZipFile(tmp_path / "png.zip", "w") as z:
        for k in range(len(vol)):
            Image.fromarray(png[k]).save(tmp_path / f"{k + 1}.png"); z.write(tmp_path / f"{k + 1}.png", f"{k + 1}.png")
    arr, meta = pc.load_input(tmp_path / "png.zip", sp_default=(2.0, 0.7, 0.7))
    assert np.array_equal(arr, png) and meta["pre8"] and meta["spacing_zyx"] == (2.0, 0.7, 0.7)
    x, _, _ = pc.to_grid(arr, pre8=True)
    assert x.shape == (len(vol), 256, 256)