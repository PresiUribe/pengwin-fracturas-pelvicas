"""Dashboard PENGWIN (S11): 3 visualizadores + métricas + inferencia en vivo.

Uso:  python app.py --export_dir <outputs/dashboard> --ckpt <outputs/s10_best.pth> [--raw_dir <CT crudos>] [--port 7860]
Sin argumentos usa las variables de entorno PENGWIN_EXPORT_DIR, PENGWIN_CKPT, PENGWIN_RAW_DIR, PENGWIN_DEVICE y PORT
(así lo lanza el contenedor de Docker). En Colab se importa y se lanza con launch(build(...)).
El túnel de Cloudflare apunta a ese puerto.
"""
import argparse, json, os, re, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gradio as gr

import pengwin_core as pc

CSS = """
.gradio-container {max-width: 1320px !important; margin: auto}
#hdr {padding: 18px 22px; border-radius: 14px; color: white;
      background: linear-gradient(120deg, #1f2a44 0%, #2d4a7a 60%, #3F88C5 100%)}
#hdr h1 {margin: 0; font-size: 26px; color: white !important}
#hdr p {margin: 4px 0 0; color: #dbe7f5 !important; font-size: 14px}
.disclaimer {background:#fff4e5; border-left:5px solid #E4572E; border-radius:8px; padding:8px 12px;
             color:#7a2e12; font-weight:600; font-size:13px}
.legend span {display:inline-block; padding:3px 10px; margin:2px 4px 2px 0; border-radius:999px; color:white;
              font-weight:600; font-size:13px}
.cards {display:grid; grid-template-columns:repeat(2, 1fr); gap:8px}
.card {background:var(--block-background-fill); border:1px solid var(--border-color-primary); border-radius:10px; padding:8px 10px}
.card b {display:block; font-size:22px; line-height:1.2}
.card small {opacity:.7; font-size:12px}
.side {gap: 10px}
"""

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

GUIA = """
1. **Elija un paciente** en la lista de la izquierda. Se carga al instante lo que el modelo predijo para ese CT.
2. Recorra las pestañas:
   - **① Volumen crudo:** el CT girando 360° solo con umbral de hueso (sin modelo).
   - **② Corte a corte:** mueva el deslizador *z*. Cajas del detector, fragmentos coloreados por hueso y distancia en mm de cada conminuto.
   - **③ 3D:** malla interactiva (arrastre para girar, rueda para acercar).
   - **④ Fragmentos:** tabla del caso y métricas globales de test.
3. **Comparar con GT** dibuja el contorno de la anotación de los ortopedistas.
4. **Analizar en vivo** (opcional) ejecuta el modelo en este servidor sobre el CT del paciente o sobre uno que suba: un volumen o todos sus cortes (DICOM, `.mha`, PNG… o un `.zip`). Tarda ~2–3 min en CPU.
"""


GR6 = int(gr.__version__.split(".")[0]) >= 6


def launch(demo, port=7860, **kw):
    if GR6: kw.update(css=CSS, theme=THEME)
    kw.setdefault("share", False)              # el acceso público es el túnel de Cloudflare, no el enlace de gradio.live
    return demo.queue().launch(server_name="0.0.0.0", server_port=port, **kw)


def build(export_dir, ckpt=None, device=None, raw_dir=None):
    export_dir = Path(export_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = json.load(open(export_dir / "config.json")) if (export_dir / "config.json").exists() else {}
    pp = {k: float(v) if k == "min_mm3" else int(v) for k, v in cfg.get("postproceso", pc.DEFAULT_PP).items()}
    model = pc.load_model(ckpt, device)[0] if ckpt and Path(ckpt).exists() else None
    case_ids = sorted(p.stem for p in export_dir.glob("*.npz"))
    cache = {}

    def get_case(cid):
        if cid not in cache: cache[cid] = pc.load_exported(export_dir / f"{cid}.npz", raw_dir)
        return cache[cid]

    def metas_df():
        m = pd.DataFrame(cfg.get("metas", []))
        if len(m): m["Test"] = m["Test"].astype(float).round(3)
        return m

    def latency_md(case):
        lines = [f"**Dispositivo del servidor:** {device.type.upper()}"
                 + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")]
        lat = cfg.get("latencia_ms", {})
        for k, v in lat.items():
            if isinstance(v, dict):
                dev = "GPU" if "CUDA" in k.upper() else "CPU"
                lines.append(f"- Volumen completo en {dev} ({v.get('cortes')} cortes): red {v.get('red_s')} s"
                             f" + instancias 3D {v.get('instancias_s')} s")
            else:
                lines.append(f"- {k}: {float(v):.1f} ms por corte")
        if case.get("latencia"):
            l = case["latencia"]
            lines.append(f"- **Esta inferencia en vivo** ({l['dispositivo']}, {l['cortes']} cortes): red {l['red_s']} s "
                         f"({l['ms_por_corte']} ms/corte) + instancias 3D {l['instancias_s']} s · total {l['total_s']} s")
        return "\n".join(lines)

    def cards(case):
        info = case["info"]; vx = " × ".join(f"{float(v):.2f}" for v in case["sp_rs"])
        cid = str(case["case_id"]).replace(" (en vivo)", "")
        cid = cid if len(cid) <= 8 else f"<span style='font-size:15px;line-height:1.3'>{cid}</span>"
        items = [(cid, "paciente" if "en vivo" not in str(case["case_id"]) else "CT en vivo"), (len(case["x"]), "cortes axiales"),
                 (int(info.is_main.sum()), "fragmentos principales"), (int((~info.is_main).sum()), "conminutos predichos")]
        html = "".join(f"<div class='card'><b>{v}</b><small>{t}</small></div>" for v, t in items)
        return (f"<div class='cards'>{html}</div><p style='font-size:12px;opacity:.7;margin:6px 2px 0'>"
                f"Vóxel {vx} mm · {case['fuente']}</p>")

    gif_dir = Path(tempfile.gettempdir()) / "pengwin_mip"; gif_dir.mkdir(exist_ok=True)

    def _win(name, lvl, wid):
        w = pc.VENTANAS.get(name)
        return (float(lvl), float(wid)) if w == "custom" else w

    def outputs_for(case, show_gt, win=None):
        z = pc.best_slice(case); cache[case["case_id"]] = case       # el estado guarda solo la clave, no los arreglos
        return (case["case_id"],
                gr.update(maximum=len(case["x"]) - 1, value=z),
                pc.render_slice(case, z, show_gt, window=win),
                pc.mesh_figure(case),
                pc.fragment_table(case),
                cards(case),
                latency_md(case))

    def on_load(cid, show_gt, wn, lvl, wid):
        if not cid: raise gr.Error("Elijan un caso.")
        return outputs_for(get_case(cid), show_gt, _win(wn, lvl, wid))

    def on_live(cid, files, dz, dxy, show_gt, wn, lvl, wid, progress=gr.Progress()):
        if model is None: raise gr.Error("No se encontró el checkpoint s10_best.pth: la inferencia en vivo no está disponible.")
        files = [f if isinstance(f, str) else f.name for f in (files or [])]
        if files:
            path = files[0] if len(files) == 1 else files
            name = (Path(files[0]).name if len(files) == 1 else f"{len(files)} archivos") + time.strftime(" · %H:%M:%S")
        elif cid:
            path, name = get_case(cid).get("raw_path"), cid
            if not path or not Path(path).exists(): raise gr.Error("No está el volumen crudo de este caso en el servidor.")
        else:
            raise gr.Error("Elijan un caso o suban un archivo .mha / .nii.gz.")
        progress(0.1, desc=f"Inferencia en {device.type.upper()}…")
        try:
            case = pc.run_live(model, device, path, pp, case_id=f"{name} (en vivo)",
                               sp_default=(float(dz or 1.0), float(dxy or 0.8), float(dxy or 0.8)))
        except (ValueError, RuntimeError) as e:
            raise gr.Error(f"No se pudo leer el CT: {e}")
        if case.get("pre8"):
            gr.Warning(f"Cortes de 8 bits sin HU ni espaciado: se asumió que vienen en ventana ósea, con {dz} mm entre "
                       f"cortes y píxel de {dxy} mm. Las distancias en mm dependen de esos valores.")
        if not len(case["info"]):
            gr.Warning("El modelo no detectó hueso pélvico en estas imágenes. Revisen que sean cortes axiales de CT de pelvis.")
        if not files and cid:                       # mismo caso del dataset: se conserva el GT para comparar
            ref = get_case(cid); case.update(gt=ref["gt"], rlut=ref["rlut"], gt_sep=ref.get("gt_sep"))
        return outputs_for(case, show_gt, _win(wn, lvl, wid))

    def on_slice(key, z, show_gt, wn, lvl, wid):
        return pc.render_slice(cache[key], z, show_gt, window=_win(wn, lvl, wid)) if key in cache else None

    def on_mip(key, thr, limpiar, progress=gr.Progress()):
        if key not in cache: return None
        f = gif_dir / f"{re.sub(r'[^0-9A-Za-z]+', '_', key)}_{int(thr)}_{int(bool(limpiar))}.gif"
        if not f.exists():
            progress(0.2, desc="Umbral HU + máscara de cuerpo + proyección girando 360°…")
            pc.mip_gif(cache[key], f, thr=int(thr), limpiar=bool(limpiar))
        return str(f)

    def on_win(name):
        return [gr.update(visible=pc.VENTANAS.get(name) == "custom")] * 2

    legend = "".join(f'<span style="background:{pc.REGION_COLORS[r]}">{n}</span>' for r, n in pc.REGIONS.items())
    kw = {} if GR6 else dict(css=CSS, theme=THEME)     # Gradio 6 movió css y theme a launch()
    with gr.Blocks(title="PENGWIN · Fracturas pélvicas en CT", **kw) as demo:
        state = gr.State(None)
        gr.HTML("<div id='hdr'><h1>PENGWIN · Fracturas pélvicas en CT</h1>"
                "<p>Detección de huesos · segmentación de fragmentos · distancia de separación en mm</p></div>")
        gr.HTML(f'<div class="disclaimer">{pc.DISCLAIMER}</div>')
        with gr.Accordion("¿Cómo usar el dashboard?", open=False):
            gr.Markdown(GUIA)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=290, elem_classes="side"):
                dd = gr.Dropdown(case_ids, value=case_ids[0] if case_ids else None, label="Paciente de test",
                                 info="Al elegirlo se cargan sus resultados (ya calculados por el modelo)")
                show_gt = gr.Checkbox(label="Comparar con GT (contorno de los expertos)", value=False)
                gr.HTML(f'<div class="legend">{legend}</div>')
                info = gr.HTML()
                with gr.Accordion("Analizar en vivo (opcional)", open=False):
                    gr.Markdown(f"Ejecuta el modelo **ahora** en {device.type.upper()} sobre el CT del paciente elegido "
                                "o sobre un archivo que suba (~2–3 min en CPU).")
                    up = gr.File(label="CT propio, opcional: un volumen o sus cortes", file_count="multiple", height=120)
                    gr.Markdown("Acepta un volumen `.mha` / `.nii.gz` / `.nrrd`, una serie DICOM, cortes sueltos "
                                "(`.mha`, `.nii`, PNG, JPG) o un `.zip` con ellos. Cualquier número de cortes y tamaño.")
                    with gr.Row():
                        dz = gr.Number(value=1.0, label="mm entre cortes", minimum=0.1, info="Solo si no vienen en el archivo")
                        dxy = gr.Number(value=0.8, label="mm por píxel", minimum=0.1, info="Solo si no vienen en el archivo")
                    b_live = gr.Button(f"Ejecutar modelo ({device.type.upper()})")
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("① Volumen crudo"):
                        gr.Markdown("Umbral de hueso en HU sobre el CT sin procesar, proyectado mientras gira 360°. **No interviene el modelo.**")
                        mip = gr.Image(type="filepath", show_label=False, height=540, interactive=False)
                        with gr.Accordion("Ajustes del umbral", open=False):
                            thr = gr.Slider(100, 700, step=25, value=pc.HU_BONE_THR, label="Umbral de hueso [HU]",
                                            info="150–250: hueso completo · 300–400: solo cortical, más limpio")
                            limpiar = gr.Checkbox(value=True, label="Quitar camilla y fantoma")
                            b_mip = gr.Button("Regenerar animación", size="sm")
                    with gr.Tab("② Corte a corte"):
                        zs = gr.Slider(0, 1, step=1, value=0, label="Corte axial z")
                        img = gr.Image(type="pil", show_label=False, height=580, interactive=False)
                        with gr.Accordion("Ventana de visualización", open=False):
                            wn = gr.Radio(list(pc.VENTANAS), value=list(pc.VENTANAS)[0], show_label=False,
                                          info="Solo cambia cómo se ve; la red siempre recibe la ventana ósea")
                            with gr.Row():
                                lvl = gr.Slider(-500, 1300, step=10, value=400, label="Nivel [HU]", visible=False)
                                wid = gr.Slider(50, 1800, step=10, value=1800, label="Ancho [HU]", visible=False)
                    with gr.Tab("③ Reconstrucción 3D"):
                        gr.Markdown("Arrastre para girar, rueda para acercar. Cada etiqueta indica el hueso y la distancia al fragmento principal.")
                        mesh = gr.Plot(show_label=False)
                    with gr.Tab("④ Fragmentos y métricas"):
                        frag = gr.Dataframe(label="Fragmentos de este paciente (predicción vs GT)", wrap=True)
                        gr.Markdown("#### Objetivos del proyecto (conjunto de test)")
                        gr.Dataframe(metas_df(), wrap=True)
                        lat = gr.Markdown()
        outs = [state, zs, img, mesh, frag, info, lat]
        vis = [wn, lvl, wid]
        mip_in = [state, thr, limpiar]
        dd.change(on_load, [dd, show_gt] + vis, outs).then(on_mip, mip_in, mip)
        b_live.click(on_live, [dd, up, dz, dxy, show_gt] + vis, outs).then(on_mip, mip_in, mip)
        for ctl in (zs, show_gt, wn, lvl, wid):
            ctl.change(on_slice, [state, zs, show_gt] + vis, img)
        wn.change(on_win, wn, [lvl, wid])
        b_mip.click(on_mip, mip_in, mip)
        if case_ids: demo.load(on_load, [dd, show_gt] + vis, outs).then(on_mip, mip_in, mip)
    return demo


if __name__ == "__main__":
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", default=env("PENGWIN_EXPORT_DIR", "data/dashboard"))
    ap.add_argument("--ckpt", default=env("PENGWIN_CKPT", "weights/s10_best.pth"))
    ap.add_argument("--raw_dir", default=env("PENGWIN_RAW_DIR", "data/images"))
    ap.add_argument("--device", default=env("PENGWIN_DEVICE") or None)
    ap.add_argument("--port", type=int, default=int(env("PORT", "7860")))
    a = ap.parse_args()
    if not Path(a.export_dir, "config.json").exists():
        raise SystemExit(f"No se encontró {a.export_dir}/config.json: copien la carpeta dashboard/ de Drive (ver data/README.md).")
    print(f"PENGWIN dashboard · export={a.export_dir} · ckpt={a.ckpt} · raw={a.raw_dir} · puerto={a.port}", flush=True)
    launch(build(a.export_dir, a.ckpt, a.device, a.raw_dir), a.port)