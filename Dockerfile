# Dashboard PENGWIN — imagen de inferencia (CPU por defecto; GPU con docker-compose.gpu.yml)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False MPLCONFIGDIR=/tmp

# fuentes para las etiquetas en mm del Visualizador 2 y libgomp para PyTorch en CPU
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch: rueda CPU (~200 MB). Para GPU NVIDIA: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.6.0
RUN pip install --index-url ${TORCH_INDEX} torch==${TORCH_VERSION}

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY pengwin_core.py app.py ./

# Rutas dentro del contenedor (los datos se montan como volúmenes de solo lectura, ver docker-compose.yml)
ENV PENGWIN_EXPORT_DIR=/data/dashboard \
    PENGWIN_CKPT=/data/weights/s10_best.pth \
    PENGWIN_RAW_DIR=/data/images \
    PORT=7860

EXPOSE 7860
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=8 \
    CMD curl -fs http://localhost:7860/ > /dev/null || exit 1

# usuario sin privilegios
RUN useradd -m pengwin
USER pengwin

CMD ["python", "app.py"]
