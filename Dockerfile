FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema:
# - libgomp1: requerido por sklearn/numpy en muchos builds
# - libgl1 + libglib2.0-0: necesarios si alguna lib intenta cargar OpenCV (aunque usemos headless, prefiero blindar)
# - prelink: provee execstack para limpiar el flag "executable stack" en el .so de onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    prelink \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install -r /app/requirements.txt

# FIX CRÍTICO: onnxruntime trae un .so que pide "executable stack" y Fly lo bloquea.
# Esto limpia el flag y evita el crash al importar.
RUN python - << 'PY'
import os, site, glob, subprocess

paths = []
for sp in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(sp, "onnxruntime", "capi", "*.so")))

# En algunos layouts el .so vive en subpaths
for sp in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(sp, "onnxruntime", "**", "*.so"), recursive=True))

paths = sorted(set(paths))

if not paths:
    raise SystemExit("ERROR: No se encontraron .so de onnxruntime para aplicar execstack -c")

print("Encontrados .so de onnxruntime:")
for p in paths:
    print(" -", p)

for p in paths:
    # execstack -c limpia el bit de executable stack
    subprocess.run(["execstack", "-c", p], check=False)

print("execstack -c aplicado.")
PY

COPY . /app

CMD ["python", "-u", "worker.py"]
