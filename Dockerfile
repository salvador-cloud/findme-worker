FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fly/Firecracker: onnxruntime wheel puede venir marcado con execstack requerido.
# Limpiamos ese flag en el .so para evitar el crash: "cannot enable executable stack".
RUN python - <<'PY'\n\
import site, os, glob\n\
paths = []\n\
for p in site.getsitepackages():\n\
    paths.extend(glob.glob(os.path.join(p, 'onnxruntime', 'capi', 'onnxruntime_pybind11_state*.so')))\n\
print('\\n'.join(paths))\n\
if not paths:\n\
    raise SystemExit('onnxruntime .so not found to patch')\n\
PY\n\
&& for f in $(python - <<'PY'\n\
import site, os, glob\n\
paths=[]\n\
for p in site.getsitepackages():\n\
    paths += glob.glob(os.path.join(p,'onnxruntime','capi','onnxruntime_pybind11_state*.so'))\n\
print(' '.join(paths))\n\
PY\n\
); do patchelf --clear-execstack "$f"; done

COPY . .

CMD ["python", "-u", "worker.py"]


