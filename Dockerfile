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

# Patch execstack flag in onnxruntime shared library (Fly/Firecracker constraint)
RUN python -c "import site,os,glob; \
paths=[]; \
[paths.extend(glob.glob(os.path.join(p,'onnxruntime','capi','onnxruntime_pybind11_state*.so'))) for p in site.getsitepackages()]; \
print('\\n'.join(paths)) if paths else None; \
exit(0 if paths else 1)" \
 && for f in $(python -c "import site,os,glob; \
paths=[]; \
[paths.extend(glob.glob(os.path.join(p,'onnxruntime','capi','onnxruntime_pybind11_state*.so'))) for p in site.getsitepackages()]; \
print(' '.join(paths))"); do \
      patchelf --clear-execstack "$f"; \
    done

COPY . .

CMD ["python", "-u", "worker.py"]
