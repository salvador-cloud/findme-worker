import os
import io
import time
import zipfile
import mimetypes
from uuid import uuid4
from typing import List, Dict, Any

import numpy as np
import cv2
from sklearn.cluster import DBSCAN

from fastapi import FastAPI
from supabase import create_client
from insightface.app import FaceAnalysis

# ----------------------------------
# Config
# ----------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET_UPLOADS", "uploads")

POLL_INTERVAL = 5  # seconds

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="findme-worker")

# ----------------------------------
# Face model (loaded ONCE)
# ----------------------------------
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ----------------------------------
# Helpers
# ----------------------------------
def is_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

def public_url(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

# ----------------------------------
# Core worker loop
# ----------------------------------
def process_album(album: dict):
    album_id = album["id"]
    upload_key = album["upload_key"]

    sb.table("albums").update({
        "status": "processing",
        "progress": 5,
        "error_message": None
    }).eq("id", album_id).execute()

    zip_bytes = sb.storage.from_(BUCKET).download(upload_key)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

    members = [m for m in zf.namelist() if is_image(m)]
    if not members:
        raise RuntimeError("ZIP has no images")

    face_rows: List[Dict[str, Any]] = []

    for idx, name in enumerate(members):
        data = zf.read(name)
        ext = os.path.splitext(name)[1].lower()

        path = f"albums/{album_id}/photos/{uuid4().hex}{ext}"
        sb.storage.from_(BUCKET).upload(
            path=path,
            file=data,
            file_options={"content-type": mimetypes.guess_type(name)[0] or "image/jpeg"}
        )

        photo = sb.table("photos").insert({
            "album_id": album_id,
            "storage_path": path
        }).execute().data[0]

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        faces = face_app.get(img)
        for f in faces:
            emb = f.embedding
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            face_rows.append({
                "photo_id": photo["id"],
                "embedding": emb.astype(np.float32),
                "url": public_url(path)
            })

        sb.table("albums").update({
            "progress": int(10 + (idx / len(members)) * 60)
        }).eq("id", album_id).execute()

    if not face_rows:
        sb.table("albums").update({
            "status": "completed",
            "progress": 100
        }).eq("id", album_id).execute()
        return

    X = np.stack([r["embedding"] for r in face_rows])
    labels = DBSCAN(eps=0.35, min_samples=1, metric="cosine").fit(X).labels_

    cluster_map = {}
    for label in set(labels):
        thumb = next(r["url"] for i, r in enumerate(face_rows) if labels[i] == label)
        cluster = sb.table("face_clusters").insert({
            "album_id": album_id,
            "thumbnail_url": thumb
        }).execute().data[0]
        cluster_map[label] = cluster["id"]

    links = []
    for i, r in enumerate(face_rows):
        links.append({
            "photo_id": r["photo_id"],
            "cluster_id": cluster_map[labels[i]]
        })

    sb.table("photo_faces").insert(links).execute()

    sb.table("albums").update({
        "status": "completed",
        "progress": 100,
        "photo_count": len(members)
    }).eq("id", album_id).execute()

# ----------------------------------
# Polling loop
# ----------------------------------
@app.on_event("startup")
def start_polling():
    while True:
        try:
            res = (
                sb.table("albums")
                .select("*")
                .eq("status", "pending")
                .limit(1)
                .execute()
            )
            if res.data:
                process_album(res.data[0])
        except Exception as e:
            print("Worker error:", e)
        time.sleep(POLL_INTERVAL)
