import os
import time
import json
import logging
from typing import List, Optional

import numpy as np
import requests
import cv2

from supabase import create_client, Client
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN


# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("findme-worker")


# -----------------------
# ENV
# -----------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "10"))

DBSCAN_EPS = float(os.environ.get("DBSCAN_EPS", "0.35"))
DBSCAN_MIN_SAMPLES = int(os.environ.get("DBSCAN_MIN_SAMPLES", "2"))


def require_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Missing env var: {name}")


def normalize_embedding(emb: np.ndarray):
    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def run_dbscan(embeddings: np.ndarray):
    if embeddings.size == 0:
        return np.array([], dtype=np.int32)
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    return clustering.fit_predict(embeddings).astype(np.int32)


class FindMeWorker:

    def __init__(self):
        require_env("SUPABASE_URL", SUPABASE_URL)
        require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)

        self.sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("Worker ready")

    def fetch_pending_jobs(self):
        resp = (
            self.sb.table("jobs")
            .select("*")
            .eq("status", "pending")
            .limit(5)
            .execute()
        )
        return resp.data or []

    def mark_job(self, job_id, status, error=None, result=None):
        payload = {"status": status}
        if error:
            payload["error"] = error[:2000]
        if result:
            payload["result"] = result

        self.sb.table("jobs").update(payload).eq("id", job_id).execute()

    def fetch_photos(self, album_id):
        resp = (
            self.sb.table("photos")
            .select("id,url")
            .eq("album_id", album_id)
            .execute()
        )
        return resp.data or []

    def download_image(self, url):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Image decode failed")
        return img

    def process_job(self, job):

        job_id = job["id"]
        album_id = job["album_id"]

        logger.info("Processing job %s", job_id)
        self.mark_job(job_id, "processing")

        try:
            photos = self.fetch_photos(album_id)

            if not photos:
                raise RuntimeError("No photos found")

            # ------------------------
            # 1) Generate embeddings
            # ------------------------
            for p in photos:

                photo_id = p["id"]
                url = p["url"]

                img = self.download_image(url)
                faces = self.face_app.get(img)

                if not faces:
                    continue

                best_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )

                emb = normalize_embedding(best_face.embedding)

                self.sb.table("photo_embeddings").upsert(
                    {
                        "photo_id": photo_id,
                        "job_id": job_id,
                        "embedding": emb,
                    },
                    on_conflict="photo_id,job_id",
                ).execute()

            # ------------------------
            # 2) Fetch embeddings
            # ------------------------
            resp = (
                self.sb.table("photo_embeddings")
                .select("photo_id,embedding")
                .eq("job_id", job_id)
                .execute()
            )

            rows = resp.data or []

            if not rows:
                raise RuntimeError("No embeddings generated")

            embeddings = np.array(
                [json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
                 for r in rows],
                dtype=np.float32,
            )

            labels = run_dbscan(embeddings)

            # ------------------------
            # 3) Update clusters
            # ------------------------
            for row, label in zip(rows, labels):

                self.sb.table("photo_faces").upsert(
                    {
                        "photo_id": row["photo_id"],
                        "cluster_id": int(label),
                    },
                    on_conflict="photo_id",
                ).execute()

            summary = {
                "photos": len(rows),
                "clusters": len(set(labels.tolist())),
            }

            self.mark_job(job_id, "done", result=summary)
            logger.info("Job done %s", summary)

        except Exception as e:
            logger.exception("Job failed")
            self.mark_job(job_id, "error", error=str(e))

    def run(self):
        while True:
            try:
                jobs = self.fetch_pending_jobs()
                for job in jobs:
                    self.process_job(job)
            except Exception:
                logger.exception("Loop error")
            time.sleep(POLL_SECONDS)


def main():
    w = FindMeWorker()
    w.run()


if __name__ == "__main__":
    main()
