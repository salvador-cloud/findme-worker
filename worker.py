import os
import time
import json
import logging
from typing import List, Optional

import numpy as np
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
# Env
# -----------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "10"))

# Para DBSCAN
DBSCAN_EPS = float(os.environ.get("DBSCAN_EPS", "0.35"))
DBSCAN_MIN_SAMPLES = int(os.environ.get("DBSCAN_MIN_SAMPLES", "2"))


# -----------------------
# Helpers
# -----------------------
def require_env(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")


def parse_embeddings(rows: List[dict], field_name: str = "embedding") -> np.ndarray:
    """
    Espera embeddings guardados como:
    - list[float]
    - string JSON "[...]" (por ejemplo guardado como text)
    """
    emb_list = []
    for r in rows:
        v = r.get(field_name)
        if v is None:
            continue
        if isinstance(v, list):
            emb_list.append(v)
        elif isinstance(v, str):
            emb_list.append(json.loads(v))
        else:
            # fallback: intentar convertir
            emb_list.append(list(v))
    return np.array(emb_list, dtype=np.float32)


def run_dbscan(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=np.int32)
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    labels = clustering.fit_predict(embeddings)
    return labels.astype(np.int32)


# -----------------------
# Core worker
# -----------------------
class FindMeWorker:
    def __init__(self) -> None:
        require_env("SUPABASE_URL", SUPABASE_URL)
        require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)

        self.sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

        # InsightFace model init (CPU)
        # name="buffalo_l" es el default clásico para FaceAnalysis
        # det_size: ajuste típico, podés tunear performance vs precisión
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("Worker initialized. Poll every %ss | DBSCAN eps=%s min_samples=%s",
                    POLL_SECONDS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

    def fetch_pending_jobs(self) -> List[dict]:
        """
        Ajustá a tu esquema real de tabla/estado.
        En tu MVP asumimos una tabla 'jobs' con status='pending'
        """
        resp = (
            self.sb.table("jobs")
            .select("*")
            .eq("status", "pending")
            .limit(10)
            .execute()
        )
        return resp.data or []

    def mark_job(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        payload = {"status": status}
        if error:
            payload["error"] = error[:1000]
        self.sb.table("jobs").update(payload).eq("id", job_id).execute()

    def process_job(self, job: dict) -> None:
        """
        OJO: esta parte depende de tu esquema real:
        - de dónde vienen las imágenes
        - cómo guardás embeddings
        - cómo escribís clusters
        """
        job_id = str(job.get("id"))
        logger.info("Processing job id=%s", job_id)
        self.mark_job(job_id, "processing")

        try:
            # Ejemplo: leer faces/embeddings ya calculados en otra tabla
            # Si tu pipeline aún no tiene embeddings, este worker debería:
            # 1) bajar imagen
            # 2) detectar caras con self.face_app.get(...)
            # 3) extraer embedding
            # 4) guardar embedding
            # 5) clusterizar
            #
            # Para no inventar endpoints/storage aquí, dejamos el ejemplo con embeddings existentes.
            faces_resp = (
                self.sb.table("photo_faces")
                .select("id, embedding")
                .eq("job_id", job_id)
                .execute()
            )
            faces = faces_resp.data or []
            if not faces:
                raise RuntimeError("No faces/embeddings found for this job")

            embeddings = parse_embeddings(faces, field_name="embedding")
            labels = run_dbscan(embeddings)

            # Guardar label por face
            for face, label in zip(faces, labels):
                self.sb.table("photo_faces").update(
                    {"cluster_id": int(label)}
                ).eq("id", face["id"]).execute()

            self.mark_job(job_id, "done")
            logger.info("Job done id=%s | faces=%s", job_id, len(faces))

        except Exception as e:
            logger.exception("Job failed id=%s", job_id)
            self.mark_job(job_id, "error", error=str(e))

    def run_forever(self) -> None:
        while True:
            try:
                jobs = self.fetch_pending_jobs()
                if jobs:
                    logger.info("Pending jobs: %s", len(jobs))
                for job in jobs:
                    self.process_job(job)
            except Exception:
                logger.exception("Loop error (will continue)")

            time.sleep(POLL_SECONDS)


def main() -> None:
    w = FindMeWorker()
    w.run_forever()


if __name__ == "__main__":
    main()
