import os
import time
import json
import logging
import io
import zipfile
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

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
# Env
# -----------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "10"))

# DBSCAN tuning
DBSCAN_EPS = float(os.environ.get("DBSCAN_EPS", "0.35"))
DBSCAN_MIN_SAMPLES = int(os.environ.get("DBSCAN_MIN_SAMPLES", "2"))

# Storage
SUPABASE_PUBLIC_BUCKET = os.environ.get("SUPABASE_PUBLIC_BUCKET", "uploads")  # tu bucket público: uploads
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HTTP_TIMEOUT_SECONDS", "30"))


# -----------------------
# Helpers
# -----------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_env(name: str, value: str) -> None:
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def normalize_progress_to_int(progress: Optional[float]) -> Optional[int]:
    """
    albums.progress es INTEGER.
    - si progress viene en 0..1 => lo pasamos a 0..100
    - si progress viene en 0..100 => lo dejamos
    - siempre devuelve int clamp 0..100
    """
    if progress is None:
        return None
    try:
        p = float(progress)
    except Exception:
        return None

    # Heurística: si es <= 1.0 asumimos ratio
    if p <= 1.0:
        p = p * 100.0

    return clamp_int(int(round(p)), 0, 100)


def to_public_storage_url(storage_path: str) -> str:
    """
    Como el bucket 'uploads' es público:
    https://<PROJECT>.supabase.co/storage/v1/object/public/uploads/<storage_path>
    """
    base = SUPABASE_URL.rstrip("/")
    path = storage_path.lstrip("/")
    return f"{base}/storage/v1/object/public/{SUPABASE_PUBLIC_BUCKET}/{path}"


def download_image(url: str) -> np.ndarray:
    """
    Descarga imagen y la decodifica a BGR (OpenCV).
    """
    r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("OpenCV could not decode image")
    return img


def pick_primary_face(faces: List[Any]) -> Optional[Any]:
    """
    Si hay múltiples caras, elegimos la "principal" por área de bbox.
    """
    if not faces:
        return None
    best = None
    best_area = -1.0
    for f in faces:
        # f.bbox: [x1,y1,x2,y2]
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = f
    return best


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
            emb_list.append(list(v))
    return np.array(emb_list, dtype=np.float32)


def run_dbscan(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=np.int32)
    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
    )
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

        # InsightFace init (CPU)
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info(
            "Worker initialized. Poll=%ss | DBSCAN eps=%s min_samples=%s | bucket=%s",
            POLL_SECONDS,
            DBSCAN_EPS,
            DBSCAN_MIN_SAMPLES,
            SUPABASE_PUBLIC_BUCKET,
        )

    # -----------------------
    # DB ops
    # -----------------------
    def fetch_pending_jobs(self) -> List[dict]:
        resp = (
            self.sb.table("jobs")
            .select("*")
            .eq("status", "pending")
            .limit(5)
            .execute()
        )
        return resp.data or []

    def mark_job(self, job_id: str, status: str, result: Optional[dict] = None, error: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"status": status, "updated_at": utc_now_iso()}
        if result is not None:
            payload["result"] = json.dumps(result)
        if error:
            payload["error"] = error[:1500]
        self.sb.table("jobs").update(payload).eq("id", job_id).execute()

    def update_album(
        self,
        album_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        completed: bool = False,
    ) -> None:
        payload: Dict[str, Any] = {"status": status}

        p_int = normalize_progress_to_int(progress)
        if p_int is not None:
            payload["progress"] = p_int  # INTEGER 0..100

        if error_message is not None:
            payload["error_message"] = error_message[:1500]

        if completed:
            payload["completed_at"] = utc_now_iso()

        self.sb.table("albums").update(payload).eq("id", album_id).execute()

    def fetch_photos(self, album_id: str) -> List[dict]:
        resp = (
            self.sb.table("photos")
            .select("id, storage_path")
            .eq("album_id", album_id)
            .execute()
        )
        return resp.data or []

    def clear_job_embeddings(self, job_id: str) -> None:
        # Limpiamos embeddings previos del job (idempotencia)
        self.sb.table("photo_embeddings").delete().eq("job_id", job_id).execute()

    def upsert_face_embedding(self, photo_id: str, album_id: str, embedding: List[float], bbox: List[float]) -> None:
        # En tu tabla face_embeddings NO hay id, así que hacemos delete+insert por foto+album para evitar duplicados
        self.sb.table("face_embeddings").delete().eq("photo_id", photo_id).eq("album_id", album_id).execute()
        self.sb.table("face_embeddings").insert(
            {
                "photo_id": photo_id,
                "album_id": album_id,
                "embedding": embedding,
                "bbox": bbox,
            }
        ).execute()

    def insert_photo_embedding(self, photo_id: str, job_id: str, embedding: List[float]) -> None:
        self.sb.table("photo_embeddings").insert(
            {
                "photo_id": photo_id,
                "job_id": job_id,
                "embedding": embedding,
            }
        ).execute()

    def clear_photo_faces_for_album(self, album_id: str) -> None:
        # photo_faces solo tiene photo_id, cluster_id, created_at
        # Borramos mappings para las fotos del album (simple y seguro)
        photos = self.fetch_photos(album_id)
        for p in photos:
            self.sb.table("photo_faces").delete().eq("photo_id", p["id"]).execute()

    def create_cluster(self, album_id: str, thumbnail_url: Optional[str]) -> str:
        resp = self.sb.table("face_clusters").insert(
            {
                "album_id": album_id,
                "thumbnail_url": thumbnail_url,
            }
        ).execute()
        data = resp.data or []
        if not data:
            raise RuntimeError("Could not create face_cluster")
        return data[0]["id"]

    def insert_photo_face(self, photo_id: str, cluster_id: str) -> None:
        # idempotencia: borramos y reinsertamos
        self.sb.table("photo_faces").delete().eq("photo_id", photo_id).execute()
        self.sb.table("photo_faces").insert(
            {
                "photo_id": photo_id,
                "cluster_id": cluster_id,
            }
        ).execute()

    # -----------------------
    # ZIP ingest (cuando API todavía no crea photos)
    # -----------------------
    def download_zip_bytes(self, zip_path: str) -> bytes:
        """
        Descarga el ZIP desde bucket público uploads. zip_path ejemplo: zips/xxxx.zip
        """
        url = to_public_storage_url(zip_path)
        r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.content

    def ingest_zip_to_photos(self, album_id: str, zip_path: str) -> int:
        """
        - Descomprime el ZIP en memoria
        - Sube cada imagen al bucket uploads en: albums/<album_id>/photos/<photo_id>.<ext>
        - Inserta filas en tabla photos: (id, album_id, storage_path)
        Retorna cantidad de fotos insertadas.
        """
        zip_bytes = self.download_zip_bytes(zip_path)
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

        inserted = 0
        for name in zf.namelist():
            n = name.lower()

            # saltar directorios y archivos no imagen
            if n.endswith("/") or not (n.endswith(".jpg") or n.endswith(".jpeg") or n.endswith(".png")):
                continue

            raw = zf.read(name)
            if not raw:
                continue

            if n.endswith(".png"):
                ext = "png"
                content_type = "image/png"
            elif n.endswith(".jpeg"):
                ext = "jpeg"
                content_type = "image/jpeg"
            else:
                ext = "jpg"
                content_type = "image/jpeg"

            photo_id = str(uuid4())
            storage_path = f"albums/{album_id}/photos/{photo_id}.{ext}"

            # 1) subir al bucket uploads
            up_res = self.sb.storage.from_(SUPABASE_PUBLIC_BUCKET).upload(
                path=storage_path,
                file=raw,
                file_options={"content-type": content_type},
            )
            if getattr(up_res, "error", None):
                raise RuntimeError(f"Storage upload failed for {storage_path}: {up_res.error}")

            # 2) insertar en photos
            ins = self.sb.table("photos").insert(
                {
                    "id": photo_id,
                    "album_id": album_id,
                    "storage_path": storage_path,
                }
            ).execute()
            if getattr(ins, "error", None):
                raise RuntimeError(f"DB insert into photos failed: {ins.error}")

            inserted += 1

        # actualizar photo_count en albums
        self.sb.table("albums").update({"photo_count": inserted}).eq("id", album_id).execute()
        return inserted

    # -----------------------
    # Core processing
    # -----------------------
    def process_job(self, job: dict) -> None:
        job_id = str(job.get("id"))
        album_id = job.get("album_id")

        logger.info("Processing job %s", job_id)

        if not album_id:
            self.mark_job(job_id, "error", error="Missing album_id in jobs row")
            return

        album_id = str(album_id)

        # Marcar estados
        self.mark_job(job_id, "processing")
        self.update_album(album_id, status="processing", progress=1, error_message=None, completed=False)

        try:
            photos = self.fetch_photos(album_id)

            # ✅ FIX: si API no creó photos, el worker las crea desde el ZIP
            if not photos:
                zip_path = job.get("zip_path")
                if not zip_path:
                    raise RuntimeError("No photos found and missing jobs.zip_path to ingest")

                logger.info("No photos rows for album %s. Ingesting ZIP zip_path=%s", album_id, zip_path)
                self.update_album(album_id, status="processing", progress=2)

                added = self.ingest_zip_to_photos(album_id=album_id, zip_path=str(zip_path))
                logger.info("ZIP ingest done. photos_added=%s album_id=%s", added, album_id)

                photos = self.fetch_photos(album_id)
                if not photos:
                    raise RuntimeError("ZIP ingest ran but photos table is still empty for this album")
                # progress después del ingest
                self.update_album(album_id, status="processing", progress=5)

            total = len(photos)
            logger.info("Album %s photos=%s", album_id, total)

            # 1) Generar embeddings por foto (principal)
            self.clear_job_embeddings(job_id)

            processed = 0
            no_face = 0

            for p in photos:
                photo_id = str(p["id"])
                storage_path = str(p["storage_path"])
                url = to_public_storage_url(storage_path)

                try:
                    img = download_image(url)
                    faces = self.face_app.get(img)
                    primary = pick_primary_face(faces)
                    if primary is None or getattr(primary, "embedding", None) is None:
                        no_face += 1
                        continue

                    emb = primary.embedding.astype(np.float32).tolist()
                    bbox = [float(x) for x in primary.bbox.tolist()]

                    # Guardar en tablas existentes
                    self.upsert_face_embedding(photo_id=photo_id, album_id=album_id, embedding=emb, bbox=bbox)
                    self.insert_photo_embedding(photo_id=photo_id, job_id=job_id, embedding=emb)

                    processed += 1

                except Exception as e:
                    logger.warning("Photo failed id=%s path=%s err=%s", photo_id, storage_path, str(e))

                # progreso incremental: 1..60 durante embeddings
                prog_ratio = processed / max(1, total)
                prog_int = clamp_int(int(round(5 + prog_ratio * 55)), 1, 60)
                self.update_album(album_id, status="processing", progress=prog_int)

            if processed == 0:
                raise RuntimeError(f"No embeddings generated. Photos={total}, no_face={no_face}")

            # 2) Clustering por job
            pe = (
                self.sb.table("photo_embeddings")
                .select("photo_id, embedding")
                .eq("job_id", job_id)
                .execute()
            ).data or []

            if not pe:
                raise RuntimeError("photo_embeddings empty after processing photos")

            embeddings = parse_embeddings(pe, field_name="embedding")
            labels = run_dbscan(embeddings)

            # En clustering, marcamos progreso 70 antes de escribir clusters
            self.update_album(album_id, status="processing", progress=70)

            # 3) Crear clusters + asignar photo_faces
            # Limpiamos mappings previos
            self.clear_photo_faces_for_album(album_id)

            # Mapa label -> cluster_id
            cluster_map: Dict[int, str] = {}

            # Para thumbnail: usar la primera foto de ese cluster
            # Armamos acceso rápido photo_id -> storage url
            photo_url_map: Dict[str, str] = {}
            for p in photos:
                photo_url_map[str(p["id"])] = to_public_storage_url(str(p["storage_path"]))

            # Primero labels != -1
            unique_labels = sorted({int(x) for x in labels.tolist() if int(x) != -1})
            for lab in unique_labels:
                thumb = None
                for row, l in zip(pe, labels):
                    if int(l) == lab:
                        thumb = photo_url_map.get(str(row["photo_id"]))
                        break
                cluster_id = self.create_cluster(album_id=album_id, thumbnail_url=thumb)
                cluster_map[lab] = cluster_id

            # Ahora asignación por foto
            assigned = 0
            noise = 0
            for row, lab in zip(pe, labels):
                photo_id = str(row["photo_id"])
                lab_int = int(lab)

                if lab_int == -1:
                    noise += 1
                    thumb = photo_url_map.get(photo_id)
                    cluster_id = self.create_cluster(album_id=album_id, thumbnail_url=thumb)
                    self.insert_photo_face(photo_id=photo_id, cluster_id=cluster_id)
                else:
                    cluster_id = cluster_map[lab_int]
                    self.insert_photo_face(photo_id=photo_id, cluster_id=cluster_id)

                assigned += 1

                # progreso 70..95 mientras asignamos
                prog_ratio = assigned / max(1, len(pe))
                prog_int = clamp_int(int(round(70 + prog_ratio * 25)), 70, 95)
                self.update_album(album_id, status="processing", progress=prog_int)

            # 4) Finalizar
            result = {
                "album_id": album_id,
                "photos_total": total,
                "photos_embedded": processed,
                "photos_no_face": no_face,
                "assigned": assigned,
                "noise": noise,
                "clusters": len(unique_labels) + noise,
                "dbscan": {"eps": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES},
            }

            self.update_album(album_id, status="completed", progress=100, completed=True)
            self.mark_job(job_id, "done", result=result)
            logger.info("Job done %s result=%s", job_id, result)

        except Exception as e:
            logger.exception("Job failed %s", job_id)
            self.update_album(album_id, status="error", progress=0, error_message=str(e), completed=False)
            self.mark_job(job_id, "error", error=str(e))

    def run_forever(self) -> None:
        while True:
            try:
                jobs = self.fetch_pending_jobs()
                if jobs:
                    logger.info("Pending jobs: %s", len(jobs))
                for j in jobs:
                    self.process_job(j)
            except Exception:
                logger.exception("Loop error (will continue)")

            time.sleep(POLL_SECONDS)


def main() -> None:
    w = FindMeWorker()
    w.run_forever()


if __name__ == "__main__":
    main()
