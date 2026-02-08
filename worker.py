import os
import json
import time
import logging
import io
import zipfile
import random
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Callable, TypeVar

import numpy as np
import requests
import cv2

import redis
from rq import Worker, Queue, Connection
from rq.job import Job

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

# Redis / RQ
REDIS_URL = os.environ.get("REDIS_URL", "").strip()
RQ_QUEUE_NAME = os.environ.get("RQ_QUEUE_NAME", "findme").strip()

# RQ reliability knobs
RQ_JOB_TIMEOUT_SECONDS = int(os.environ.get("RQ_JOB_TIMEOUT_SECONDS", "1800"))  # 30m
RQ_RESULT_TTL_SECONDS = int(os.environ.get("RQ_RESULT_TTL_SECONDS", "3600"))   # 1h
RQ_FAILURE_TTL_SECONDS = int(os.environ.get("RQ_FAILURE_TTL_SECONDS", "86400")) # 1d

# DBSCAN tuning
DBSCAN_EPS = float(os.environ.get("DBSCAN_EPS", "0.35"))
DBSCAN_MIN_SAMPLES = int(os.environ.get("DBSCAN_MIN_SAMPLES", "2"))

# Storage
SUPABASE_PUBLIC_BUCKET = os.environ.get("SUPABASE_PUBLIC_BUCKET", "uploads")
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HTTP_TIMEOUT_SECONDS", "30"))

# Face thumbnails tuning
FACE_THUMB_SIZE = int(os.environ.get("FACE_THUMB_SIZE", "320"))  # max side length
FACE_PAD_RATIO = float(os.environ.get("FACE_PAD_RATIO", "0.30"))  # bbox padding
THUMB_JPEG_QUALITY = int(os.environ.get("THUMB_JPEG_QUALITY", "85"))  # cost control

# Merge tuning (post DBSCAN)
MERGE_CENTROID_COS = float(os.environ.get("MERGE_CENTROID_COS", "0.60"))
MERGE_BESTPAIR_COS = float(os.environ.get("MERGE_BESTPAIR_COS", "0.70"))
MAX_FACES_PER_CLUSTER_FOR_PAIRWISE = int(os.environ.get("MAX_FACES_PER_CLUSTER_FOR_PAIRWISE", "40"))

# Guards / retention
MAX_PHOTOS_PER_ALBUM = int(os.environ.get("MAX_PHOTOS_PER_ALBUM", "500"))
ALBUM_RETENTION_DAYS = int(os.environ.get("ALBUM_RETENTION_DAYS", "7"))
CLEANUP_EVERY_SECONDS = int(os.environ.get("CLEANUP_EVERY_SECONDS", "3600"))  # 1h

# Requeue stale processing jobs (still useful even with RQ)
JOB_STALE_MINUTES = int(os.environ.get("JOB_STALE_MINUTES", "30"))

# Cost controls
DELETE_ZIP_AFTER_INGEST = os.environ.get("DELETE_ZIP_AFTER_INGEST", "0").strip() in ("1", "true", "True", "yes", "YES")


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
    albums.progress es INTEGER (0..100).
    - si progress viene en 0..1 => lo pasamos a 0..100
    - si progress viene en 0..100 => lo dejamos
    """
    if progress is None:
        return None
    try:
        p = float(progress)
    except Exception:
        return None

    if p <= 1.0:
        p = p * 100.0

    return clamp_int(int(round(p)), 0, 100)


def to_public_storage_url(storage_path: str) -> str:
    base = SUPABASE_URL.rstrip("/")
    path = storage_path.lstrip("/")
    return f"{base}/storage/v1/object/public/{SUPABASE_PUBLIC_BUCKET}/{path}"


# --- retry wrapper for flaky network / occasional Supabase disconnects ---
T = TypeVar("T")


def with_retry(fn: Callable[[], T], *, tries: int = 4, base_sleep: float = 0.35) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0.0, 0.2)
            logger.warning("Retryable error (attempt %s/%s): %s", attempt + 1, tries, str(e))
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


# --- requests session ---
def _requests_session() -> requests.Session:
    return requests.Session()


def download_image(session: requests.Session, url: str) -> np.ndarray:
    r = session.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("OpenCV could not decode image")
    return img


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


def crop_face_thumb(img_bgr: np.ndarray, bbox: List[float]) -> bytes:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    pad_x = bw * FACE_PAD_RATIO
    pad_y = bh * FACE_PAD_RATIO

    xx1 = int(max(0, np.floor(x1 - pad_x)))
    yy1 = int(max(0, np.floor(y1 - pad_y)))
    xx2 = int(min(w, np.ceil(x2 + pad_x)))
    yy2 = int(min(h, np.ceil(y2 + pad_y)))

    if xx2 <= xx1 or yy2 <= yy1:
        raise RuntimeError("Invalid crop bbox")

    crop = img_bgr[yy1:yy2, xx1:xx2].copy()

    ch, cw = crop.shape[:2]
    mx = max(ch, cw)
    if mx > FACE_THUMB_SIZE:
        scale = FACE_THUMB_SIZE / float(mx)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, jpg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode face thumbnail")
    return jpg.tobytes()


# -----------------------
# Merge helpers (post-DBSCAN)
# -----------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def build_clusters_from_labels(faces_rows: List[Dict[str, Any]], labels: np.ndarray) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
        lab_int = int(lab)
        if lab_int == -1:
            continue
        clusters.setdefault(lab_int, []).append(idx)
    return clusters


def compute_centroid(embs: np.ndarray, idxs: List[int]) -> np.ndarray:
    c = np.mean(embs[idxs, :], axis=0)
    return l2_normalize(c)


def best_pair_cosine(embs: np.ndarray, idxs_a: List[int], idxs_b: List[int]) -> float:
    a = idxs_a[:MAX_FACES_PER_CLUSTER_FOR_PAIRWISE]
    b = idxs_b[:MAX_FACES_PER_CLUSTER_FOR_PAIRWISE]
    best = -1.0
    for i in a:
        vi = l2_normalize(embs[i])
        for j in b:
            vj = l2_normalize(embs[j])
            s = float(np.dot(vi, vj))
            if s > best:
                best = s
    return best


def cluster_photo_set(faces_rows: List[Dict[str, Any]], idxs: List[int]) -> set:
    return {str(faces_rows[i]["photo_id"]) for i in idxs}


def merge_clusters(
    faces_rows: List[Dict[str, Any]],
    embs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    clusters = build_clusters_from_labels(faces_rows, labels)
    if len(clusters) <= 1:
        return labels

    centroids: Dict[int, np.ndarray] = {}
    photosets: Dict[int, set] = {}
    for lab, idxs in clusters.items():
        centroids[lab] = compute_centroid(embs, idxs)
        photosets[lab] = cluster_photo_set(faces_rows, idxs)

    parent = {lab: lab for lab in clusters.keys()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    labs = sorted(clusters.keys())
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]

            if len(photosets[a].intersection(photosets[b])) > 0:
                continue

            c_sim = cosine_sim(centroids[a], centroids[b])
            if c_sim < MERGE_CENTROID_COS:
                continue

            bp = best_pair_cosine(embs, clusters[a], clusters[b])
            if bp < MERGE_BESTPAIR_COS:
                continue

            union(a, b)

    root_to_new: Dict[int, int] = {}
    old_to_new: Dict[int, int] = {}
    next_id = 0
    for lab in labs:
        r = find(lab)
        if r not in root_to_new:
            root_to_new[r] = next_id
            next_id += 1
        old_to_new[lab] = root_to_new[r]

    new_labels = labels.copy()
    for idx, lab in enumerate(labels.tolist()):
        lab_int = int(lab)
        if lab_int == -1:
            continue
        new_labels[idx] = old_to_new[lab_int]

    return new_labels.astype(np.int32)


# -----------------------
# Worker core (logic intact)
# -----------------------
class FindMeWorker:
    def __init__(self) -> None:
        require_env("SUPABASE_URL", SUPABASE_URL)
        require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)

        self.sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        self._last_cleanup_ts = 0.0
        self._session = _requests_session()

        logger.info(
            "Worker initialized (RQ). queue=%s | job_timeout=%ss | bucket=%s | dbscan eps=%s min_samples=%s | merge centroid=%s bestpair=%s | max_photos=%s | retention_days=%s | stale_job_min=%s | delete_zip_after_ingest=%s | thumb_q=%s",
            RQ_QUEUE_NAME,
            RQ_JOB_TIMEOUT_SECONDS,
            SUPABASE_PUBLIC_BUCKET,
            DBSCAN_EPS,
            DBSCAN_MIN_SAMPLES,
            MERGE_CENTROID_COS,
            MERGE_BESTPAIR_COS,
            MAX_PHOTOS_PER_ALBUM,
            ALBUM_RETENTION_DAYS,
            JOB_STALE_MINUTES,
            DELETE_ZIP_AFTER_INGEST,
            THUMB_JPEG_QUALITY,
        )

    # -----------------------
    # DB ops
    # -----------------------
    def claim_job(self, job_id: str) -> bool:
        now = utc_now_iso()

        def _do():
            return (
                self.sb.table("jobs")
                .update({"status": "processing", "updated_at": now})
                .eq("id", job_id)
                .eq("status", "pending")
                .execute()
            )

        resp = with_retry(_do)
        data = resp.data or []
        return len(data) > 0

    def requeue_stale_processing_jobs(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=JOB_STALE_MINUTES)
        cutoff_iso = cutoff.isoformat()

        def _do():
            return (
                self.sb.table("jobs")
                .update({"status": "pending", "updated_at": utc_now_iso(), "error": "requeued_stale_processing"})
                .eq("status", "processing")
                .lt("updated_at", cutoff_iso)
                .execute()
            )

        try:
            resp = with_retry(_do)
            n = len(resp.data or [])
            if n > 0:
                logger.warning("Requeued stale jobs: %s (cutoff=%s)", n, cutoff_iso)
        except Exception as e:
            logger.warning("Failed to requeue stale jobs (non-fatal): %s", str(e))

    def fetch_job_row(self, job_id: str) -> Optional[dict]:
        resp = with_retry(
            lambda: (
                self.sb.table("jobs")
                .select("*")
                .eq("id", job_id)
                .limit(1)
                .execute()
            )
        )
        rows = resp.data or []
        return rows[0] if rows else None

    def mark_job(self, job_id: str, status: str, result: Optional[dict] = None, error: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"status": status, "updated_at": utc_now_iso()}
        if result is not None:
            payload["result"] = json.dumps(result)
        if error:
            payload["error"] = error[:1500]
        with_retry(lambda: self.sb.table("jobs").update(payload).eq("id", job_id).execute())

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
            payload["progress"] = p_int

        if error_message is not None:
            payload["error_message"] = error_message[:1500]

        if completed:
            payload["completed_at"] = utc_now_iso()

        with_retry(lambda: self.sb.table("albums").update(payload).eq("id", album_id).execute())

    def fetch_photos(self, album_id: str) -> List[dict]:
        resp = with_retry(
            lambda: (
                self.sb.table("photos")
                .select("id, storage_path")
                .eq("album_id", album_id)
                .execute()
            )
        )
        return resp.data or []

    def clear_job_embeddings(self, job_id: str) -> None:
        with_retry(lambda: self.sb.table("photo_embeddings").delete().eq("job_id", job_id).execute())

    def clear_album_face_embeddings(self, album_id: str) -> None:
        with_retry(lambda: self.sb.table("face_embeddings").delete().eq("album_id", album_id).execute())

    def clear_album_clusters(self, album_id: str) -> None:
        with_retry(lambda: self.sb.table("face_clusters").delete().eq("album_id", album_id).execute())

    def clear_photo_faces_for_album(self, album_id: str) -> None:
        photos = self.fetch_photos(album_id)
        for p in photos:
            with_retry(lambda pid=p["id"]: self.sb.table("photo_faces").delete().eq("photo_id", pid).execute())

    def create_cluster(self, album_id: str, thumbnail_url: Optional[str]) -> str:
        resp = with_retry(
            lambda: self.sb.table("face_clusters").insert({"album_id": album_id, "thumbnail_url": thumbnail_url}).execute()
        )
        data = resp.data or []
        if not data:
            raise RuntimeError("Could not create face_cluster")
        return data[0]["id"]

    def upsert_photo_face(self, photo_id: str, cluster_id: str) -> None:
        with_retry(
            lambda: self.sb.table("photo_faces").upsert(
                {"photo_id": photo_id, "cluster_id": cluster_id},
                on_conflict="photo_id,cluster_id",
            ).execute()
        )

    def insert_face_embedding_row(
        self,
        face_id: str,
        photo_id: str,
        album_id: str,
        embedding: List[float],
        bbox: List[float],
    ) -> None:
        with_retry(
            lambda: self.sb.table("face_embeddings").insert(
                {
                    "id": face_id,
                    "photo_id": photo_id,
                    "album_id": album_id,
                    "embedding": embedding,
                    "bbox": bbox,
                }
            ).execute()
        )

    def upload_bytes(self, storage_path: str, raw: bytes, content_type: str) -> None:
        def _do():
            return self.sb.storage.from_(SUPABASE_PUBLIC_BUCKET).upload(
                path=storage_path,
                file=raw,
                file_options={"content-type": content_type},
            )

        up_res = with_retry(_do)
        if getattr(up_res, "error", None):
            raise RuntimeError(f"Storage upload failed for {storage_path}: {up_res.error}")

    def remove_paths(self, paths: List[str]) -> None:
        paths = [p for p in paths if p]
        if not paths:
            return
        with_retry(lambda: self.sb.storage.from_(SUPABASE_PUBLIC_BUCKET).remove(paths))

    # -----------------------
    # ZIP ingest
    # -----------------------
    def download_zip_bytes(self, zip_path: str) -> bytes:
        url = to_public_storage_url(zip_path)
        r = self._session.get(url, timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.content

    def ingest_zip_to_photos(self, album_id: str, zip_path: str) -> int:
        zip_bytes = self.download_zip_bytes(zip_path)
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

        inserted = 0
        for name in zf.namelist():
            n = name.lower()
            if n.endswith("/") or not (n.endswith(".jpg") or n.endswith(".jpeg") or n.endswith(".png")):
                continue

            raw = zf.read(name)
            if not raw:
                continue

            if inserted >= MAX_PHOTOS_PER_ALBUM:
                raise RuntimeError(f"Album exceeds max allowed photos ({MAX_PHOTOS_PER_ALBUM})")

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

            self.upload_bytes(storage_path=storage_path, raw=raw, content_type=content_type)

            ins = with_retry(
                lambda: self.sb.table("photos").insert(
                    {"id": photo_id, "album_id": album_id, "storage_path": storage_path}
                ).execute()
            )
            if getattr(ins, "error", None):
                raise RuntimeError(f"DB insert into photos failed: {ins.error}")

            inserted += 1

        self.update_album(album_id, status="processing", progress=5)
        with_retry(lambda: self.sb.table("albums").update({"photo_count": inserted}).eq("id", album_id).execute())
        return inserted

    # -----------------------
    # Retention cleanup
    # -----------------------
    def delete_album_everywhere(self, album_id: str) -> None:
        album_id = str(album_id)
        logger.info("Deleting album everywhere album_id=%s", album_id)

        photos = (
            with_retry(lambda: self.sb.table("photos").select("id,storage_path").eq("album_id", album_id).execute())
        ).data or []

        photo_paths = [p.get("storage_path") for p in photos if p.get("storage_path")]
        self.remove_paths(photo_paths)

        faces = (
            with_retry(lambda: self.sb.table("face_embeddings").select("id").eq("album_id", album_id).execute())
        ).data or []
        face_thumb_paths = [f"albums/{album_id}/faces/{r['id']}.jpg" for r in faces if r.get("id")]
        self.remove_paths(face_thumb_paths)

        alb = with_retry(lambda: self.sb.table("albums").select("upload_key").eq("id", album_id).execute())
        if not getattr(alb, "error", None) and alb.data and alb.data[0].get("upload_key"):
            self.remove_paths([alb.data[0]["upload_key"]])

        photo_ids = [p.get("id") for p in photos if p.get("id")]
        for pid in photo_ids:
            with_retry(lambda _pid=pid: self.sb.table("photo_faces").delete().eq("photo_id", _pid).execute())

        with_retry(lambda: self.sb.table("face_embeddings").delete().eq("album_id", album_id).execute())
        with_retry(lambda: self.sb.table("face_clusters").delete().eq("album_id", album_id).execute())
        with_retry(lambda: self.sb.table("photos").delete().eq("album_id", album_id).execute())
        with_retry(lambda: self.sb.table("jobs").delete().eq("album_id", album_id).execute())
        with_retry(lambda: self.sb.table("albums").delete().eq("id", album_id).execute())

    def cleanup_old_albums(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=ALBUM_RETENTION_DAYS)
        resp = with_retry(
            lambda: (
                self.sb.table("albums")
                .select("id,status,created_at")
                .lt("created_at", cutoff.isoformat())
                .in_("status", ["completed", "error"])
                .limit(200)
                .execute()
            )
        )
        rows = resp.data or []
        if not rows:
            logger.info("Cleanup: no old albums to delete (cutoff=%s)", cutoff.isoformat())
            return

        logger.info("Cleanup: deleting %s old albums (cutoff=%s)", len(rows), cutoff.isoformat())
        for r in rows:
            try:
                self.delete_album_everywhere(r["id"])
            except Exception as e:
                logger.warning("Cleanup failed album_id=%s err=%s", r.get("id"), str(e))

    # -----------------------
    # Core processing (unchanged)
    # -----------------------
    def process_job(self, job: dict) -> None:
        job_id = str(job.get("id"))
        album_id = job.get("album_id")

        logger.info("Processing job %s", job_id)

        if not album_id:
            self.mark_job(job_id, "error", error="Missing album_id in jobs row")
            return

        album_id = str(album_id)

        self.update_album(album_id, status="processing", progress=1, error_message=None, completed=False)

        try:
            photos = self.fetch_photos(album_id)

            zip_path = str(job.get("zip_path") or "")
            if not photos:
                if not zip_path:
                    raise RuntimeError("No photos found and missing jobs.zip_path to ingest")

                logger.info("No photos rows for album %s. Ingesting ZIP zip_path=%s", album_id, zip_path)
                self.update_album(album_id, status="processing", progress=2)

                added = self.ingest_zip_to_photos(album_id=album_id, zip_path=zip_path)
                logger.info("ZIP ingest done. photos_added=%s album_id=%s", added, album_id)

                photos = self.fetch_photos(album_id)
                if not photos:
                    raise RuntimeError("ZIP ingest ran but photos table is still empty for this album")

            total_photos = len(photos)
            if total_photos > MAX_PHOTOS_PER_ALBUM:
                raise RuntimeError(f"Album exceeds max allowed photos ({MAX_PHOTOS_PER_ALBUM})")

            logger.info("Album %s photos=%s", album_id, total_photos)

            self.clear_job_embeddings(job_id)
            self.clear_photo_faces_for_album(album_id)
            self.clear_album_clusters(album_id)
            self.clear_album_face_embeddings(album_id)

            faces_rows: List[Dict[str, Any]] = []
            face_thumb_url_by_face_id: Dict[str, str] = {}

            photos_with_faces = 0
            photos_no_face = 0
            faces_total = 0

            for idx, p in enumerate(photos):
                photo_id = str(p["id"])
                storage_path = str(p["storage_path"])
                url = to_public_storage_url(storage_path)

                try:
                    img = download_image(self._session, url)
                    faces = self.face_app.get(img) or []

                    faces = [
                        f for f in faces
                        if getattr(f, "embedding", None) is not None and getattr(f, "bbox", None) is not None
                    ]

                    if not faces:
                        photos_no_face += 1
                        continue

                    photos_with_faces += 1

                    for f in faces:
                        face_id = str(uuid4())
                        emb = f.embedding.astype(np.float32).tolist()
                        bbox = [float(x) for x in f.bbox.tolist()]

                        thumb_bytes = crop_face_thumb(img, bbox)
                        thumb_path = f"albums/{album_id}/faces/{face_id}.jpg"
                        self.upload_bytes(storage_path=thumb_path, raw=thumb_bytes, content_type="image/jpeg")
                        thumb_url = to_public_storage_url(thumb_path)
                        face_thumb_url_by_face_id[face_id] = thumb_url

                        self.insert_face_embedding_row(
                            face_id=face_id,
                            photo_id=photo_id,
                            album_id=album_id,
                            embedding=emb,
                            bbox=bbox,
                        )

                        faces_rows.append({"face_id": face_id, "photo_id": photo_id, "embedding": emb})
                        faces_total += 1

                except Exception as e:
                    logger.warning("Photo failed id=%s path=%s err=%s", photo_id, storage_path, str(e))

                prog_ratio = (idx + 1) / max(1, total_photos)
                prog_int = clamp_int(int(round(5 + prog_ratio * 55)), 5, 60)
                self.update_album(album_id, status="processing", progress=prog_int)

            if faces_total == 0:
                raise RuntimeError(f"No faces detected/embedded. photos={total_photos}, photos_no_face={photos_no_face}")

            embeddings = np.array([r["embedding"] for r in faces_rows], dtype=np.float32)
            labels = run_dbscan(embeddings)
            labels = merge_clusters(faces_rows, embeddings, labels)

            self.update_album(album_id, status="processing", progress=70)

            cluster_map: Dict[int, str] = {}
            unique_labels = sorted({int(x) for x in labels.tolist() if int(x) != -1})

            for lab in unique_labels:
                rep_face_id = None
                for r, l in zip(faces_rows, labels):
                    if int(l) == lab:
                        rep_face_id = r["face_id"]
                        break
                thumb_url = face_thumb_url_by_face_id.get(rep_face_id) if rep_face_id else None
                cluster_id = self.create_cluster(album_id=album_id, thumbnail_url=thumb_url)
                cluster_map[lab] = cluster_id

            assigned_faces = 0
            noise_faces = 0

            for i, (r, lab) in enumerate(zip(faces_rows, labels)):
                photo_id = str(r["photo_id"])
                lab_int = int(lab)

                if lab_int == -1:
                    noise_faces += 1
                    thumb_url = face_thumb_url_by_face_id.get(str(r["face_id"]))
                    cluster_id = self.create_cluster(album_id=album_id, thumbnail_url=thumb_url)
                    self.upsert_photo_face(photo_id=photo_id, cluster_id=cluster_id)
                else:
                    cluster_id = cluster_map[lab_int]
                    self.upsert_photo_face(photo_id=photo_id, cluster_id=cluster_id)

                assigned_faces += 1
                prog_ratio = (i + 1) / max(1, len(faces_rows))
                prog_int = clamp_int(int(round(70 + prog_ratio * 25)), 70, 95)
                self.update_album(album_id, status="processing", progress=prog_int)

            if DELETE_ZIP_AFTER_INGEST and zip_path:
                try:
                    self.remove_paths([zip_path])
                    logger.info("Deleted original ZIP after ingest: %s", zip_path)
                except Exception as e:
                    logger.warning("Failed to delete original ZIP (non-fatal): %s", str(e))

            result = {
                "album_id": album_id,
                "photos_total": total_photos,
                "photos_with_faces": photos_with_faces,
                "photos_no_face": photos_no_face,
                "faces_total": faces_total,
                "faces_assigned": assigned_faces,
                "faces_noise": noise_faces,
                "clusters": len(unique_labels) + noise_faces,
                "dbscan": {"eps": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES},
                "merge": {
                    "centroid_cos": MERGE_CENTROID_COS,
                    "bestpair_cos": MERGE_BESTPAIR_COS,
                    "max_pairwise": MAX_FACES_PER_CLUSTER_FOR_PAIRWISE,
                },
                "limits": {
                    "max_photos_per_album": MAX_PHOTOS_PER_ALBUM,
                    "retention_days": ALBUM_RETENTION_DAYS,
                },
                "cost_controls": {
                    "delete_zip_after_ingest": DELETE_ZIP_AFTER_INGEST,
                    "thumb_jpeg_quality": THUMB_JPEG_QUALITY,
                    "face_thumb_size": FACE_THUMB_SIZE,
                },
            }

            self.update_album(album_id, status="completed", progress=100, completed=True)
            self.mark_job(job_id, "done", result=result)
            logger.info("Job done %s", job_id)

        except Exception as e:
            logger.exception("Job failed %s", job_id)
            try:
                self.update_album(album_id, status="error", progress=0, error_message=str(e), completed=False)
            except Exception as ee:
                logger.warning("Failed to update album error state (non-fatal): %s", str(ee))
            try:
                self.mark_job(job_id, "error", error=str(e))
            except Exception as ee:
                logger.warning("Failed to mark job error state (non-fatal): %s", str(ee))


# -----------------------
# RQ entrypoint: consume job_id from Redis, then process via Supabase
# -----------------------
_worker_singleton: Optional[FindMeWorker] = None


def _get_worker() -> FindMeWorker:
    global _worker_singleton
    if _worker_singleton is None:
        _worker_singleton = FindMeWorker()
    return _worker_singleton


def process_job_id(job_id: str) -> None:
    """
    This is the function executed by RQ.
    It receives a job_id (string) from Redis, loads job row from DB, claims it, and processes it.
    """
    w = _get_worker()

    # periodic cleanup + stale requeue (cheap, keeps system healthy)
    now_ts = time.time()
    if not hasattr(w, "_last_cleanup_ts"):
        w._last_cleanup_ts = 0.0  # type: ignore
    if now_ts - w._last_cleanup_ts >= CLEANUP_EVERY_SECONDS:  # type: ignore
        w._last_cleanup_ts = now_ts  # type: ignore
        try:
            w.cleanup_old_albums()
        except Exception:
            logger.exception("Cleanup loop error (will continue)")
    w.requeue_stale_processing_jobs()

    job_id = str(job_id or "").strip()
    if not job_id:
        return

    job_row = w.fetch_job_row(job_id)
    if not job_row:
        logger.warning("Job id not found in DB: %s", job_id)
        return

    # Claim lock: only one worker proceeds (horizontal safe)
    if not w.claim_job(job_id):
        logger.info("Job already claimed/processed by another worker: %s", job_id)
        return

    w.process_job(job_row)


def main() -> None:
    require_env("REDIS_URL", REDIS_URL)
    require_env("SUPABASE_URL", SUPABASE_URL)
    require_env("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)

    redis_conn = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=10,
        socket_connect_timeout=10,
        health_check_interval=30,
    )

    with Connection(redis_conn):
        q = Queue(
            RQ_QUEUE_NAME,
            default_timeout=RQ_JOB_TIMEOUT_SECONDS,
        )

        logger.info("RQ worker boot. queue=%s redis=%s", RQ_QUEUE_NAME, "configured")

        w = Worker(
            [q],
            connection=redis_conn,
            default_worker_ttl=420,
            job_monitoring_interval=30,
            log_job_description=False,
        )
        w.work(
            with_scheduler=False,
            logging_level=logging.INFO,
        )


if __name__ == "__main__":
    main()
