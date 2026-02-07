import os
import time
import json
import logging
import io
import zipfile
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

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
SUPABASE_PUBLIC_BUCKET = os.environ.get("SUPABASE_PUBLIC_BUCKET", "uploads")
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HTTP_TIMEOUT_SECONDS", "30"))

# Face thumbnails tuning
FACE_THUMB_SIZE = int(os.environ.get("FACE_THUMB_SIZE", "320"))  # max side length
FACE_PAD_RATIO = float(os.environ.get("FACE_PAD_RATIO", "0.30"))  # bbox padding

# Merge tuning (post DBSCAN)
MERGE_CENTROID_COS = float(os.environ.get("MERGE_CENTROID_COS", "0.60"))
MERGE_BESTPAIR_COS = float(os.environ.get("MERGE_BESTPAIR_COS", "0.70"))
MAX_FACES_PER_CLUSTER_FOR_PAIRWISE = int(os.environ.get("MAX_FACES_PER_CLUSTER_FOR_PAIRWISE", "40"))


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


def download_image(url: str) -> np.ndarray:
    r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("OpenCV could not decode image")
    return img


def parse_embeddings(rows: List[dict], field_name: str = "embedding") -> np.ndarray:
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


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def crop_face_thumb(img_bgr: np.ndarray, bbox: List[float]) -> bytes:
    """
    Recorta cara usando bbox [x1,y1,x2,y2] con padding.
    Devuelve JPG bytes.
    """
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

    # resize to stable size (max side = FACE_THUMB_SIZE)
    ch, cw = crop.shape[:2]
    mx = max(ch, cw)
    if mx > FACE_THUMB_SIZE:
        scale = FACE_THUMB_SIZE / float(mx)
        new_w = max(1, int(round(cw * scale)))
        new_h = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, jpg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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
    """
    label -> list of indices in faces_rows
    (solo labels != -1)
    """
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
    """
    Máxima similitud entre cualquier embedding de A y cualquiera de B.
    Submuestrea para no explotar CPU.
    """
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
    """
    Une clusters DBSCAN si parecen la misma persona.
    Anti-merge duro: si dos clusters aparecen en la MISMA foto => no pueden ser la misma persona.
    Esto evita merges falsos (dos personas distintas co-ocurrentes).
    """
    clusters = build_clusters_from_labels(faces_rows, labels)
    if len(clusters) <= 1:
        return labels

    centroids: Dict[int, np.ndarray] = {}
    photosets: Dict[int, set] = {}
    for lab, idxs in clusters.items():
        centroids[lab] = compute_centroid(embs, idxs)
        photosets[lab] = cluster_photo_set(faces_rows, idxs)

    # union-find
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

            # Anti-merge: si comparten alguna photo_id, son dos personas distintas en la misma foto
            if len(photosets[a].intersection(photosets[b])) > 0:
                continue

            c_sim = cosine_sim(centroids[a], centroids[b])
            if c_sim < MERGE_CENTROID_COS:
                continue

            bp = best_pair_cosine(embs, clusters[a], clusters[b])
            if bp < MERGE_BESTPAIR_COS:
                continue

            union(a, b)

    # map root -> new compact label
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
            "Worker initialized. Poll=%ss | DBSCAN eps=%s min_samples=%s | bucket=%s | merge centroid=%s bestpair=%s",
            POLL_SECONDS,
            DBSCAN_EPS,
            DBSCAN_MIN_SAMPLES,
            SUPABASE_PUBLIC_BUCKET,
            MERGE_CENTROID_COS,
            MERGE_BESTPAIR_COS,
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
            payload["progress"] = p_int

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
        # legacy table (dejamos limpio igual)
        self.sb.table("photo_embeddings").delete().eq("job_id", job_id).execute()

    def clear_album_face_embeddings(self, album_id: str) -> None:
        self.sb.table("face_embeddings").delete().eq("album_id", album_id).execute()

    def clear_album_clusters(self, album_id: str) -> None:
        self.sb.table("face_clusters").delete().eq("album_id", album_id).execute()

    def clear_photo_faces_for_album(self, album_id: str) -> None:
        photos = self.fetch_photos(album_id)
        for p in photos:
            self.sb.table("photo_faces").delete().eq("photo_id", p["id"]).execute()

    def create_cluster(self, album_id: str, thumbnail_url: Optional[str]) -> str:
        resp = self.sb.table("face_clusters").insert(
            {"album_id": album_id, "thumbnail_url": thumbnail_url}
        ).execute()
        data = resp.data or []
        if not data:
            raise RuntimeError("Could not create face_cluster")
        return data[0]["id"]

    def upsert_photo_face(self, photo_id: str, cluster_id: str) -> None:
        """
        Multifaces: una foto puede estar en múltiples clusters.
        Usamos upsert por PK compuesta (photo_id, cluster_id).
        """
        self.sb.table("photo_faces").upsert(
            {"photo_id": photo_id, "cluster_id": cluster_id},
            on_conflict="photo_id,cluster_id",
        ).execute()

    def insert_face_embedding_row(self, face_id: str, photo_id: str, album_id: str, embedding: List[float], bbox: List[float]) -> None:
        # Nota: requiere que face_embeddings tenga columna "id".
        self.sb.table("face_embeddings").insert(
            {
                "id": face_id,
                "photo_id": photo_id,
                "album_id": album_id,
                "embedding": embedding,
                "bbox": bbox,
            }
        ).execute()

    def upload_bytes(self, storage_path: str, raw: bytes, content_type: str) -> None:
        up_res = self.sb.storage.from_(SUPABASE_PUBLIC_BUCKET).upload(
            path=storage_path,
            file=raw,
            file_options={"content-type": content_type},
        )
        if getattr(up_res, "error", None):
            raise RuntimeError(f"Storage upload failed for {storage_path}: {up_res.error}")

    # -----------------------
    # ZIP ingest (cuando API todavía no crea photos)
    # -----------------------
    def download_zip_bytes(self, zip_path: str) -> bytes:
        url = to_public_storage_url(zip_path)
        r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
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

            ins = self.sb.table("photos").insert(
                {"id": photo_id, "album_id": album_id, "storage_path": storage_path}
            ).execute()
            if getattr(ins, "error", None):
                raise RuntimeError(f"DB insert into photos failed: {ins.error}")

            inserted += 1

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

        # Estados
        self.mark_job(job_id, "processing")
        self.update_album(album_id, status="processing", progress=1, error_message=None, completed=False)

        try:
            photos = self.fetch_photos(album_id)

            # Si API no creó photos, ingerimos ZIP
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

                self.update_album(album_id, status="processing", progress=5)

            total_photos = len(photos)
            logger.info("Album %s photos=%s", album_id, total_photos)

            # Idempotencia: limpiar output previo del álbum
            self.clear_job_embeddings(job_id)
            self.clear_photo_faces_for_album(album_id)
            self.clear_album_clusters(album_id)
            self.clear_album_face_embeddings(album_id)

            # 1) Detectar TODAS las caras + guardar embeddings por cara
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
                    img = download_image(url)
                    faces = self.face_app.get(img) or []

                    # filtrar caras sin embedding
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

                        # thumbnail de cara (recorte)
                        thumb_bytes = crop_face_thumb(img, bbox)
                        thumb_path = f"albums/{album_id}/faces/{face_id}.jpg"
                        self.upload_bytes(storage_path=thumb_path, raw=thumb_bytes, content_type="image/jpeg")
                        thumb_url = to_public_storage_url(thumb_path)
                        face_thumb_url_by_face_id[face_id] = thumb_url

                        # persist en DB: 1 row por cara
                        self.insert_face_embedding_row(
                            face_id=face_id,
                            photo_id=photo_id,
                            album_id=album_id,
                            embedding=emb,
                            bbox=bbox,
                        )

                        faces_rows.append(
                            {
                                "face_id": face_id,
                                "photo_id": photo_id,
                                "embedding": emb,
                            }
                        )
                        faces_total += 1

                except Exception as e:
                    logger.warning("Photo failed id=%s path=%s err=%s", photo_id, storage_path, str(e))

                # progreso 5..60 por fotos procesadas (detección)
                prog_ratio = (idx + 1) / max(1, total_photos)
                prog_int = clamp_int(int(round(5 + prog_ratio * 55)), 5, 60)
                self.update_album(album_id, status="processing", progress=prog_int)

            if faces_total == 0:
                raise RuntimeError(f"No faces detected/embedded. photos={total_photos}, photos_no_face={photos_no_face}")

            # 2) Clustering por CARA (no por foto)
            embeddings = np.array([r["embedding"] for r in faces_rows], dtype=np.float32)
            labels = run_dbscan(embeddings)

            # 2b) Merge inteligente post-DBSCAN (reduce split frontal vs perfil)
            labels = merge_clusters(faces_rows, embeddings, labels)

            self.update_album(album_id, status="processing", progress=70)

            # 3) Crear clusters + asignar photo_faces
            cluster_map: Dict[int, str] = {}

            unique_labels = sorted({int(x) for x in labels.tolist() if int(x) != -1})

            # Crear clusters para labels válidos
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
                    # ruido: cluster individual por cara
                    noise_faces += 1
                    thumb_url = face_thumb_url_by_face_id.get(str(r["face_id"]))
                    cluster_id = self.create_cluster(album_id=album_id, thumbnail_url=thumb_url)
                    self.upsert_photo_face(photo_id=photo_id, cluster_id=cluster_id)
                else:
                    cluster_id = cluster_map[lab_int]
                    self.upsert_photo_face(photo_id=photo_id, cluster_id=cluster_id)

                assigned_faces += 1

                # progreso 70..95 por asignación
                prog_ratio = (i + 1) / max(1, len(faces_rows))
                prog_int = clamp_int(int(round(70 + prog_ratio * 25)), 70, 95)
                self.update_album(album_id, status="processing", progress=prog_int)

            # 4) Final
            result = {
                "album_id": album_id,
                "photos_total": total_photos,
                "photos_with_faces": photos_with_faces,
                "photos_no_face": photos_no_face,
                "faces_total": faces_total,
                "faces_assigned": assigned_faces,
                "faces_noise": noise_faces,
                "clusters": len(unique_labels) + noise_faces,  # noise crea cluster propio
                "dbscan": {"eps": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES},
                "merge": {
                    "centroid_cos": MERGE_CENTROID_COS,
                    "bestpair_cos": MERGE_BESTPAIR_COS,
                    "max_pairwise": MAX_FACES_PER_CLUSTER_FOR_PAIRWISE,
                },
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
