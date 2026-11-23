"""
app/services/camera_service.py
Servicio de transmisión y reconocimiento de video en tiempo real.
Ejecuta la captura y procesamiento de rostros en un hilo en segundo plano
para transmitir el video fluido a la interfaz web mediante MJPEG.
"""

import os
# Limitar hilos de bajo nivel para que no sature todos los núcleos del procesador
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from app.core.config import settings
from app.services.detection_tracker import DetectionTracker

CATALOG_PATH = Path("data/known_faces.pkl")

class CameraService:
    def __init__(self):
        self.camera_source = os.getenv("CAMERA_SOURCE", "0")
        if self.camera_source.isdigit():
            self.camera_source = int(self.camera_source)

        # Umbral único de similitud (HU05): se lee de la configuración.
        self.threshold = settings.SIMILARITY_THRESHOLD
        # Frecuencia de inferencia: procesar 1 de cada N fotogramas (5 por defecto = ~3-4 veces por segundo)
        self.inference_interval = int(os.getenv("INFERENCE_INTERVAL", "5"))
        # HU06: seguimiento de detecciones; solo las confirmadas generan
        # un evento. Se alimenta solo en ciclos de inferencia.
        self.tracker = DetectionTracker(
            confirm_frames=settings.DETECTION_CONFIRM_FRAMES,
            confirm_window_seconds=settings.DETECTION_CONFIRM_WINDOW_SECONDS,
            cooldown_seconds=settings.DETECTION_COOLDOWN_SECONDS,
        )
        self.known_faces: Dict[str, np.ndarray] = {}
        self.face_app = None

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Último cuadro codificado en JPEG (para streaming)
        self.latest_jpeg: Optional[bytes] = None
        # Último cuadro nativo puro en alta resolución (sin recuadros, para registro biométrico)
        self.latest_raw_frame: Optional[np.ndarray] = None
        # Últimas detecciones para la UI
        self.latest_detections: List[dict] = []
        self.camera_active = False
        self.fps = 0.0

        self.load_catalog()

    def load_catalog(self):
        """Carga el catálogo de firmas biométricas."""
        if CATALOG_PATH.exists():
            try:
                with open(CATALOG_PATH, "rb") as f:
                    self.known_faces = pickle.load(f)
                print(f"[CameraService] Catálogo cargado: {list(self.known_faces.keys())}")
            except Exception as e:
                print(f"[CameraService] Error al cargar catálogo: {e}")
                self.known_faces = {}
        else:
            self.known_faces = {}

    def _init_arcface(self):
        """Inicializa InsightFace / ArcFace si aún no está cargado."""
        if self.face_app is None:
            try:
                from insightface.app import FaceAnalysis
                print("[CameraService] Inicializando ArcFace optimizado (buffalo_sc)...")
                app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=0, det_size=(320, 320))
                self.face_app = app
                print("[CameraService] ArcFace listo.")
            except Exception as e:
                print(f"[CameraService] Error iniciando ArcFace: {e}")
                self.face_app = None

    def _identify(self, face_embedding):
        """Compara un rostro contra el catálogo conocido."""
        if not self.known_faces:
            return "Desconocido", 0.0

        emb = face_embedding / np.linalg.norm(face_embedding)
        best_name = "Desconocido"
        best_score = -1.0

        for name, known_emb in self.known_faces.items():
            similarity = float(np.dot(emb, known_emb))
            if similarity > best_score:
                best_score = similarity
                if similarity >= self.threshold:
                    best_name = name

        return best_name, max(best_score, 0.0)

    def start(self):
        """Inicia el hilo de captura si no está corriendo."""
        if self.running:
            return
        self.load_catalog()
        self._init_arcface()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene el hilo de captura."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.camera_active = False

    def _emit_event(self, confirmed: dict):
        """
        Registra en BD una detección confirmada (HU06). Un error de BD
        nunca debe romper el bucle de captura.
        """
        try:
            from app.db.database import SessionLocal
            from app.db.models import Event

            with SessionLocal() as db:
                db.add(Event(
                    is_recognized=confirmed["is_recognized"],
                    person_name=confirmed["name"],
                    confidence=round(confirmed["confidence"], 1),
                    status="confirmed",
                    detection_count=confirmed["detection_count"],
                    first_seen=confirmed["first_seen"],
                    confirmed_at=confirmed["confirmed_at"],
                ))
                db.commit()
        except Exception as e:
            print(f"[CameraService] No se pudo registrar el evento: {e}")

    def _capture_loop(self):
        """Bucle continuo de captura y procesamiento."""
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            print(f"[CameraService] No se pudo abrir la cámara {self.camera_source}")
            self.camera_active = False
            self.running = False
            return

        # Solicitar resolución nativa alta a la cámara (ej. 720p o 1080p del iPhone/Webcam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_active = True
        frame_idx = 0
        fps_start = time.time()
        fps_counter = 0

        cached_faces = []

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_idx += 1
            fps_counter += 1

            # 1. Guardar copia nativa pura en alta resolución (para capturas de registro biométrico)
            raw_snapshot = frame.copy()

            # 2. Redimensionar para la transmisión en vivo y detección ligera en CPU
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                display_frame = cv2.resize(frame, (640, int(h * scale)))
            else:
                display_frame = frame.copy()

            # Procesar inferencia de ArcFace cada N fotogramas (por defecto cada 5)
            # Esto evita quemar la CPU calculando vectores 30 veces por segundo innecesariamente
            if self.face_app is not None and frame_idx % self.inference_interval == 0:
                try:
                    cached_faces = self.face_app.get(display_frame)
                except Exception as e:
                    print(f"[CameraService] Error en detección: {e}")
                    cached_faces = []

                # HU06: alimentar el tracker SOLO en ciclos de inferencia.
                # Una detección aislada no genera evento; solo las que se
                # confirman tras varios ciclos consecutivos lo hacen.
                observations = []
                for face in cached_faces:
                    obs_name, obs_score = self._identify(face.embedding)
                    observations.append({
                        "name": obs_name,
                        "confidence": round(obs_score * 100, 1),
                        "is_authorized": obs_name != "Desconocido",
                        "box": [int(v) for v in face.bbox.astype(int)],
                    })
                for confirmed in self.tracker.observe(observations):
                    self._emit_event(confirmed)

            detections_info = []

            # Dibujar detecciones sobre el cuadro de visualización
            for face in cached_faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                name, score = self._identify(face.embedding)
                is_authorized = (name != "Desconocido")
                track = self.tracker.lookup(name, [int(x1), int(y1), int(x2), int(y2)])
                status = track.status if track else "pending"
                detection_count = track.detection_count if track else 0

                if status == "confirmed":
                    color = (0, 255, 0) if is_authorized else (0, 0, 255)
                    label = f"{name} ({score * 100:.1f}%)"
                else:
                    # Pendiente de confirmación (HU06): se dibuja en ámbar
                    # y no genera evento ni registro.
                    color = (0, 165, 255)
                    label = f"{name} ({score * 100:.1f}%) - Verificando..."

                # Bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Etiqueta
                y_label = max(y1 - 10, 20)
                cv2.putText(
                    display_frame,
                    label,
                    (x1, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA
                )

                detections_info.append({
                    "name": name,
                    "confidence": round(score * 100, 1),
                    "is_authorized": is_authorized,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "status": status,
                    "detection_count": detection_count
                })

            # Medir FPS cada segundo
            now = time.time()
            if now - fps_start >= 1.0:
                self.fps = round(fps_counter / (now - fps_start), 1)
                fps_counter = 0
                fps_start = now

            # Codificar cuadro a JPEG para el streaming
            ret, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_raw_frame = raw_snapshot
                    self.latest_detections = detections_info

            time.sleep(0.01)

        cap.release()
        self.camera_active = False

    def generate_mjpeg(self):
        """Generador HTTP para StreamingResponse multipart/x-mixed-replace."""
        self.start()
        while self.running:
            with self.lock:
                frame_bytes = self.latest_jpeg

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.04)  # ~25 FPS en el stream

    def get_snapshot_jpeg(self) -> Optional[bytes]:
        """
        Devuelve el fotograma nativo puro en alta resolución (sin anotaciones ni recuadros)
        en formato JPEG con calidad 95 para el registro biométrico de usuarios.
        """
        self.start()
        with self.lock:
            if self.latest_raw_frame is None:
                return None
            frame_to_encode = self.latest_raw_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ret:
            return buffer.tobytes()
        return None

    def get_status(self):
        """Retorna el estado actual del servicio para la interfaz web."""
        with self.lock:
            return {
                "active": self.camera_active,
                "fps": self.fps,
                "registered_persons": list(self.known_faces.keys()),
                "current_detections": self.latest_detections,
                "timestamp": int(time.time() * 1000)
            }

# Instancia singleton del servicio de cámara
camera_service = CameraService()

