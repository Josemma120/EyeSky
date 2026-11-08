"""
app/services/arcface_service.py
Lógica de cálculo, extracción y validación biométrica con ArcFace.
Descarta fotos sin rostro y valida un mínimo de 4 imágenes válidas.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

# Umbral por defecto
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
MIN_FACES_REQUIRED = 4
CATALOG_PATH = Path("data/known_faces.pkl")

_face_app = None

def get_face_app():
    """Carga diferida del modelo ArcFace (buffalo_sc)."""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            _face_app.prepare(ctx_id=0, det_size=(320, 320))
        except Exception as e:
            print(f"[ArcFace] Error inicializando FaceAnalysis: {e}")
            _face_app = None
    return _face_app

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calcula la similitud de coseno entre dos vectores normalizados."""
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

def is_match(similarity: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Determina si la similitud supera el umbral configurado."""
    return similarity >= threshold

def read_image_safe(file_path: Path) -> Optional[np.ndarray]:
    """
    Lee una imagen de disco con soporte para JPG, PNG, WEBP y HEIC (iPhone).
    """
    suffix = file_path.suffix.lower()
    if suffix in [".heic", ".heif"]:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            from PIL import Image
            pil_img = Image.open(file_path)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ArcFace] No se pudo leer HEIC: {e}")
            return None
    return cv2.imread(str(file_path))

def register_person_biometrics(
    person_name: str,
    image_paths: List[Path],
    min_required: int = MIN_FACES_REQUIRED
) -> Tuple[bool, str, List[Path], List[str]]:
    """
    HU02 / HU03:
    1. Analiza cada imagen con ArcFace.
    2. Descarta y elimina automáticamente cualquier imagen donde no se detecte un rostro.
    3. Exige un mínimo forzoso de `min_required` (4) rostros válidos.
    Retorna: (éxito, mensaje, lista_fotos_validas, lista_nombres_descartadas)
    """
    app = get_face_app()
    if app is None:
        return False, "El modelo ArcFace no está disponible en este momento.", [], []

    embeddings = []
    valid_paths: List[Path] = []
    rejected_names: List[str] = []

    for img_path in image_paths:
        img = read_image_safe(img_path)
        if img is None:
            rejected_names.append(f"{img_path.name} (archivo no legible)")
            try:
                img_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        try:
            faces = app.get(img)
            if faces:
                # Tomar el rostro principal de mayor tamaño
                best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(best_face.embedding)
                valid_paths.append(img_path)
            else:
                # NO TIENE ROSTRO: descartar y borrar de disco
                rejected_names.append(f"{img_path.name} (sin rostro detectado)")
                try:
                    img_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"[ArcFace] Error al procesar {img_path.name}: {e}")
            rejected_names.append(f"{img_path.name} (error de análisis)")
            try:
                img_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Validar condición forzosa: mínimo 4 rostros válidos
    if len(valid_paths) < min_required:
        # Limpiar todas las fotos subidas ya que no cumplió el requerimiento
        for p in valid_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        detalle_rechazo = f" Fotos descartadas: {', '.join(rejected_names)}." if rejected_names else ""
        msg = (
            f"Registro rechazado: Se requieren al menos {min_required} fotos con rostro visible para garantizar la precisión de ArcFace. "
            f"Solo {len(valid_paths)} foto(s) contenían rostro.{detalle_rechazo}"
        )
        return False, msg, [], rejected_names

    # Promediar los vectores de los rostros válidos y normalizar
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    # Actualizar catálogo pkl
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    known_faces = {}
    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH, "rb") as f:
                known_faces = pickle.load(f)
        except Exception:
            known_faces = {}

    known_faces[person_name] = mean_embedding

    with open(CATALOG_PATH, "wb") as f:
        pickle.dump(known_faces, f)

    # Recargar en caliente en el servicio de cámara
    try:
        from app.services.camera_service import camera_service
        camera_service.load_catalog()
    except Exception as e:
        print(f"[ArcFace] Advertencia recargando catálogo: {e}")

    detalle_descarte = f" (Se descartaron {len(rejected_names)} fotos sin rostro)." if rejected_names else ""
    return True, f"Firma biométrica generada con éxito con {len(valid_paths)} fotos válidas.{detalle_descarte}", valid_paths, rejected_names

def remove_person_biometrics(person_name: str) -> bool:
    """Elimina una persona del catálogo known_faces.pkl."""
    if not CATALOG_PATH.exists():
        return False

    try:
        with open(CATALOG_PATH, "rb") as f:
            known_faces = pickle.load(f)

        if person_name in known_faces:
            del known_faces[person_name]
            with open(CATALOG_PATH, "wb") as f:
                pickle.dump(known_faces, f)

            from app.services.camera_service import camera_service
            camera_service.load_catalog()
            return True
    except Exception as e:
        print(f"[ArcFace] Error eliminando {person_name} del catálogo: {e}")

    return False
