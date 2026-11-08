"""
app/services/arcface_service.py
Motor de validación de calidad biométrica (HU02 / HU03) y extracción ArcFace.
Aplica los criterios de aceptación de Taiga:
- Mínimo 5 fotos válidas.
- Exactamente 1 rostro visible por foto.
- Resolución de imagen >= 640x480 px.
- Tamaño del rostro >= 150x150 px.
- Iluminación uniforme (sin contraluz ni sombras fuertes).
- Nitidez / enfoque (sin desenfoque por varianza de Laplaciano).
- Formato JPG o PNG, peso máximo 5 MB.
- Desglose detallado de motivos de rechazo.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Constantes de Calidad Taiga
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
MIN_FACES_REQUIRED = 5
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MIN_IMG_WIDTH = 640
MIN_IMG_HEIGHT = 480
MIN_FACE_WIDTH = 150
MIN_FACE_HEIGHT = 150

MIN_SHARPNESS_LAPLACIAN = 45.0  # Varianza mínima de Laplaciano para nitidez
MIN_BRIGHTNESS = 40.0           # Brillo mínimo promedio (0-255)
MAX_BRIGHTNESS = 220.0          # Brillo máximo promedio (evitar sobreexposición)
MAX_SHADOW_ASYMMETRY = 70.0     # Diferencia máxima entre mitades del rostro

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

def validate_photo_quality(file_path: Path) -> Tuple[bool, str, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Evalúa estrictamente los criterios de calidad de Taiga para una fotografía:
    1. Formato (JPG o PNG).
    2. Tamaño del archivo (máximo 5 MB).
    3. Resolución de la imagen (mínimo 640x480 px).
    4. Exactamente un solo rostro visible.
    5. Tamaño del rostro detectado (mínimo 150x150 px).
    6. Nitidez / ausencia de desenfoque.
    7. Iluminación adecuada y ausencia de sombras extremas o contraluz.

    Retorna: (es_valida, motivo_rechazo, embedding_arcface, metricas_calidad)
    """
    filename = file_path.name

    # 1. Validación de Formato
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Formato no permitido ({ext}). Solo se aceptan formatos JPG y PNG.", None, None

    # 2. Validación de Tamaño en Disco (máx 5 MB)
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return False, f"El archivo supera el tamaño máximo permitido de 5 MB ({size_mb:.2f} MB).", None, None
    except Exception as e:
        return False, f"Error al leer tamaño de archivo: {e}", None, None

    # Lectura de imagen
    img = cv2.imread(str(file_path))
    if img is None:
        return False, "La imagen no pudo ser leída o el archivo está dañado.", None, None

    # 3. Validación de Resolución mínima (640x480 px en cualquier orientación)
    h, w = img.shape[:2]
    es_valida_resolucion = (w >= MIN_IMG_WIDTH and h >= MIN_IMG_HEIGHT) or (w >= MIN_IMG_HEIGHT and h >= MIN_IMG_WIDTH)
    if not es_valida_resolucion:
        return False, f"Resolución de imagen insuficiente ({w}x{h} px). La imagen debe medir al menos 640x480 px.", None, None

    # Detección con ArcFace
    app = get_face_app()
    if app is None:
        return False, "Modelo de detección facial no disponible.", None, None

    try:
        faces = app.get(img)
    except Exception as e:
        return False, f"Error durante la inferencia facial: {e}", None, None

    # 4. Validación de Un Solo Rostro Visible
    if len(faces) == 0:
        return False, "No se detectó ningún rostro visible en la fotografía.", None, None
    if len(faces) > 1:
        return False, f"La fotografía contiene {len(faces)} rostros detectados. Cada foto debe contener un solo rostro visible.", None, None

    face = faces[0]
    box = face.bbox.astype(int)
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    face_w = x2 - x1
    face_h = y2 - y1

    # 5. Validación de Tamaño del Rostro (mínimo 150x150 px)
    if face_w < MIN_FACE_WIDTH or face_h < MIN_FACE_HEIGHT:
        return False, f"El rostro detectado es demasiado pequeño ({face_w}x{face_h} px). El tamaño mínimo del rostro es 150x150 px.", None, None

    # Recorte del rostro para análisis fotométrico
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False, "Coordenadas del rostro no válidas.", None, None

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # 6. Validación de Nitidez (Varianza del Laplaciano)
    sharpness = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS_LAPLACIAN:
        return False, f"La imagen presenta desenfoque o falta de nitidez (índice de nitidez: {sharpness:.1f}, mínimo requerido: {MIN_SHARPNESS_LAPLACIAN}).", None, None

    # 7. Validación de Iluminación y Sombras Fuertes
    mean_brightness = float(np.mean(gray_face))
    if mean_brightness < MIN_BRIGHTNESS:
        return False, f"Iluminación deficiente: el rostro está demasiado oscuro (brillo: {mean_brightness:.1f}/255, mínimo: {MIN_BRIGHTNESS}).", None, None
    if mean_brightness > MAX_BRIGHTNESS:
        return False, f"Sobreexposición o contraluz excesivo: el rostro está sobreexpuesto (brillo: {mean_brightness:.1f}/255, máximo: {MAX_BRIGHTNESS}).", None, None

    # Comprobar asimetría de sombras entre mitad izquierda y derecha
    mid_x = gray_face.shape[1] // 2
    if mid_x > 5:
        left_half = float(np.mean(gray_face[:, :mid_x]))
        right_half = float(np.mean(gray_face[:, mid_x:]))
        shadow_asymmetry = abs(left_half - right_half)
        if shadow_asymmetry > MAX_SHADOW_ASYMMETRY:
            return False, f"Iluminación no uniforme: sombras duras detectadas en un lado del rostro (diferencia lumínica: {shadow_asymmetry:.1f}, máx permitido: {MAX_SHADOW_ASYMMETRY}).", None, None

    # Estimación de pose / orientación mediante los 5 landmarks faciales
    pose_type = "frontal"
    if hasattr(face, "kps") and face.kps is not None and len(face.kps) == 5:
        # kps: [ojo_izq, ojo_der, nariz, boca_izq, boca_der]
        left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_dist = abs(right_eye[0] - left_eye[0])
        if eye_dist > 0:
            diff_nose = (nose[0] - eye_center_x) / eye_dist
            if diff_nose > 0.15:
                pose_type = "giro_derecha"
            elif diff_nose < -0.15:
                pose_type = "giro_izquierda"

    metrics = {
        "dimensions": f"{w}x{h}",
        "face_box": f"{face_w}x{face_h}",
        "sharpness": round(sharpness, 1),
        "brightness": round(mean_brightness, 1),
        "pose": pose_type
    }

    return True, "Criterios de calidad aprobados.", face.embedding, metrics

def register_person_biometrics(
    person_name: str,
    image_paths: List[Path],
    min_required: int = MIN_FACES_REQUIRED
) -> Tuple[bool, str, List[Path], List[str]]:
    """
    Procesa un lote de fotografías aplicando las reglas de calidad de Taiga.
    Elimina del disco las fotos que no cumplan los criterios e informa los motivos.
    Si quedan al menos min_required (5) fotos válidas, genera el vector ArcFace.
    """
    embeddings = []
    valid_paths: List[Path] = []
    rejection_reports: List[str] = []

    for img_path in image_paths:
        is_valid, reason, emb, _ = validate_photo_quality(img_path)

        if is_valid and emb is not None:
            embeddings.append(emb)
            valid_paths.append(img_path)
        else:
            rejection_reports.append(f"{img_path.name}: {reason}")
            # Descartar y borrar del disco la foto rechazada
            try:
                img_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Validar mínimo forzoso de 5 fotos
    if len(valid_paths) < min_required:
        # Limpiar las fotos que hayan pasado si el lote total no cumple las 5 requeridas
        for p in valid_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        detalles = "\n- " + "\n- ".join(rejection_reports) if rejection_reports else ""
        msg = (
            f"Registro no autorizado: Se requieren al menos {min_required} fotografías que cumplan todos los criterios de calidad "
            f"(solo {len(valid_paths)} foto(s) fueron válidas). Motivos de rechazo:{detalles}"
        )
        return False, msg, [], rejection_reports

    # Promedio y normalización del vector biométrico
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    # Actualizar catálogo persistente data/known_faces.pkl
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

    # Recarga en caliente para la cámara en tiempo real
    try:
        from app.services.camera_service import camera_service
        camera_service.load_catalog()
    except Exception as e:
        print(f"[ArcFace] Advertencia recargando catálogo: {e}")

    descarte_info = f" ({len(rejection_reports)} fotos descartadas por calidad)." if rejection_reports else ""
    return True, f"Firma biométrica generada con éxito con {len(valid_paths)} fotos aprobadas.{descarte_info}", valid_paths, rejection_reports

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
