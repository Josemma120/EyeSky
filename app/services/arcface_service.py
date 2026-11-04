import numpy as np

# Umbral de referencia para ArcFace (típicamente entre 0.60 y 0.70)
DEFAULT_THRESHOLD = 0.65

# Instancia global diferida del modelo
_face_app = None

def get_face_app():
    """
    Carga diferida del modelo InsightFace/ArcFace para no bloquear
    el arranque del servidor si no se requiere de inmediato.
    """
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            # 'buffalo_s' es un modelo ligero ideal para pruebas
            _face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[ArcFace] Advertencia: No se pudo inicializar FaceAnalysis aún: {e}")
            _face_app = None
    return _face_app

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calcula la similitud de coseno entre dos vectores biométricos de ArcFace."""
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

def is_match(similarity: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Determina si la similitud supera el umbral de coincidencia."""
    return similarity >= threshold

