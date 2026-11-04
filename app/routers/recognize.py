from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.services.arcface_service import calculate_similarity, is_match, DEFAULT_THRESHOLD
import numpy as np

router = APIRouter(prefix="/api/recognize", tags=["Reconocimiento Facial"])

class CompareVectorsIn(BaseModel):
    vector1: List[float]
    vector2: List[float]
    threshold: float = DEFAULT_THRESHOLD

@router.get("/status")
def recognition_status():
    """Consulta el estado del servicio de reconocimiento y el umbral configurado."""
    return {
        "status": "ready",
        "model": "ArcFace",
        "default_threshold": DEFAULT_THRESHOLD,
        "description": "Comparación por similitud de coseno sobre embeddings de 512 dimensiones"
    }

@router.post("/compare")
def compare_vectors(body: CompareVectorsIn):
    """
    Ruta básica de prueba: Recibe dos vectores y calcula si corresponden a la misma persona.
    """
    v1 = np.array(body.vector1, dtype=np.float32)
    v2 = np.array(body.vector2, dtype=np.float32)

    similarity = calculate_similarity(v1, v2)
    match = is_match(similarity, threshold=body.threshold)

    return {
        "similarity": round(similarity, 4),
        "threshold": body.threshold,
        "is_match": match,
        "status": "authorized" if match else "unknown"
    }

