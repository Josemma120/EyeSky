"""
app/routers/stream.py
Endpoints para transmisión de video en tiempo real y estado del reconocimiento.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.camera_service import camera_service

router = APIRouter(prefix="/api/stream", tags=["Streaming de Video en Vivo"])

@router.get("/video_feed")
def video_feed():
    """Transmite el flujo de video con las detecciones de ArcFace en formato MJPEG."""
    return StreamingResponse(
        camera_service.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/status")
def stream_status():
    """Devuelve las personas detectadas en tiempo real y el estado de la cámara."""
    return camera_service.get_status()

@router.post("/reload")
def reload_catalog():
    """Recarga el catálogo de firmas biométricas sin reiniciar el servidor."""
    camera_service.load_catalog()
    return {
        "ok": True,
        "message": "Catálogo biométrico recargado",
        "registered": list(camera_service.known_faces.keys())
    }

@router.post("/start")
def start_camera():
    """Inicia el streaming de la cámara."""
    camera_service.start()
    return {"ok": True, "active": camera_service.camera_active}

@router.post("/stop")
def stop_camera():
    """Detiene el streaming de la cámara."""
    camera_service.stop()
    return {"ok": True, "active": camera_service.camera_active}

