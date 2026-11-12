"""
app/routers/dataset.py
Endpoints para validar la nitidez de una foto y limpiar el dataset local.
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.services.dataset_quality import (
    IMAGE_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    MIN_SHARPNESS,
    calculate_sharpness_from_bytes,
    clean_dataset,
)

router = APIRouter(prefix="/api/dataset", tags=["Calidad del Dataset"])


@router.post("/validate-sharpness")
async def validate_sharpness(
    file: UploadFile = File(...),
    threshold: float = Query(
        MIN_SHARPNESS,
        ge=0,
        description="Valor mínimo de varianza del Laplaciano para considerar la foto nítida.",
    ),
):
    """
    Valida la nitidez de una fotografía sin registrarla.

    Devuelve el índice de nitidez (varianza del Laplaciano) y si supera el
    umbral configurado. Es útil para revisar una foto antes de subirla al
    dataset o para explicar el criterio de calidad usado por ArcFace.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Debes enviar un archivo de imagen.")

    extension = Path(file.filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no permitido ({extension or 'sin extensión'}). "
                f"Usa uno de: {', '.join(sorted(IMAGE_EXTENSIONS))}."
            ),
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="El archivo supera el tamaño máximo permitido de 10 MB.",
        )

    try:
        sharpness = calculate_sharpness_from_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_sharp = sharpness >= threshold
    return {
        "ok": True,
        "filename": file.filename,
        "sharpness": round(sharpness, 2),
        "threshold": threshold,
        "is_sharp": is_sharp,
        "status": "nitida" if is_sharp else "borrosa",
        "message": (
            f"La imagen es nítida (índice {sharpness:.1f} >= {threshold:.1f})."
            if is_sharp
            else f"La imagen está borrosa (índice {sharpness:.1f} < {threshold:.1f})."
        ),
    }


@router.post("/clean")
def clean_dataset_endpoint(
    dry_run: bool = Query(
        True,
        description="Si es true, solo simula la limpieza. Usa false para borrar archivos.",
    ),
    threshold: float = Query(
        MIN_SHARPNESS,
        ge=0,
        description="Umbral mínimo de nitidez para conservar una imagen.",
    ),
    remove_duplicates: bool = Query(
        True,
        description="Elimina imágenes repetidas dentro del dataset.",
    ),
):
    """
    Revisa y limpia las fotografías de `dataset/`.

    Detecta archivos con formato no permitido, corruptos, borrosos y duplicados.
    Por seguridad, `dry_run=true` viene activado por defecto: primero muestra
    qué se eliminaría y solo borra cuando se envía `dry_run=false`.

    Después de una limpieza real conviene ejecutar `python register_faces.py`
    para regenerar `data/known_faces.pkl` con las fotos que quedaron.
    """
    return clean_dataset(
        dry_run=dry_run,
        threshold=threshold,
        remove_duplicates=remove_duplicates,
    )
