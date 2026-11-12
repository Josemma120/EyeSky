"""
app/services/dataset_quality.py
Utilidades para validar la nitidez de una imagen y limpiar el dataset local.

- La nitidez se mide con la varianza del Laplaciano: cuanto mayor es el valor,
  más bordes definidos tiene la imagen y menos desenfoque presenta.
- La limpieza puede ejecutarse en modo simulación (`dry_run`) para revisar qué
  archivos se eliminarían antes de borrarlos definitivamente.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

DATASET_DIR = Path("dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Mismo umbral que usa el registro biométrico de ArcFace.
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS_LAPLACIAN", "45.0"))


def calculate_sharpness(image: np.ndarray) -> float:
    """Calcula la varianza del Laplaciano de una imagen BGR o en escala de grises."""
    if image is None or image.size == 0:
        raise ValueError("La imagen está vacía o no se pudo procesar.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_sharpness_from_bytes(data: bytes) -> float:
    """Calcula la nitidez a partir de los bytes de una imagen subida."""
    if not data:
        raise ValueError("El archivo está vacío.")

    if len(data) > MAX_UPLOAD_BYTES:
        size_mb = len(data) / (1024 * 1024)
        raise ValueError(f"El archivo supera el tamaño máximo de 10 MB ({size_mb:.2f} MB).")

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("La imagen no pudo ser leída o el formato no es válido.")

    return calculate_sharpness(image)


def calculate_sharpness_from_path(path: Path) -> Optional[float]:
    """Calcula la nitidez de un archivo; devuelve None si no se puede leer."""
    image = cv2.imread(str(path))
    if image is None:
        return None
    return calculate_sharpness(image)


def _file_hash(path: Path) -> str:
    """Calcula SHA-256 para detectar imágenes duplicadas."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_folders() -> List[Path]:
    if not DATASET_DIR.exists():
        return []
    return sorted(
        (item for item in DATASET_DIR.iterdir() if item.is_dir()),
        key=lambda item: item.name.lower(),
    )


def clean_dataset(
    dry_run: bool = True,
    threshold: float = MIN_SHARPNESS,
    remove_duplicates: bool = True,
) -> Dict[str, object]:
    """
    Revisa las fotos de cada carpeta de `dataset/` y detecta:

    - Archivos con extensión no permitida.
    - Archivos corruptos o que OpenCV no puede leer.
    - Imágenes borrosas (varianza del Laplaciano menor al umbral).
    - Imágenes duplicadas dentro del dataset.

    Si `dry_run=False`, elimina los archivos problemáticos. Si es `True`, solo
    informa qué se eliminaría. No borra carpetas de personas.
    """
    folders = _dataset_folders()
    if not folders:
        return {
            "dry_run": dry_run,
            "threshold": threshold,
            "scanned": 0,
            "kept": 0,
            "removed_count": 0,
            "removed": [],
            "errors": [],
            "message": "No se encontraron carpetas de personas en dataset/.",
        }

    scanned = 0
    kept = 0
    removed: List[Dict[str, object]] = []
    errors: List[str] = []
    seen_hashes: Dict[str, Path] = {}

    for folder in folders:
        for file in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
            if not file.is_file():
                continue

            scanned += 1
            reason: Optional[str] = None
            sharpness: Optional[float] = None

            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                reason = "formato_no_soportado"
            else:
                sharpness = calculate_sharpness_from_path(file)
                if sharpness is None:
                    reason = "archivo_corrupto"
                elif sharpness < threshold:
                    reason = "borrosa"
                elif remove_duplicates:
                    try:
                        digest = _file_hash(file)
                    except OSError as e:
                        reason = f"error_leyendo_archivo: {e}"
                    else:
                        if digest in seen_hashes:
                            reason = f"duplicada_de: {seen_hashes[digest].name}"
                        else:
                            seen_hashes[digest] = file

            if reason is None:
                kept += 1
                continue

            deleted = False
            if not dry_run:
                try:
                    file.unlink()
                    deleted = True
                except OSError as e:
                    errors.append(f"{file}: {e}")

            removed.append(
                {
                    "path": str(file),
                    "person": folder.name,
                    "filename": file.name,
                    "reason": reason,
                    "sharpness": round(sharpness, 2) if sharpness is not None else None,
                    "deleted": deleted,
                }
            )

    action = "eliminados" if not dry_run else "se eliminarían"
    message = (
        f"Dataset revisado: {scanned} archivo(s), {kept} conservado(s), "
        f"{len(removed)} {action}."
    )
    if errors:
        message += f" {len(errors)} archivo(s) no se pudieron borrar."

    return {
        "dry_run": dry_run,
        "threshold": threshold,
        "scanned": scanned,
        "kept": kept,
        "removed_count": len(removed),
        "removed": removed,
        "errors": errors,
        "message": message,
    }
