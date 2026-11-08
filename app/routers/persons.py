import os
import re
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import Person
from app.services.arcface_service import register_person_biometrics, remove_person_biometrics, MIN_FACES_REQUIRED

router = APIRouter(prefix="/api/persons", tags=["Personas Autorizadas"])

DATASET_DIR = Path("dataset")

class PersonCreate(BaseModel):
    name: str
    photo_path: Optional[str] = None

class PersonOut(BaseModel):
    id: int
    name: str
    photo_path: Optional[str] = None

    class Config:
        from_attributes = True

@router.get("/", response_model=List[PersonOut])
def list_persons(db: Session = Depends(get_db)):
    """Obtiene la lista de personas autorizadas registradas en el sistema (HU04)."""
    return db.query(Person).all()

@router.post("/", response_model=PersonOut, status_code=201)
def register_person(person_in: PersonCreate, db: Session = Depends(get_db)):
    """Registra una persona básica en la base de datos (HU01)."""
    clean_name = person_in.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío.")

    new_person = Person(name=clean_name, photo_path=person_in.photo_path)
    db.add(new_person)
    db.commit()
    db.refresh(new_person)
    return new_person

@router.post("/upload", status_code=201)
async def upload_and_register_person(
    name: str = Form(...),
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    HU01 / HU02 / HU03:
    - Exige forzosamente al menos 4 fotografías.
    - Filtra y descarta automáticamente imágenes que no contengan rostros.
    - Extrae la firma biométrica con ArcFace y actualiza el sistema en caliente.
    """
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="El nombre de la persona es obligatorio.")

    # 1. Validación forzosa de cantidad mínima
    if not photos or len(photos) < MIN_FACES_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"Requisito mínimo no cumplido: Debes subir al menos {MIN_FACES_REQUIRED} fotografías claras del rostro (has enviado {len(photos) if photos else 0})."
        )

    # Crear carpeta en dataset/ para la persona
    person_folder = DATASET_DIR / clean_name
    person_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    try:
        for idx, photo in enumerate(photos):
            if not photo.filename:
                continue

            # Obtener extensión del archivo
            ext = Path(photo.filename).suffix.lower()
            if not ext:
                ext = ".jpg"

            # Nombre de archivo seguro
            target_filename = f"foto_{idx + 1}_{int(Path(photo.filename).stem[:10] if Path(photo.filename).stem.isdigit() else idx)}{ext}"
            target_path = person_folder / target_filename

            # Guardar en disco temporalmente para análisis
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)

            saved_paths.append(target_path)

        if not saved_paths or len(saved_paths) < MIN_FACES_REQUIRED:
            raise HTTPException(status_code=400, detail=f"Se requieren mínimo {MIN_FACES_REQUIRED} archivos de imagen válidos.")

        # 2. Análisis con ArcFace: descarta fotos sin rostro y valida mínimo de 4
        success, message, valid_paths, rejected_names = register_person_biometrics(
            clean_name,
            saved_paths,
            min_required=MIN_FACES_REQUIRED
        )

        if not success:
            # Si no cumplió las 4 fotos válidas con rostro, eliminar la carpeta para mantener limpio el dataset
            if person_folder.exists() and not any(person_folder.iterdir()):
                shutil.rmtree(person_folder, ignore_errors=True)
            raise HTTPException(status_code=400, detail=message)

        # 3. Registrar o actualizar en la base de datos SQLite
        db_person = db.query(Person).filter(Person.name == clean_name).first()
        if not db_person:
            db_person = Person(
                name=clean_name,
                photo_path=str(valid_paths[0])
            )
            db.add(db_person)
        else:
            db_person.photo_path = str(valid_paths[0])

        db.commit()
        db.refresh(db_person)

        return {
            "ok": True,
            "message": message,
            "person": {
                "id": db_person.id,
                "name": db_person.name,
                "photo_path": db_person.photo_path
            },
            "valid_photos_count": len(valid_paths),
            "rejected_photos_count": len(rejected_names),
            "rejected_files": rejected_names
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno procesando fotos: {str(e)}")

@router.delete("/{person_id}")
def delete_person(person_id: int, db: Session = Depends(get_db)):
    """Elimina una persona de la base de datos y del catálogo biométrico de ArcFace."""
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if not db_person:
        raise HTTPException(status_code=404, detail="Persona no encontrada.")

    name = db_person.name
    db.delete(db_person)
    db.commit()

    # Quitar del catálogo pkl
    remove_person_biometrics(name)

    return {"ok": True, "message": f"Persona '{name}' eliminada correctamente del sistema."}
