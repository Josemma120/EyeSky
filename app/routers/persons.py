import os
import re
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator

from app.db.database import get_db
from app.db.models import Person
from app.services.arcface_service import (
    register_person_biometrics,
    remove_person_biometrics,
    MIN_FACES_REQUIRED
)
from app.services.dataset_sync import sync_dataset_to_db

router = APIRouter(prefix="/api/persons", tags=["Personas Autorizadas"])

DATASET_DIR = Path("dataset")
MAX_PERSONS_PER_DOMICILE = 6
ALLOWED_ROLES = {"Residente", "Familiar", "Empleado"}

# ---- Modelos Pydantic ----

class PersonCreate(BaseModel):
    name: str
    unique_code: Optional[str] = None
    role: Optional[str] = "Residente"
    domicile: Optional[str] = "Domicilio 1"
    photo_path: Optional[str] = None

class PersonOut(BaseModel):
    id: int
    unique_code: str
    name: str
    role: str
    domicile: str
    status: str
    photo_path: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---- Helpers de Validación ----

def check_domicile_limit(db: Session, domicile: str):
    """Criterio Taiga: Límite de hasta 6 personas autorizadas por domicilio."""
    count = db.query(Person).filter(
        Person.domicile == domicile,
        Person.status == "Autorizada"
    ).count()

    if count >= MAX_PERSONS_PER_DOMICILE:
        raise HTTPException(
            status_code=400,
            detail=f"Límite alcanzado: Se ha alcanzado el límite máximo de {MAX_PERSONS_PER_DOMICILE} personas autorizadas para '{domicile}'."
        )
    return count

def check_duplicate_person(db: Session, name: str, unique_code: Optional[str], exclude_id: Optional[int] = None):
    """Criterio Taiga: El sistema impide registros duplicados."""
    query = db.query(Person).filter(
        (Person.name.ilike(name)) | (Person.unique_code == unique_code)
    )
    if exclude_id is not None:
        query = query.filter(Person.id != exclude_id)

    existing = query.first()
    if existing:
        if existing.name.lower() == name.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Registro duplicado: Ya existe una persona registrada con el nombre '{name}'."
            )
        if unique_code and existing.unique_code == unique_code:
            raise HTTPException(
                status_code=400,
                detail=f"Registro duplicado: Ya existe una persona con el identificador único '{unique_code}'."
            )

def generate_default_code(db: Session, domicile: str) -> str:
    """Genera un código único de negocio si no se proporciona (ej: DOM-001)."""
    total = db.query(Person).count() + 1
    return f"DOM-{total:03d}"


# ---- Rutas API ----

@router.get("/", response_model=List[PersonOut])
def list_persons(db: Session = Depends(get_db)):
    """Obtiene la lista de todas las personas registradas en el sistema (HU04)."""
    return db.query(Person).order_by(Person.id.asc()).all()

@router.post("/sync-dataset")
def sync_existing_dataset(db: Session = Depends(get_db)):
    """
    Importa a la base de datos las personas que ya existen en `dataset/` o en
    `data/known_faces.pkl`. No duplica registros y es seguro ejecutarlo varias veces.
    """
    result = sync_dataset_to_db(db)
    return {
        "ok": True,
        "message": (
            f"Sincronización completada: {len(result['created'])} nueva(s), "
            f"{len(result['updated'])} actualizada(s), "
            f"{len(result['skipped'])} ya existente(s)."
        ),
        **result,
    }

@router.get("/domicile-stats")
def get_domicile_stats(domicile: str = "Domicilio 1", db: Session = Depends(get_db)):
    """Informa sobre el cupo de personas autorizadas en un domicilio (máx 6)."""
    count = db.query(Person).filter(
        Person.domicile == domicile,
        Person.status == "Autorizada"
    ).count()

    return {
        "domicile": domicile,
        "authorized_count": count,
        "max_limit": MAX_PERSONS_PER_DOMICILE,
        "available_slots": max(0, MAX_PERSONS_PER_DOMICILE - count),
        "limit_reached": count >= MAX_PERSONS_PER_DOMICILE
    }

@router.post("/", response_model=PersonOut, status_code=201)
def register_person_basic(person_in: PersonCreate, db: Session = Depends(get_db)):
    """
    Vía básica: Registra una persona en la base de datos.
    Queda como 'Pendiente' hasta que sus fotos sean validadas biométricamente (HU01 / HU02).
    """
    clean_name = person_in.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío.")

    clean_domicile = (person_in.domicile or "Domicilio 1").strip()
    clean_role = (person_in.role or "Residente").capitalize()
    if clean_role not in ALLOWED_ROLES:
        clean_role = "Residente"

    clean_code = (person_in.unique_code or "").strip()
    if not clean_code:
        clean_code = generate_default_code(db, clean_domicile)

    # 1. Validar límite de 6 personas
    check_domicile_limit(db, clean_domicile)

    # 2. Validar duplicados
    check_duplicate_person(db, clean_name, clean_code)

    # 3. Guardar con estado 'Pendiente'
    new_person = Person(
        name=clean_name,
        unique_code=clean_code,
        role=clean_role,
        domicile=clean_domicile,
        status="Pendiente",
        photo_path=person_in.photo_path
    )
    db.add(new_person)
    db.commit()
    db.refresh(new_person)
    return new_person

@router.post("/upload", status_code=201)
async def upload_and_register_person(
    name: str = Form(...),
    role: str = Form("Residente"),
    unique_code: Optional[str] = Form(None),
    domicile: str = Form("Domicilio 1"),
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Criterios de Aceptación Taiga (HU01, HU02, HU03):
    1. Límite de hasta 6 personas autorizadas por domicilio.
    2. Registro con nombre, rol (Residente, Familiar, Empleado) e identificador único.
    3. Impide registros duplicados.
    4. Mínimo 5 fotos válidas con estricto control de calidad.
    5. Estado 'Autorizada' solo si las fotos son aprobadas por ArcFace.
    """
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="El nombre de la persona es obligatorio.")

    clean_domicile = (domicile or "Domicilio 1").strip()
    clean_role = (role or "Residente").capitalize()
    if clean_role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Rol no permitido '{role}'. Debe ser uno de: {', '.join(sorted(ALLOWED_ROLES))}."
        )

    clean_code = (unique_code or "").strip()
    if not clean_code:
        clean_code = generate_default_code(db, clean_domicile)

    # 1. Regla: Límite estricto de 6 personas autorizadas por domicilio
    check_domicile_limit(db, clean_domicile)

    # 2. Regla: Impedir duplicados por nombre o código único
    check_duplicate_person(db, clean_name, clean_code)

    # 3. Regla: Validar cantidad mínima de fotografías recibidas (al menos 5)
    if not photos or len(photos) < MIN_FACES_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"Requisito no cumplido: Debes subir al menos {MIN_FACES_REQUIRED} fotografías para el registro biométrico (has enviado {len(photos) if photos else 0})."
        )

    person_folder = DATASET_DIR / clean_name
    person_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    try:
        for idx, photo in enumerate(photos):
            if not photo.filename:
                continue

            ext = Path(photo.filename).suffix.lower()
            if not ext:
                ext = ".jpg"

            target_filename = f"foto_{idx + 1}_{int(datetime.utcnow().timestamp())}{ext}"
            target_path = person_folder / target_filename

            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)

            saved_paths.append(target_path)

        if len(saved_paths) < MIN_FACES_REQUIRED:
            raise HTTPException(
                status_code=400,
                detail=f"Se requieren mínimo {MIN_FACES_REQUIRED} archivos de imagen válidos."
            )

        # 4. Evaluación de Calidad Biometría (HU02 / HU03)
        # Filtra fotos sin rostro, fotos borrosas, mala luz, resolución insuficiente, etc.
        success, message, valid_paths, rejection_reports = register_person_biometrics(
            clean_name,
            saved_paths,
            min_required=MIN_FACES_REQUIRED
        )

        if not success:
            # Si no cumplió las 5 fotos válidas, limpiar la carpeta
            if person_folder.exists() and not any(person_folder.iterdir()):
                shutil.rmtree(person_folder, ignore_errors=True)
            raise HTTPException(status_code=400, detail=message)

        # 5. Crear la persona en la base de datos con estado explícito 'Autorizada'
        new_person = Person(
            name=clean_name,
            unique_code=clean_code,
            role=clean_role,
            domicile=clean_domicile,
            status="Autorizada",
            photo_path=str(valid_paths[0])
        )
        db.add(new_person)
        db.commit()
        db.refresh(new_person)

        return {
            "ok": True,
            "message": f"Persona '{clean_name}' ({clean_role}) registrada exitosamente como 'Autorizada'.",
            "person": {
                "id": new_person.id,
                "unique_code": new_person.unique_code,
                "name": new_person.name,
                "role": new_person.role,
                "domicile": new_person.domicile,
                "status": new_person.status,
                "photo_path": new_person.photo_path
            },
            "valid_photos_count": len(valid_paths),
            "rejected_photos_count": len(rejection_reports),
            "rejections": rejection_reports
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

    # Quitar del catálogo persistente pkl
    remove_person_biometrics(name)

    # Limpiar carpeta en dataset si existe
    person_dir = DATASET_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir, ignore_errors=True)

    return {"ok": True, "message": f"Persona '{name}' eliminada correctamente del sistema."}
