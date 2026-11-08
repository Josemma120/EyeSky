"""
app/services/dataset_sync.py
Sincroniza las personas que ya existen en `dataset/` y en `data/known_faces.pkl`
con la tabla `persons` de la base de datos.

La lista de "Personas Autorizadas" del panel se alimenta de SQLite, mientras que
`register_faces.py` solo genera el catálogo biométrico. Esta sincronización evita
que las personas del dataset desaparezcan de la lista por no haber sido creadas
desde el formulario de registro.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Set

from sqlalchemy.orm import Session

from app.db.models import Person

DATASET_DIR = Path("dataset")
CATALOG_PATH = Path("data/known_faces.pkl")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_ROLE = "Residente"
DEFAULT_DOMICILE = "Domicilio 1"


def _normalize(name: str) -> str:
    """Normaliza un nombre para comparar sin distinguir mayúsculas ni espacios."""
    return " ".join(name.strip().split()).casefold()


def load_dataset_people() -> Dict[str, Path]:
    """Devuelve {nombre_normalizado: carpeta} de las carpetas de `dataset/`."""
    people: Dict[str, Path] = {}
    if not DATASET_DIR.exists():
        return people

    for folder in DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue
        key = _normalize(folder.name)
        if key:
            people[key] = folder

    return people


def load_biometric_people() -> Dict[str, str]:
    """Devuelve {nombre_normalizado: nombre_original} de `known_faces.pkl`."""
    people: Dict[str, str] = {}
    if not CATALOG_PATH.exists():
        return people

    try:
        with open(CATALOG_PATH, "rb") as f:
            catalog = pickle.load(f)

        if isinstance(catalog, dict):
            for name in catalog.keys():
                display_name = str(name).strip()
                key = _normalize(display_name)
                if key:
                    people[key] = display_name
    except Exception as e:
        print(f"[DatasetSync] No se pudo leer el catálogo biométrico: {e}")

    return people


def _first_photo(folder: Path) -> Path | None:
    """Devuelve la primera imagen válida de una carpeta de persona, si existe."""
    try:
        images = sorted(
            (
                item for item in folder.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda item: item.name.lower(),
        )
    except OSError:
        return None

    return images[0] if images else None


def _used_codes(db: Session) -> Set[str]:
    return {
        code for (code,) in db.query(Person.unique_code).all()
        if code
    }


def _next_code(used_codes: Set[str], index: int) -> str:
    """Genera un identificador DOM-### que no esté ocupado."""
    while True:
        code = f"DOM-{index:03d}"
        if code not in used_codes:
            used_codes.add(code)
            return code
        index += 1


def sync_dataset_to_db(db: Session) -> Dict[str, object]:
    """
    Importa a la tabla `persons` las personas del dataset/catálogo que aún no
    estén registradas. Es idempotente: se puede ejecutar en cada arranque.

    - Si la persona existe en el catálogo biométrico, queda como 'Autorizada'.
    - Si solo existe la carpeta del dataset, queda como 'Pendiente'.
    - No modifica rol, domicilio ni el estado de personas ya 'Autorizadas'.
    """
    dataset_people = load_dataset_people()
    biometric_people = load_biometric_people()

    candidates: Dict[str, str] = {}
    for key, folder in dataset_people.items():
        candidates[key] = folder.name.strip()

    # Incluye personas que solo estén en el .pkl (sin carpeta en dataset/).
    for key, display_name in biometric_people.items():
        candidates.setdefault(key, display_name)

    existing = {
        _normalize(person.name): person
        for person in db.query(Person).all()
    }

    used_codes = _used_codes(db)
    initial_count = len(existing)
    created: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []

    for key, display_name in candidates.items():
        folder = dataset_people.get(key)
        has_biometrics = key in biometric_people
        person = existing.get(key)

        if person is not None:
            changed = False

            # Si ya tiene firma biométrica pero seguía como 'Pendiente', autorizarla.
            if has_biometrics and person.status != "Autorizada":
                person.status = "Autorizada"
                changed = True

            if not person.photo_path and folder is not None:
                photo = _first_photo(folder)
                if photo is not None:
                    person.photo_path = str(photo)
                    changed = True

            if changed:
                updated.append(person.name)
            else:
                skipped.append(person.name)
            continue

        photo = _first_photo(folder) if folder is not None else None
        new_person = Person(
            name=display_name,
            unique_code=_next_code(used_codes, initial_count + len(created) + 1),
            role=DEFAULT_ROLE,
            domicile=DEFAULT_DOMICILE,
            status="Autorizada" if has_biometrics else "Pendiente",
            photo_path=str(photo) if photo is not None else None,
        )
        db.add(new_person)
        existing[key] = new_person
        created.append(display_name)

    if created or updated:
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "dataset_count": len(dataset_people),
        "biometric_count": len(biometric_people),
    }
