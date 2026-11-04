from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.db.database import get_db
from app.db.models import Person

router = APIRouter(prefix="/api/persons", tags=["Personas Autorizadas"])

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
    """Obtiene la lista básica de personas autorizadas registradas (HU04)."""
    return db.query(Person).all()

@router.post("/", response_model=PersonOut, status_code=201)
def register_person(person_in: PersonCreate, db: Session = Depends(get_db)):
    """Registra una persona básica en el sistema (HU01)."""
    if not person_in.name.strip():
        raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")
    
    new_person = Person(name=person_in.name, photo_path=person_in.photo_path)
    db.add(new_person)
    db.commit()
    db.refresh(new_person)
    return new_person

