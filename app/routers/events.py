from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.db.database import get_db
from app.db.models import Event

router = APIRouter(prefix="/api/events", tags=["Eventos y Alertas"])

class EventCreate(BaseModel):
    is_recognized: bool
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    image_path: Optional[str] = None
    status: str = "confirmed"
    detection_count: int = 1
    first_seen: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None

class EventOut(BaseModel):
    id: int
    timestamp: datetime
    is_recognized: bool
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "confirmed"
    detection_count: int = 1
    first_seen: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

@router.get("/", response_model=List[EventOut])
def list_events(db: Session = Depends(get_db)):
    """Consulta el historial de eventos registrados (HU08)."""
    return db.query(Event).order_by(Event.timestamp.desc()).limit(50).all()

@router.post("/", response_model=EventOut, status_code=201)
def create_event(event_in: EventCreate, db: Session = Depends(get_db)):
    """Registra un evento de detección confirmada (HU06) o manual."""
    new_event = Event(
        is_recognized=event_in.is_recognized,
        person_name=event_in.person_name,
        confidence=event_in.confidence,
        image_path=event_in.image_path,
        status=event_in.status,
        detection_count=event_in.detection_count,
        first_seen=event_in.first_seen,
        confirmed_at=event_in.confirmed_at
    )
    db.add(new_event)
    db.commit()
    db.refresh(new_event)
    return new_event

