from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from app.db.database import Base

class Person(Base):
    """Modelo básico para registrar personas autorizadas (HU01)."""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    photo_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Event(Base):
    """Modelo básico para registrar eventos de detección (HU08)."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_recognized = Column(Boolean, default=False)
    person_name = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    image_path = Column(String, nullable=True)

