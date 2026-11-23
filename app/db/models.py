from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from app.db.database import Base

class Person(Base):
    """
    Modelo de personas autorizadas por domicilio (HU01).
    Criterios del sistema:
    - Nombre completo
    - Rol o parentesco: 'Residente', 'Familiar' o 'Empleado'
    - Identificador único de negocio (unique_code)
    - Domicilio asignado (límite máx. 6 por domicilio)
    - Estado: 'Autorizada' (solo con fotos biométricas válidas) o 'Pendiente'
    """
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    unique_code = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False, default="Residente")
    domicile = Column(String, nullable=False, default="Domicilio 1")
    status = Column(String, nullable=False, default="Pendiente")
    photo_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Event(Base):
    """
    Modelo para registrar eventos de detección (HU06, HU08).
    Solo se registran detecciones confirmadas: una detección aislada no
    genera un evento. status es 'pending' o 'confirmed'; los eventos
    guardados por el sistema siempre son 'confirmed'.
    """
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_recognized = Column(Boolean, default=False)
    person_name = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    image_path = Column(String, nullable=True)
    status = Column(String, nullable=False, default="confirmed")
    detection_count = Column(Integer, nullable=False, default=1)
    first_seen = Column(DateTime, nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
