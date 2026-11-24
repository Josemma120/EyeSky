"""
app/services/detection_tracker.py
Módulo de seguimiento y validación de detecciones en tiempo real (HU06).

Criterios de Aceptación del Sistema (HU06):
1. Una detección aislada no genera alerta.
2. La detección debe mantenerse durante 3 revisiones seguidas (~1 segundo) antes de confirmarse.
3. Si la detección desaparece antes de cumplir esa condición, se descarta y no se genera alerta.
4. Una detección confirmada continúa con la identificación y queda guardada como un evento en la base de datos.
5. La regla de confirmación puede auditarse y validarse mediante pruebas automatizadas.
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from app.db.database import SessionLocal
from app.db.models import Event

REQUIRED_CONSECUTIVE_REVISIONS = 3
EVENT_COOLDOWN_SECONDS = 15.0  # Cooldown para no saturar la BD mientras la misma persona confirmada siga en pantalla


class TrackedFace:
    """Representa el seguimiento de un rostro a lo largo de las revisiones de inferencia."""

    def __init__(self, track_id: int, box: List[int], name: str, confidence: float, is_authorized: bool):
        self.track_id = track_id
        self.box = box
        self.name = name
        self.confidence = confidence
        self.is_authorized = is_authorized
        self.consecutive_count = 1
        self.is_confirmed = False
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.event_logged = False
        self.last_event_time = 0.0

    def update(self, box: List[int], name: str, confidence: float, is_authorized: bool, required_revisions: int):
        self.box = box
        self.name = name
        self.confidence = confidence
        self.is_authorized = is_authorized
        self.last_seen = time.time()
        self.consecutive_count += 1

        if self.consecutive_count >= required_revisions:
            self.is_confirmed = True


class DetectionTracker:
    """
    Rastreador de rostros por solapamiento (IoU) o proximidad espacial.
    Aplica el filtro de 3 revisiones consecutivas antes de confirmar y registrar eventos.
    """

    def __init__(
        self,
        required_revisions: int = REQUIRED_CONSECUTIVE_REVISIONS,
        cooldown_sec: float = EVENT_COOLDOWN_SECONDS,
    ):
        self.required_revisions = required_revisions
        self.cooldown_sec = cooldown_sec
        self.tracks: Dict[int, TrackedFace] = {}
        self.next_track_id = 1

    def _iou(self, boxA: List[int], boxB: List[int]) -> float:
        """Calcula la intersección sobre unión (IoU) entre dos bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

        union_area = float(boxA_area + boxB_area - inter_area)
        return (inter_area / union_area) if union_area > 0 else 0.0

    def _center_distance(self, boxA: List[int], boxB: List[int]) -> float:
        """Calcula la distancia euclidiana entre centros de cajas."""
        cA_x = (boxA[0] + boxA[2]) / 2.0
        cA_y = (boxA[1] + boxA[3]) / 2.0
        cB_x = (boxB[0] + boxB[2]) / 2.0
        cB_y = (boxB[1] + boxB[3]) / 2.0
        return ((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2) ** 0.5

    def update(self, raw_detections: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Procesa una lista de detecciones de la inferencia actual.
        raw_detections: [{"name": str, "confidence": float, "is_authorized": bool, "box": [x1, y1, x2, y2]}]

        Retorna:
          1. display_detections: Lista enriquecida con estado de confirmación ('Validando...' o 'Confirmado')
             para dibujar en el video y mostrar en UI.
          2. confirmed_events: Lista de eventos que alcanzaron confirmación en esta revisión para guardar en BD.
        """
        now = time.time()
        matched_track_ids = set()
        display_detections = []
        confirmed_events = []

        for det in raw_detections:
            box = det["box"]
            name = det["name"]
            conf = det["confidence"]
            is_auth = det["is_authorized"]

            # Intentar asociar con un track existente por IoU o distancia al centro
            best_track_id = None
            best_iou = 0.20  # Umbral mínimo de solapamiento

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._iou(box, track.box)
                dist = self._center_distance(box, track.box)

                # Coincide si hay solapamiento o si la distancia al centro es pequeña (< 100 px)
                if iou > best_iou or (iou > 0.05 and dist < 100):
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                track = self.tracks[best_track_id]
                matched_track_ids.add(best_track_id)
                track.update(box, name, conf, is_auth, self.required_revisions)
            else:
                # Nueva detección (revisión 1)
                new_track = TrackedFace(self.next_track_id, box, name, conf, is_auth)
                self.tracks[self.next_track_id] = new_track
                matched_track_ids.add(self.next_track_id)
                self.next_track_id += 1
                track = new_track

            # Comprobar si corresponde emitir evento en la base de datos (Criterio 4)
            if track.is_confirmed:
                can_log_event = (not track.event_logged) or ((now - track.last_event_time) > self.cooldown_sec)
                if can_log_event:
                    event_data = {
                        "is_recognized": track.is_authorized,
                        "person_name": track.name if track.is_authorized else "Desconocido",
                        "confidence": track.confidence,
                        "timestamp": datetime.utcnow()
                    }
                    confirmed_events.append(event_data)
                    track.event_logged = True
                    track.last_event_time = now

            # Metadatos para la interfaz y overlay visual
            display_detections.append({
                "track_id": track.track_id,
                "name": track.name,
                "confidence": track.confidence,
                "is_authorized": track.is_authorized,
                "is_confirmed": track.is_confirmed,
                "consecutive_count": track.consecutive_count,
                "required_count": self.required_revisions,
                "box": track.box,
                "status_text": (
                    f"{track.name} ({track.confidence}%)"
                    if track.is_confirmed
                    else f"Validando... ({track.consecutive_count}/{self.required_revisions})"
                )
            })

        # Descartar tracks que no aparecieron en esta revisión (Criterio 3)
        # Si un rostro se detectó 1 o 2 veces y desaparece, se elimina sin generar alerta ni evento.
        unmatched_ids = [t_id for t_id in self.tracks if t_id not in matched_track_ids]
        for t_id in unmatched_ids:
            del self.tracks[t_id]

        return display_detections, confirmed_events

    def reset(self):
        """Reinicia el estado de todos los tracks."""
        self.tracks.clear()
        self.next_track_id = 1


def save_confirmed_event_to_db(event_data: dict) -> Optional[int]:
    """Guarda automáticamente un evento de detección confirmada en la base de datos SQLite (HU06 / HU08)."""
    try:
        with SessionLocal() as db:
            new_event = Event(
                is_recognized=event_data.get("is_recognized", False),
                person_name=event_data.get("person_name"),
                confidence=event_data.get("confidence"),
                timestamp=event_data.get("timestamp", datetime.utcnow())
            )
            db.add(new_event)
            db.commit()
            db.refresh(new_event)
            return new_event.id
    except Exception as e:
        print(f"[DetectionTracker] Error guardando evento confirmado en BD: {e}")
        return None
