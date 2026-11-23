"""
app/services/detection_tracker.py
Seguimiento de detecciones para la HU06: una detección solo se confirma
después de mantenerse durante un número de ciclos consecutivos de análisis.

Lógica pura, sin dependencias de cámara ni de base de datos, para que la
regla de confirmación pueda evaluarse mediante pruebas del sistema.

El tracker se alimenta SOLO en los ciclos de inferencia (un ciclo de
análisis cada N fotogramas), nunca en cada fotograma, para no contar
cuadros con detecciones obsoletas en caché.

Limitación conocida: los rostros desconocidos se siguen por proximidad del
centroide del recuadro entre ciclos; dos desconocidos que se crucen pueden
confundirse entre sí.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Track:
    """Estado de seguimiento de una identidad detectada."""

    key: str
    name: str
    is_recognized: bool
    confidence: float
    hits: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    confirmed_at: Optional[float] = None
    centroid: tuple = (0.0, 0.0)
    box: tuple = (0, 0, 0, 0)

    @property
    def status(self) -> str:
        return "confirmed" if self.confirmed_at is not None else "pending"

    @property
    def detection_count(self) -> int:
        return self.hits


class DetectionTracker:
    """
    Confirma detecciones que se mantienen durante varios ciclos de análisis
    consecutivos dentro de una ventana de tiempo, y suprime alertas repetidas
    de la misma identidad durante un periodo de enfriamiento (cooldown).
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        confirm_window_seconds: float = 2.0,
        cooldown_seconds: float = 30.0,
        unknown_match_distance: float = 60.0,
    ):
        self.confirm_frames = confirm_frames
        self.confirm_window_seconds = confirm_window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.unknown_match_distance = unknown_match_distance
        self._tracks: Dict[str, Track] = {}
        self._last_alert: Dict[str, float] = {}
        self._unknown_counter = 0

    @staticmethod
    def _centroid(box) -> tuple:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a: tuple, b: tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _sweep(self, now: float) -> None:
        """Elimina los tracks que desaparecieron antes de cumplir la condición."""
        expired = [
            key
            for key, track in self._tracks.items()
            if now - track.last_seen > self.confirm_window_seconds
        ]
        for key in expired:
            del self._tracks[key]
        # La memoria de enfriamiento caduca sola.
        stale_alerts = [
            key
            for key, alert_at in self._last_alert.items()
            if now - alert_at > self.cooldown_seconds
        ]
        for key in stale_alerts:
            del self._last_alert[key]

    def _match_unknown(self, centroid: tuple) -> Optional[Track]:
        """Busca el track de desconocido más cercano al centroide dado."""
        best = None
        best_distance = self.unknown_match_distance
        for track in self._tracks.values():
            if track.is_recognized:
                continue
            distance = self._distance(centroid, track.centroid)
            if distance <= best_distance:
                best = track
                best_distance = distance
        return best

    def lookup(self, name: str, box) -> Optional[Track]:
        """Devuelve el track que corresponde a una detección, sin modificarlo."""
        if name != "Desconocido":
            return self._tracks.get(f"person:{name}")
        return self._match_unknown(self._centroid(box))

    def get_tracks(self) -> List[Track]:
        """Foto actual de los tracks activos (para el estado en vivo)."""
        return list(self._tracks.values())

    def observe(self, detections: List[dict], now: Optional[float] = None) -> List[dict]:
        """
        Registra un ciclo de análisis y devuelve los eventos de las
        detecciones que acaban de confirmarse en este ciclo.

        Cada detección es un dict con: name, confidence, is_authorized, box.
        """
        if now is None:
            now = time.time()

        self._sweep(now)
        newly_confirmed: List[dict] = []

        for detection in detections:
            name = detection.get("name", "Desconocido")
            is_recognized = bool(detection.get("is_authorized", name != "Desconocido"))
            confidence = float(detection.get("confidence", 0.0))
            box = tuple(detection.get("box", (0, 0, 0, 0)))
            centroid = self._centroid(box)

            if is_recognized:
                key = f"person:{name}"
                track = self._tracks.get(key)
            else:
                track = self._match_unknown(centroid)
                if track is None:
                    self._unknown_counter += 1
                    key = f"unknown:{self._unknown_counter}"
                else:
                    key = track.key

            if track is None:
                track = Track(
                    key=key,
                    name=name,
                    is_recognized=is_recognized,
                    confidence=confidence,
                    hits=1,
                    first_seen=now,
                    last_seen=now,
                    centroid=centroid,
                    box=box,
                )
                self._tracks[key] = track
            else:
                # La ventana de confirmación expiró sin confirmar: el conteo
                # se reinicia y la detección actual es el primer ciclo nuevo.
                if track.confirmed_at is None and (now - track.first_seen) > self.confirm_window_seconds:
                    track.hits = 1
                    track.first_seen = now
                else:
                    track.hits += 1
                track.last_seen = now
                track.name = name
                track.is_recognized = is_recognized
                track.confidence = confidence
                track.centroid = centroid
                track.box = box

            if track.confirmed_at is None and track.hits >= self.confirm_frames:
                track.confirmed_at = now
                last_alert = self._last_alert.get(key)
                if last_alert is None or (now - last_alert) >= self.cooldown_seconds:
                    self._last_alert[key] = now
                    newly_confirmed.append(
                        {
                            "key": key,
                            "name": track.name,
                            "is_recognized": track.is_recognized,
                            "confidence": track.confidence,
                            "detection_count": track.hits,
                            "first_seen": datetime.fromtimestamp(track.first_seen),
                            "confirmed_at": datetime.fromtimestamp(track.confirmed_at),
                        }
                    )

        return newly_confirmed
