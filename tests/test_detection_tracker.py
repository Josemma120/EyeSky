"""
tests/test_detection_tracker.py
Pruebas del sistema para la regla de confirmación de la HU06:
una detección solo genera una alerta después de mantenerse durante
3 ciclos consecutivos de análisis (aprox. 1 segundo).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.detection_tracker import DetectionTracker


def det(name="Ana", confidence=92.5, box=(100, 100, 200, 200)):
    return {
        "name": name,
        "confidence": confidence,
        "is_authorized": name != "Desconocido",
        "box": list(box),
    }


def test_un_solo_hit_no_confirma():
    tracker = DetectionTracker()
    events = tracker.observe([det()], now=1000.0)
    assert events == []
    tracks = tracker.get_tracks()
    assert len(tracks) == 1
    assert tracks[0].status == "pending"
    assert tracks[0].detection_count == 1


def test_tres_ciclos_consecutivos_confirman():
    tracker = DetectionTracker()
    assert tracker.observe([det()], now=1000.0) == []
    assert tracker.observe([det()], now=1000.3) == []
    events = tracker.observe([det()], now=1000.6)
    assert len(events) == 1
    event = events[0]
    assert event["name"] == "Ana"
    assert event["is_recognized"] is True
    assert event["detection_count"] == 3
    assert event["first_seen"].timestamp() == 1000.0
    assert event["confirmed_at"].timestamp() == 1000.6
    assert tracker.get_tracks()[0].status == "confirmed"


def test_desaparicion_antes_de_tres_no_genera_alerta_y_poda_el_track():
    tracker = DetectionTracker()
    tracker.observe([det()], now=1000.0)
    tracker.observe([det()], now=1000.3)
    # La detección desaparece: ciclos sin detecciones.
    assert tracker.observe([], now=1000.6) == []
    assert tracker.observe([], now=1000.9) == []
    # Pasada la ventana, el track pendiente se poda.
    assert tracker.observe([], now=1003.0) == []
    assert tracker.get_tracks() == []


def test_expiracion_de_ventana_reinicia_el_conteo():
    tracker = DetectionTracker()
    tracker.observe([det()], now=1000.0)
    tracker.observe([det()], now=1000.5)
    # El tercer ciclo llega fuera de la ventana de 2 s: no confirma y
    # el conteo se reinicia con este ciclo como primero.
    assert tracker.observe([det()], now=1002.5) == []
    assert tracker.get_tracks()[0].detection_count == 1
    assert tracker.get_tracks()[0].status == "pending"
    assert tracker.observe([det()], now=1002.8) == []
    events = tracker.observe([det()], now=1003.1)
    assert len(events) == 1
    assert events[0]["detection_count"] == 3


def test_cooldown_suprime_alertas_repetidas_de_la_misma_identidad():
    tracker = DetectionTracker()
    tracker.observe([det()], now=1000.0)
    tracker.observe([det()], now=1000.3)
    assert len(tracker.observe([det()], now=1000.6)) == 1
    # Desaparece y reaparece dentro del enfriamiento (30 s): se confirma
    # de nuevo pero no genera otra alerta.
    tracker.observe([], now=1005.0)
    tracker.observe([det()], now=1010.0)
    tracker.observe([det()], now=1010.3)
    assert tracker.observe([det()], now=1010.6) == []
    assert tracker.get_tracks()[0].status == "confirmed"
    # Pasado el enfriamiento, una nueva confirmación sí genera alerta.
    tracker.observe([], now=1040.0)
    tracker.observe([det()], now=1045.0)
    tracker.observe([det()], now=1045.3)
    events = tracker.observe([det()], now=1045.6)
    assert len(events) == 1
    assert events[0]["name"] == "Ana"


def test_identidades_conocidas_y_desconocidas_se_trackean_por_separado():
    tracker = DetectionTracker()
    unknown_box = (400, 100, 500, 200)
    tracker.observe([det("Ana"), det("Desconocido", 45.0, unknown_box)], now=1000.0)
    tracker.observe([det("Ana"), det("Desconocido", 45.0, unknown_box)], now=1000.3)
    events = tracker.observe([det("Ana"), det("Desconocido", 45.0, unknown_box)], now=1000.6)
    assert len(events) == 2
    assert {e["name"] for e in events} == {"Ana", "Desconocido"}
    assert len(tracker.get_tracks()) == 2


def test_dos_desconocidos_en_posiciones_distintas_son_tracks_distintos():
    tracker = DetectionTracker()
    box_a = (100, 100, 200, 200)
    box_b = (400, 100, 500, 200)
    tracker.observe(
        [det("Desconocido", 40.0, box_a), det("Desconocido", 42.0, box_b)],
        now=1000.0,
    )
    tracker.observe(
        [det("Desconocido", 40.0, box_a), det("Desconocido", 42.0, box_b)],
        now=1000.3,
    )
    events = tracker.observe(
        [det("Desconocido", 40.0, box_a), det("Desconocido", 42.0, box_b)],
        now=1000.6,
    )
    assert len(events) == 2
    assert len(tracker.get_tracks()) == 2


def test_un_desconocido_que_se_mueve_poco_mantiene_el_mismo_track():
    tracker = DetectionTracker()
    tracker.observe([det("Desconocido", 40.0, (100, 100, 200, 200))], now=1000.0)
    tracker.observe([det("Desconocido", 41.0, (105, 102, 205, 202))], now=1000.3)
    events = tracker.observe([det("Desconocido", 42.0, (110, 104, 210, 204))], now=1000.6)
    assert len(events) == 1
    assert len(tracker.get_tracks()) == 1
    assert events[0]["detection_count"] == 3


def test_lookup_devuelve_estado_y_conteo_para_el_panel():
    tracker = DetectionTracker()
    tracker.observe([det()], now=1000.0)
    track = tracker.lookup("Ana", (100, 100, 200, 200))
    assert track is not None
    assert track.status == "pending"
    assert track.detection_count == 1
    tracker.observe([det()], now=1000.3)
    tracker.observe([det()], now=1000.6)
    track = tracker.lookup("Ana", (100, 100, 200, 200))
    assert track.status == "confirmed"
    assert track.detection_count == 3
    assert tracker.lookup("Luis", (100, 100, 200, 200)) is None
