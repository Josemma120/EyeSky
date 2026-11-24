"""
test_hu06_confirmation.py
Pruebas automatizadas para los Criterios de Aceptación de HU06:
'Validar una detección antes de generar una alerta' (HU06).

Criterios validados:
1. Una detección aislada no genera una alerta.
2. La detección debe mantenerse durante 3 revisiones seguidas antes de confirmarse.
3. Si la detección desaparece antes de cumplir esa condición, no se genera alerta.
4. Una detección confirmada continúa con la identificación y queda guardada como un evento.
5. La regla de confirmación puede revisarse con pruebas del sistema.
"""

import unittest
from app.services.detection_tracker import (
    DetectionTracker,
    REQUIRED_CONSECUTIVE_REVISIONS,
    save_confirmed_event_to_db
)
from app.db.database import SessionLocal
from app.db.models import Event


class TestHU06ConfirmationLogic(unittest.TestCase):

    def setUp(self):
        """Inicializa un rastreador fresco antes de cada prueba."""
        self.tracker = DetectionTracker(required_revisions=3, cooldown_sec=10.0)

    def test_criterio_1_deteccion_aislada_no_genera_alerta(self):
        """Criterio 1: Una detección aislada (1 sola revisión) NO debe confirmar ni generar alerta."""
        raw_detection = [{
            "name": "Jose",
            "confidence": 92.5,
            "is_authorized": True,
            "box": [100, 100, 250, 250]
        }]

        display_dets, events = self.tracker.update(raw_detection)

        self.assertEqual(len(display_dets), 1)
        self.assertFalse(display_dets[0]["is_confirmed"], "Una detección aislada no debe estar confirmada.")
        self.assertEqual(display_dets[0]["consecutive_count"], 1)
        self.assertEqual(len(events), 0, "No debe generarse evento ni alerta en la primera revisión.")

    def test_criterio_2_dos_revisiones_no_confirman(self):
        """Criterio 2: 2 revisiones consecutivas aún NO alcanzan el umbral de 3."""
        raw_detection = [{
            "name": "Jose",
            "confidence": 93.0,
            "is_authorized": True,
            "box": [102, 101, 252, 251]
        }]

        # Revisión 1
        self.tracker.update(raw_detection)
        # Revisión 2
        display_dets, events = self.tracker.update(raw_detection)

        self.assertEqual(len(display_dets), 1)
        self.assertFalse(display_dets[0]["is_confirmed"])
        self.assertEqual(display_dets[0]["consecutive_count"], 2)
        self.assertEqual(len(events), 0, "Con 2 revisiones no debe generarse evento.")

    def test_criterio_3_desaparicion_antes_de_confirmar_descarta_alerta(self):
        """Criterio 3: Si la detección desaparece antes de 3 revisiones, se descarta sin alerta."""
        raw_detection = [{
            "name": "Desconocido",
            "confidence": 42.0,
            "is_authorized": False,
            "box": [150, 150, 300, 300]
        }]

        # Revisión 1 y 2
        self.tracker.update(raw_detection)
        self.tracker.update(raw_detection)

        # Revisión 3: El rostro desaparece de la cámara
        display_dets, events = self.tracker.update([])

        self.assertEqual(len(display_dets), 0, "No debe haber detecciones activas si el rostro desapareció.")
        self.assertEqual(len(events), 0, "No debe haberse emitido ningún evento/alerta.")
        self.assertEqual(len(self.tracker.tracks), 0, "El track pendiente debe haber sido descartado.")

    def test_criterio_2_y_4_tres_revisiones_confirman_y_emiten_evento(self):
        """Criterios 2 y 4: Al mantenerse 3 revisiones seguidas, se confirma y se emite el evento."""
        raw_detection = [{
            "name": "Angel Luna",
            "confidence": 95.0,
            "is_authorized": True,
            "box": [120, 120, 270, 270]
        }]

        # Revisión 1
        self.tracker.update(raw_detection)
        # Revisión 2
        self.tracker.update(raw_detection)
        # Revisión 3 (Confirmación)
        display_dets, events = self.tracker.update(raw_detection)

        self.assertEqual(len(display_dets), 1)
        self.assertTrue(display_dets[0]["is_confirmed"], "A la 3ra revisión consecutiva debe confirmarse.")
        self.assertEqual(display_dets[0]["consecutive_count"], 3)
        self.assertEqual(len(events), 1, "Debe emitirse exactamente 1 evento al confirmarse.")
        self.assertEqual(events[0]["person_name"], "Angel Luna")
        self.assertTrue(events[0]["is_recognized"])

    def test_criterio_4_guardado_automatico_en_base_de_datos(self):
        """Criterio 4: El evento confirmado se guarda físicamente en la tabla 'events' de SQLite."""
        event_data = {
            "is_recognized": True,
            "person_name": "Test_HU06_Persona",
            "confidence": 98.7
        }

        event_id = save_confirmed_event_to_db(event_data)
        self.assertIsNotNone(event_id, "El evento debe haberse persistido en la base de datos con un ID asignado.")

        # Verificar en SQLite
        with SessionLocal() as db:
            saved_event = db.query(Event).filter(Event.id == event_id).first()
            self.assertIsNotNone(saved_event)
            self.assertEqual(saved_event.person_name, "Test_HU06_Persona")
            self.assertTrue(saved_event.is_recognized)
            self.assertAlmostEqual(saved_event.confidence, 98.7, places=1)

            # Limpiar el registro de prueba
            db.delete(saved_event)
            db.commit()

    def test_cooldown_evita_saturacion_de_eventos_repetidos(self):
        """Verifica que una persona ya confirmada no genere alertas repetidas por cada frame."""
        raw_detection = [{
            "name": "Jose",
            "confidence": 91.0,
            "is_authorized": True,
            "box": [100, 100, 250, 250]
        }]

        # 3 revisiones para confirmar
        self.tracker.update(raw_detection)
        self.tracker.update(raw_detection)
        _, events_frame3 = self.tracker.update(raw_detection)
        self.assertEqual(len(events_frame3), 1, "Emite evento en la confirmación.")

        # Revisiones 4, 5 y 6 (la persona continúa parada frente a la cámara)
        _, events_frame4 = self.tracker.update(raw_detection)
        _, events_frame5 = self.tracker.update(raw_detection)

        self.assertEqual(len(events_frame4), 0, "No debe spamear eventos en revisiones subsiguientes inmediatas.")
        self.assertEqual(len(events_frame5), 0, "No debe spamear eventos en revisiones subsiguientes inmediatas.")


if __name__ == "__main__":
    unittest.main()
