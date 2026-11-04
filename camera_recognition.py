"""
camera_recognition.py
Abre la cámara web en tiempo real, detecta rostros, extrae sus vectores con ArcFace
y los compara contra el catálogo 'data/known_faces.pkl' generado por register_faces.py.
"""

import os
# Limitar hilos de procesamiento en CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import pickle
import time
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Configuración de la fuente de video:
# - Pon 0 o 1 si usas una app como Iriun Webcam o Camo (USB / Virtual Cam).
# - O pon la URL de streaming si usas una app de IP Camera (ej: "http://192.168.1.50:8080/video").
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

CATALOG_PATH = Path("data/known_faces.pkl")

# Umbral de similitud de ArcFace:
SIMILARITY_THRESHOLD = 0.55
# Frecuencia de inferencia: procesar 1 de cada N cuadros (5 = 4 veces por seg, ideal para CPU)
INFERENCE_INTERVAL = int(os.getenv("INFERENCE_INTERVAL", "5"))

def load_known_faces():
    """Carga los vectores de las personas registradas."""
    if not CATALOG_PATH.exists():
        print(f"Error: No se encontró el catálogo '{CATALOG_PATH}'.")
        print("Primero ejecuta: python register_faces.py")
        return None
    with open(CATALOG_PATH, "rb") as f:
        known = pickle.load(f)
    print(f"Catálogo cargado con {len(known)} persona(s): {list(known.keys())}")
    return known

def init_detector():
    """Carga el detector y ArcFace optimizado para CPU."""
    print("Iniciando modelo ArcFace optimizado...")
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    # 320x320 o 480x480 acelera la detección en CPU al triple que 640x640
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app

def identify_face(face_embedding, known_faces):
    """
    Compara el embedding detectado contra todos los embeddings registrados
    mediante producto punto (similitud de coseno).
    """
    # Normalizar vector
    emb = face_embedding / np.linalg.norm(face_embedding)

    best_name = "Desconocido"
    best_score = -1.0

    for name, known_emb in known_faces.items():
        similarity = float(np.dot(emb, known_emb))
        if similarity > best_score:
            best_score = similarity
            if similarity >= SIMILARITY_THRESHOLD:
                best_name = name

    return best_name, max(best_score, 0.0)

def main():
    known_faces = load_known_faces()
    if not known_faces:
        return

    app = init_detector()

    # Abrir la cámara web o stream de red
    print(f"Conectando a la fuente de video: {CAMERA_SOURCE} ...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara web. Verifica que esté conectada y habilitada.")
        return

    print("\n--- Cámara iniciada ---")
    print("Presiona la tecla 'q' para salir.")

    fps_time = time.time()
    frame_count = 0
    cached_faces = []

    # Configurar resolución de captura para mayor fluidez
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir video de la cámara.")
            break

        frame_count += 1

        # Reducir tamaño si la cámara envía resolución gigante (ej: 1080p o 4K)
        h, w = frame.shape[:2]
        if w > 800:
            scale = 640 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Ejecutar ArcFace cada N fotogramas para no saturar la CPU
        if frame_count % INFERENCE_INTERVAL == 0:
            cached_faces = app.get(frame)

        for face in cached_faces:
            # Obtener coordenadas del recuadro
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Identificar rostro
            name, score = identify_face(face.embedding, known_faces)

            # Color: Verde para autorizados, Rojo para desconocidos
            is_authorized = (name != "Desconocido")
            color = (0, 255, 0) if is_authorized else (0, 0, 255)

            # Dibujar caja del rostro
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con el nombre y porcentaje de similitud
            label = f"{name} ({score * 100:.1f}%)"
            y_label = max(y1 - 10, 20)
            cv2.putText(
                frame,
                label,
                (x1, y_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        # Mostrar FPS en la esquina superior izquierda
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_time)
            fps_time = now
            cv2.setWindowTitle("EyeSky - Reconocimiento Facial ArcFace", f"EyeSky | FPS: {fps:.1f}")

        cv2.imshow("EyeSky - Reconocimiento Facial ArcFace", frame)

        # Salir con la tecla 'q' o ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada.")

if __name__ == "__main__":
    main()

