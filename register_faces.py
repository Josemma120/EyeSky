"""
register_faces.py
Procesa las fotos de la carpeta 'dataset/' y genera los vectores biométricos (embeddings)
de cada persona usando ArcFace. Guarda el catálogo en 'data/known_faces.pkl'.
"""

import os
import pickle
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = Path("dataset")
OUTPUT_FILE = Path("data/known_faces.pkl")

def init_arcface():
    """Inicializa el modelo de detección y ArcFace."""
    print("[1/3] Cargando modelo ArcFace (buffalo_sc)...")
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def process_dataset(app):
    """Recorre las carpetas de 'dataset' y extrae los embeddings."""
    if not DATASET_DIR.exists():
        print(f"Error: La carpeta '{DATASET_DIR}' no existe.")
        return {}

    known_faces = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Cada subcarpeta en dataset/ es el nombre de una persona
    person_folders = [f for f in DATASET_DIR.iterdir() if f.is_dir()]

    if not person_folders:
        print(f"No se encontraron carpetas de personas en '{DATASET_DIR}'.")
        print("Crea una carpeta por persona, ej: dataset/Emmanuel/foto1.jpg")
        return {}

    print(f"[2/3] Procesando {len(person_folders)} personas encontradas...")

    for folder in person_folders:
        person_name = folder.name
        embeddings = []

        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in valid_extensions:
                continue

            # Cargar imagen
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [!] No se pudo leer la imagen: {img_path.name}")
                continue

            # Detectar rostro y extraer embedding de ArcFace
            faces = app.get(img)
            if not faces:
                print(f"  [!] No se detectó rostro en: {person_name}/{img_path.name}")
                continue

            # Tomar el rostro principal
            emb = faces[0].embedding
            embeddings.append(emb)
            print(f"  ✓ {person_name}: Rostro extraído de {img_path.name}")

        if embeddings:
            # Si hay varias fotos, promediamos y normalizamos el vector resultante
            mean_embedding = np.mean(embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            known_faces[person_name] = mean_embedding
            print(f"  ==> Persona '{person_name}' registrada exitosamente ({len(embeddings)} foto(s))\n")
        else:
            print(f"  [X] No se pudo registrar a '{person_name}' (sin rostros válidos)\n")

    return known_faces

def save_catalogo(known_faces):
    """Guarda el diccionario de embeddings en data/known_faces.pkl."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(known_faces, f)
    print(f"[3/3] Catálogo biométrico guardado en: {OUTPUT_FILE}")
    print(f"Total de personas registradas: {len(known_faces)}")
    for name in known_faces:
        print(f" - {name}")

if __name__ == "__main__":
    print("=== REGISTRO DE ROSTROS CON ARCFACE ===")
    app = init_arcface()
    catalog = process_dataset(app)
    if catalog:
        save_catalogo(catalog)
        print("\n¡Listo! Ahora puedes ejecutar 'camera_recognition.py' para probar la cámara.")
    else:
        print("\nNo se registraron personas. Agrega fotos a 'dataset/' y vuelve a intentar.")

