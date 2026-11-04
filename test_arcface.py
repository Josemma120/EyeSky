"""
test_arcface.py - Script simple para validar la lógica matemática de ArcFace
"""
import numpy as np
from app.services.arcface_service import calculate_similarity, is_match

def test_similarity_logic():
    print("--- Prueba rápida de lógica ArcFace ---")
    
    # Simular dos vectores faciales (embeddings de 512 dimensiones)
    np.random.seed(42)
    vector_persona_a = np.random.randn(512).astype(np.float32)
    
    # Vector casi idéntico (misma persona con pequeña variación)
    vector_persona_a_variacion = vector_persona_a + np.random.normal(0, 0.1, 512).astype(np.float32)
    
    # Vector de una persona distinta
    vector_persona_b = np.random.randn(512).astype(np.float32)
    
    sim_misma = calculate_similarity(vector_persona_a, vector_persona_a_variacion)
    sim_distinta = calculate_similarity(vector_persona_a, vector_persona_b)
    
    print(f"Similitud misma persona: {sim_misma:.4f} -> Coincide: {is_match(sim_misma)}")
    print(f"Similitud persona distinta: {sim_distinta:.4f} -> Coincide: {is_match(sim_distinta)}")

if __name__ == "__main__":
    test_similarity_logic()

