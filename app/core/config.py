import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "EyeSky")
    PORT: int = int(os.getenv("PORT", 8000))
    # Umbral único de similitud (HU05): el valor efectivo del sistema es
    # 0.55; todos los servicios lo leen desde aquí.
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.55))
    # Confirmación de detecciones (HU06): ciclos consecutivos de análisis
    # dentro de la ventana para confirmar, y enfriamiento entre alertas.
    DETECTION_CONFIRM_FRAMES: int = int(os.getenv("DETECTION_CONFIRM_FRAMES", 3))
    DETECTION_CONFIRM_WINDOW_SECONDS: float = float(os.getenv("DETECTION_CONFIRM_WINDOW_SECONDS", 2.0))
    DETECTION_COOLDOWN_SECONDS: float = float(os.getenv("DETECTION_COOLDOWN_SECONDS", 30))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/eyesky.db")

settings = Settings()

