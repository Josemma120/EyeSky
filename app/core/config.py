import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "EyeSky")
    PORT: int = int(os.getenv("PORT", 8000))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/eyesky.db")

settings = Settings()

