from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.db.database import engine, Base
from app.routers import persons, recognize, events, stream

# Crear automáticamente las tablas en SQLite al iniciar
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Sistema Inteligente de Seguridad y Reconocimiento Facial (Edge-to-Cloud)",
    version="0.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conectar routers
app.include_router(stream.router)
app.include_router(persons.router)
app.include_router(recognize.router)
app.include_router(events.router)

# Servir archivos estáticos del panel web
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    """Sirve la interfaz web del sistema."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "project": settings.PROJECT_NAME,
        "status": "online",
        "docs_url": "/docs",
        "stream_url": "/api/stream/video_feed"
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "project": settings.PROJECT_NAME}
