import sqlite3
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

db_path = Path("data")
db_path.mkdir(exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def ensure_schema_migrations():
    """Garantiza que la base de datos SQLite contenga las columnas requeridas por Taiga."""
    if "sqlite" in settings.DATABASE_URL:
        db_file = Path("data/eyesky.db")
        if db_file.exists():
            try:
                conn = sqlite3.connect(str(db_file))
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(persons)")
                cols = [c[1] for c in cur.fetchall()]

                if cols:
                    if "unique_code" not in cols:
                        cur.execute("ALTER TABLE persons ADD COLUMN unique_code TEXT DEFAULT ''")
                    if "role" not in cols:
                        cur.execute("ALTER TABLE persons ADD COLUMN role TEXT DEFAULT 'Residente'")
                    if "domicile" not in cols:
                        cur.execute("ALTER TABLE persons ADD COLUMN domicile TEXT DEFAULT 'Domicilio 1'")
                    if "status" not in cols:
                        cur.execute("ALTER TABLE persons ADD COLUMN status TEXT DEFAULT 'Pendiente'")
                    conn.commit()
                conn.close()
            except Exception as e:
                print(f"[Database] Migración ligera: {e}")
