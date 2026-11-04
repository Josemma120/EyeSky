# EyeSky - Reconocimiento Facial en Vivo con ArcFace

Sistema Inteligente de Seguridad y Reconocimiento Facial (Edge-to-Cloud).

---

## 🚀 Cómo iniciar el sistema con interfaz web

### 1. Activar el entorno virtual:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Registrar las fotos (si agregas nuevas personas a `dataset/`):
```powershell
python register_faces.py
```

### 3. Iniciar el servidor web de EyeSky:
```powershell
uvicorn app.main:app --reload --port 8000
```

### 4. Abrir la interfaz web en tu navegador:
Abre cualquier navegador (Chrome, Edge, Safari en tu PC o teléfono) e ingresa a:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🖥️ Características del Panel Web

- **Streaming en vivo:** Transmisión directa desde tu cámara (o iPhone con Iriun Webcam) con los recuadros verdes (autorizados) y rojos (desconocidos).
- **Indicador en tiempo real:** Muestra inmediatamente quién está frente a la cámara con el porcentaje de coincidencia.
- **Lista de autorizados:** Despliega todas las personas registradas en el catálogo biométrico de ArcFace.
- **Historial de detecciones:** Bitácora en vivo con la hora exacta en que cada persona fue reconocida.
- **Documentación de la API:** Disponible en [http://localhost:8000/docs](http://localhost:8000/docs).
