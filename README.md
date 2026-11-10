# EyeSky - Reconocimiento Facial en Vivo con ArcFace

> Sistema Inteligente de Seguridad y Reconocimiento Facial para Entornos Residenciales y PyMEs (Edge-to-Cloud).

---

## 📋 Requisitos Previos

- Python 3.10 o 3.11 instalado en el sistema.
- Cámara web integrada o dispositivo móvil conectado (ej. iPhone con **Iriun Webcam**).

---

## 🚀 Instalación y Puesta en Marcha

Si acabas de clonar el repositorio en una máquina nueva, sigue estos pasos en orden:

### 1. Crear el entorno virtual de Python
```powershell
python -m venv venv
```

### 2. Activar el entorno virtual
En Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```
*(Si PowerShell bloquea la ejecución de scripts, ejecuta primero: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Instalar las dependencias
```powershell
pip install -r requirements.txt
```

### 4. Configurar las variables de entorno
Crea tu archivo `.env` a partir del ejemplo:
```powershell
Copy-Item .env.example .env
```
*(Puedes ajustar aquí el umbral de similitud `SIMILARITY_THRESHOLD=0.55` y la fuente de la cámara `CAMERA_SOURCE=0`)*.

---

## 👤 Registro de Personas (Dataset)

1. Coloca las fotografías de las personas autorizadas dentro de la carpeta `dataset/`, creando una carpeta por persona:
   ```text
   dataset/
   ├── Emmanuel/
   │   ├── foto1.jpg
   │   └── foto2.jpg
   └── Janeth/
       ├── foto1.jpg
       └── foto2.jpg
   ```
2. Ejecuta el script de extracción biométrica con ArcFace:
   ```powershell
   python register_faces.py
   ```
   *(Esto generará el archivo `data/known_faces.pkl` con los vectores biométricos).*

3. Al iniciar el servidor, las personas de `dataset/` y de `known_faces.pkl` se importan automáticamente a la base de datos y aparecen en **Personas Autorizadas**. Si agregas personas con el servidor ya encendido, usa el botón **🔄 Sincronizar Dataset y Catálogo** o el endpoint `POST /api/persons/sync-dataset`.

---

## 🖥️ Ejecutar el Sistema (Panel Web)

### 1. Iniciar el servidor de desarrollo
```powershell
uvicorn app.main:app --reload --port 8000
```

### 2. Abrir la interfaz web
Ingresa desde cualquier navegador a:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📸 Ejecución alternativa por consola (Ventana OpenCV)
Si prefieres ver la cámara directamente en una ventana nativa de escritorio:
```powershell
python camera_recognition.py
```
*(Presiona la tecla **`q`** para cerrar la ventana).*

---

## 📘 Documentación de la API REST
FastAPI genera automáticamente la documentación interactiva Swagger en:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### Listado paginado y búsqueda de personas

```http
GET /api/persons/paginated?page=1&page_size=5&search=Angel
```

Parámetros opcionales:

| Parámetro | Descripción |
|-----------|-------------|
| `page` | Número de página (inicia en 1). |
| `page_size` | Registros por página (1 a 100). |
| `search` | Busca por nombre o identificador único (`DOM-001`). |
| `domicile` | Filtra por domicilio exacto. |
| `status` | Filtra por estado (`Autorizada` o `Pendiente`). |
