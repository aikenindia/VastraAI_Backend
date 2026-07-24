import os
import time
from dotenv import load_dotenv
load_dotenv()  # must run before any "from routes..." import below, since
                # those modules read os.environ.get(...) at import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.model_loader import download_models

print("=" * 50)
print("🚀 KGD OB Prediction API starting...")
print(f"   PORT: {os.environ.get('PORT', '8000')}")
print("=" * 50)

# Download models FIRST before any route imports
download_models()

from routes.spec    import router as spec_router
from routes.image   import router as image_router
from routes.extract import router as extract_router
from routes.debug   import router as debug_router

app = FastAPI(
    title="Ken Global Designs — OB Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(spec_router,    prefix="/spec")
app.include_router(image_router,   prefix="/image")
app.include_router(extract_router, prefix="/spec")
app.include_router(debug_router,   prefix="/debug")

_start_time = time.time()

@app.get("/")
def home():
    uptime = round(time.time() - _start_time)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "routes": {
            "specification": "POST /spec/predict",
            "image":         "POST /image/predict",
            "extract_pdf":   "POST /spec/extract-pdf",
            "extract_text":  "POST /spec/extract",
            "debug_labels":  "GET /debug/labels",
            "docs":          "/docs"
        }
    }

@app.get("/health")
def health():
    """Railway health check endpoint"""
    return JSONResponse({"status": "ok"}, status_code=200)
