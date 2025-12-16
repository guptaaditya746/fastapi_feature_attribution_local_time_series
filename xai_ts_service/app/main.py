from fastapi import FastAPI
from app.api.v1 import router as api_router
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="XAI Time-Series Service", version="1.0.0")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "XAI Time-Series Service is running"}
