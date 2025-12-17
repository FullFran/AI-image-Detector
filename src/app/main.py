from fastapi import FastAPI

from app.api.endpoints import router as api_router

app = FastAPI(title="AI Image Detector API", version="1.0.0")

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "AI Image Detector API is running. Docs at /docs"}
