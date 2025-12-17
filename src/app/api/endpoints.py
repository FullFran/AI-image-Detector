import io
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ai_detector.visualization.plots import create_report_plot
from app.schemas.dtos import AnalysisResult
from app.services.classification import ClassificationService

router = APIRouter()


@router.post("/classify", response_model=AnalysisResult)
async def classify_image(file: UploadFile = File(...)):
    service = ClassificationService()
    try:
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)

        result = service.predict(image_np)

        end_time = time.time()

        return AnalysisResult(
            filename=file.filename,
            prediction=result["prediction"],
            probability_real=result["probability_real"],
            probability_fake=result["probability_fake"],
            features=result["features"],
            processing_time_ms=(end_time - start_time) * 1000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-visual")
async def analyze_visual(file: UploadFile = File(...)):
    """
    Devuelve una imagen con el análisis visual (Grid: Original + Gradientes)
    y el resultado en el título.
    """
    service = ClassificationService()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)

        # 1. Obtener predicción
        result = service.predict(image_np)

        # 2. Re-analizar para obtener las matrices (gradients) que no estan en 'features'
        # El metodo predict devuelve 'features', pero necesitamos las matrices completas
        # que devuelve detector.analyze().
        # Podemos acceder al detector del servicio.
        full_analysis = service.detector.analyze(image_np)

        prediction_label = result["prediction"]  # REAL o IA/FAKE
        probability = (
            result["probability_fake"]
            if "FAKE" in prediction_label
            else result["probability_real"]
        )
        title = f"Prediction: {prediction_label} ({probability:.1%})"

        # 3. Generar plot
        buf = create_report_plot(image_np, full_analysis, title)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
