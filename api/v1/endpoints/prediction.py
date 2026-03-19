# api/v1/endpoints/prediction.py
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from core.logging import setup_logging
from models.prediction import (
    PredictionResponse,
    BatchPredictionResponse,
    ImageProcessor,
    DataFrameProcessor
    )
from services.prediction_service import PredictionProcessor

# Настройка логирования
logger = setup_logging(log_file="prediction.log", console=True, remove_file=True, logger_name="prediction")

router = APIRouter(tags=["prediction"])


@router.post("/upload", response_model=PredictionResponse | BatchPredictionResponse)
async def predict_upload(
    image: UploadFile = File(..., description="Uploaded image file"),
    dataframe: UploadFile = File(..., description="Uploaded CSV/Excel file"),
):
    """Универсальный эндпоинт для предсказания на одном или нескольких изображениях.

    Выполняет предсказание на основе загруженных изображения и данных:
    - Если в dataframe 1 строка и 1 изображение: одиночное предсказание
    - Если в dataframe несколько строк и несколько изображений: пакетное предсказание

    Args:
        image: Загруженный файл изображения (JPEG, PNG и др.).
        dataframe: Загруженный CSV или Excel файл с данными строк.

    Returns:
        PredictionResponse для одиночного предсказания.
        BatchPredictionResponse для пакетного предсказания.

    Raises:
        HTTPException 503: Модель не загружена.
        HTTPException 400: Ошибка валидации данных.
        HTTPException 500: Внутренняя ошибка сервера.
    """
    from main import model, multimodal_processor

    logger.info(
        f"Received prediction request with image: {image.filename}, dataframe: {dataframe.filename}"
    )

    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please try again later.",
        )

    if multimodal_processor is None:
        logger.error("Multimodal processor not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multimodal processor is not loaded. Please try again later.",
        )

    try:
        # Read uploaded files
        image_bytes = await image.read()
        dataframe_bytes = await dataframe.read()

        # Validate image format using ImageProcessor
        pil_image = ImageProcessor.load_image_from_bytes(image_bytes)
        if not ImageProcessor.validate_image_format(pil_image):
            raise ValueError(
                f"Invalid image format: {pil_image.format}. Supported: JPEG, PNG, BMP, GIF, TIFF"
            )
        logger.info(f"Image validated: {image.filename} ({pil_image.format})")

        # Load dataframe
        df = DataFrameProcessor.load_dataframe_from_bytes(
            dataframe_bytes, dataframe.filename
        )
        DataFrameProcessor.validate_dataframe(df)

        logger.info(f"Dataframe loaded: {df.shape}")

        # Initialize prediction processor
        predictor = PredictionProcessor(model, multimodal_processor)

        logger.info("Performing single prediction")
        row_dict = DataFrameProcessor.get_row_as_dict(df, 0)

        response = predictor.predict_single(
            image_bytes=image_bytes,
            dataframe_row=row_dict,
            image_filename=image.filename,
        )
        logger.info(
            f"Single prediction result: {response.prediction} (confidence: {response.confidence})"
        )
        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )
