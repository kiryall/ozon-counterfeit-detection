# api/v1/endpoints/prediction.py
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from models.prediction import (
    BatchPredictionResponse,
    DataFrameProcessor,
)
from services.prediction_service import PredictionProcessor
from core.logging import setup_logging

# Настройка логирования
logger = setup_logging(log_file="batch_prediction.log", console=True, remove_file=True, logger_name="batch_prediction")

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/upload-batch")
async def predict_batch_multiple(
    images: list[UploadFile] = File(..., description="Uploaded image files"),
    dataframe: UploadFile = File(..., description="Uploaded CSV/Excel file"),
) -> BatchPredictionResponse:
    """
    Batch prediction endpoint for multiple images and dataframe rows.

    Args:
        images: List of uploaded image files
        dataframe: Uploaded CSV or Excel file with data rows

    Returns:
        BatchPredictionResponse with predictions for all rows
    """
    from main import model, multimodal_processor

    logger.info(f"Received batch prediction request with {len(images)} images")

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
        image_bytes_list = []
        for img_file in images:
            image_bytes_list.append(await img_file.read())

        dataframe_bytes = await dataframe.read()

        # Load dataframe
        df = DataFrameProcessor.load_dataframe_from_bytes(
            dataframe_bytes, dataframe.filename
        )
        DataFrameProcessor.validate_dataframe(df)

        logger.info(f"Dataframe loaded: {df.shape}, Images: {len(image_bytes_list)}")

        # Initialize prediction processor
        predictor = PredictionProcessor(model, multimodal_processor)

        # Perform batch prediction
        response = predictor.predict_batch(
            image_bytes_list=image_bytes_list,
            dataframe=df,
            image_filenames=[img.filename for img in images],
        )
        logger.info(f"Batch prediction completed: {response.count} predictions")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )
