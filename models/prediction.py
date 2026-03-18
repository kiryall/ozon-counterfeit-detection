# models/prediction.py
from pandas import DataFrame
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from PIL import Image
from io import BytesIO
import numpy as np


class PredictionType(str, Enum):
    """Enum for prediction types"""
    FAKE = "FAKE"
    REAL = "REAL"


class SinglePredictionRequest(BaseModel):
    """
    Request model for single prediction (1 image + 1 row from dataframe).
    Accepts uploaded files via form data.
    """
    # Note: In FastAPI, file uploads are handled separately via UploadFile
    # This model is used for response validation only
    pass


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch prediction (multiple images + multiple rows).
    Accepts uploaded files via form data.
    """
    # Note: In FastAPI, file uploads are handled separately via UploadFile
    # This model is used for response validation only
    pass


class PredictionResponse(BaseModel):
    """
    Response model for single prediction.
    """
    prediction: PredictionType = Field(..., description="Prediction result: FAKE or REAL")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    item_id: Optional[str] = Field(None, description="Item ID from dataframe")


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch predictions.
    """
    predictions: List[PredictionType] = Field(..., description="List of prediction results")
    confidences: List[float] = Field(..., description="List of confidence scores")
    item_ids: Optional[List[str]] = Field(None, description="List of item IDs from dataframe")
    count: int = Field(..., description="Number of predictions")
    message: Optional[str] = Field(None, description="Additional message")


class ImageProcessor:
    """Helper class for processing uploaded images"""

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
        """Convert bytes to PIL Image"""
        try:
            image = Image.open(BytesIO(image_bytes))
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from bytes: {str(e)}")

    @staticmethod
    def validate_image_format(image: Image.Image) -> bool:
        """Validate image format"""
        valid_formats = {'JPEG', 'PNG', 'BMP', 'GIF', 'TIFF'}
        return image.format in valid_formats

    @staticmethod
    def get_image_array(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)


class DataFrameProcessor:
    """Helper class for processing uploaded dataframe files"""

    @staticmethod
    def load_dataframe_from_bytes(file_bytes: bytes, filename: str) -> DataFrame:
        """Load DataFrame from uploaded file"""
        try:
            if filename.endswith('.csv'):
                df = DataFrame.from_csv(BytesIO(file_bytes), index=False)
                # Handle case where pandas reads first column as index
                if df.index.name is not None:
                    df = df.reset_index()
                return df
            elif filename.endswith(('.xlsx', '.xls')):
                df = DataFrame.from_excel(BytesIO(file_bytes))
                return df
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            raise ValueError(f"Failed to load dataframe from file: {str(e)}")

    @staticmethod
    def validate_dataframe(df: DataFrame) -> bool:
        """Validate dataframe structure"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        if len(df) == 0:
            raise ValueError("DataFrame has no rows")
        return True

    @staticmethod
    def get_row_as_dict(df: DataFrame, index: int) -> dict:
        """Get single row as dictionary"""
        if index >= len(df):
            raise IndexError(f"Row index {index} out of range")
        return df.iloc[index].to_dict()