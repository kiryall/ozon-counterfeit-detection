# services/prediction_service.py
from typing import List
from pandas import DataFrame
from models.prediction import (
    PredictionType,
    PredictionResponse,
    BatchPredictionResponse,
    ImageProcessor
)
from core.logging import setup_logging

# Настройка логирования
logger = setup_logging(log_file="prediction.log", console=True, remove_file=True, logger_name="prediction_service")


class PredictionProcessor:
    """Класс для предсказания"""

    def __init__(self, model, multimodal_processor):
        """
        Initialize prediction processor
        
        Args:
            model: Trained MultiModalClassifier
            multimodal_processor: MultiModalFeatureUnion for feature extraction
        """
        self.model = model
        self.multimodal_processor = multimodal_processor

    def predict_single(self, image_bytes: bytes, dataframe_row: dict, image_filename: str = None) -> PredictionResponse:
        """
        Single prediction: 1 image + 1 dataframe row
        
        Args:
            image_bytes: Image file bytes
            dataframe_row: Dictionary containing single row data
            image_filename: Original image filename
            
        Returns:
            PredictionResponse with prediction and confidence
        """

        try:
            logger.info(f"Starting single prediction for image: {image_filename if image_filename else 'unknown'} and dataframe row with item_id: {dataframe_row.get('item_id', 'unknown')}")
            # Load and process image
            image = ImageProcessor.load_image_from_bytes(image_bytes)
            if not ImageProcessor.validate_image_format(image):
                raise ValueError("Invalid image format")

            # Create single-row dataframe
            df = DataFrame([dataframe_row])

            # Extract features using multimodal processor
            features = self.multimodal_processor.transform(df)

            # Get prediction probabilities
            probas = self.model.predict_proba(features)
            
            # Get prediction (0 = REAL, 1 = FAKE based on classification threshold)
            pred_class = self.model.predict(features)[0]
            confidence = float(probas[0][pred_class])

            prediction = PredictionType.FAKE if pred_class == 1 else PredictionType.REAL

            item_id = dataframe_row.get('item_id', None)

            logger.info(f"Single prediction completed: {prediction} with confidence {confidence:.4f} for item_id: {item_id}")

            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                item_id=item_id
            )

        except Exception as e:
            logger.error(f"Error in single prediction: {str(e)}")
            raise Exception(f"Error in single prediction: {str(e)}")
        

    def predict_batch(self, image_bytes_list: List[bytes], dataframe: DataFrame, image_filenames: List[str] = None) -> BatchPredictionResponse:
        """
        Batch prediction: multiple images + multiple dataframe rows
        
        Args:
            image_bytes_list: List of image file bytes
            dataframe: DataFrame with multiple rows
            image_filenames: List of original image filenames
            
        Returns:
            BatchPredictionResponse with predictions and confidences
        """
        try:
            logger.info(f"Starting batch prediction for {len(image_bytes_list)} images and dataframe with {len(dataframe)} rows")

            if len(image_bytes_list) != len(dataframe):
                raise ValueError(f"Number of images ({len(image_bytes_list)}) must match number of rows ({len(dataframe)})")

            # Load and process images
            images = []
            for image_bytes in image_bytes_list:
                image = ImageProcessor.load_image_from_bytes(image_bytes)
                if not ImageProcessor.validate_image_format(image):
                    raise ValueError("Invalid image format in batch")
                images.append(image)

            # Extract features using multimodal processor
            features = self.multimodal_processor.transform(dataframe)

            # Get prediction probabilities
            probas = self.model.predict_proba(features)
            preds = self.model.predict(features)

            predictions = []
            confidences = []
            
            for pred_class, proba_row in zip(preds, probas):
                confidence = float(proba_row[pred_class])
                prediction = PredictionType.FAKE if pred_class == 1 else PredictionType.REAL
                predictions.append(prediction)
                confidences.append(confidence)

            item_ids = dataframe.get('item_id', []).tolist() if 'item_id' in dataframe.columns else None

            logger.info(f"Batch prediction completed for {len(predictions)} items")

            return BatchPredictionResponse(
                predictions=predictions,
                confidences=confidences,
                item_ids=item_ids,
                count=len(predictions)
            )

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise Exception(f"Error in batch prediction: {str(e)}")