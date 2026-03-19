# services/prediction_service.py

import json
import os
from typing import List
from pandas import DataFrame
from models.prediction import (
    PredictionType,
    PredictionResponse,
    BatchPredictionResponse,
    ImageProcessor
)
from core import config
from core.logging import setup_logging

# Настройка логирования
logger = setup_logging(log_file="prediction.log", console=True, remove_file=True, logger_name="prediction_service")


class PredictionProcessor:
    """Класс для выполнения предсказаний на основе модели и процессора признаков.

    Предоставляет методы для одиночного и пакетного предсказания.
    """

    def __init__(self, model, multimodal_processor, cat_features_path: str = None):
        """Инициализация процессора предсказаний.

        Args:
            model: Обученный классификатор MultiModalClassifier.
            multimodal_processor: Процессор MultiModalFeatureUnion для извлечения признаков.
            cat_features_path: Путь к файлу с категориальными признаками (cat_features.json).
        """
        self.model = model
        self.multimodal_processor = multimodal_processor
        
        # Загружаем категориальные признаки из JSON
        self.cat_features = []
        cat_path = cat_features_path or config.CAT_FEATURES_PATH
        if cat_path and os.path.exists(cat_path):
            with open(cat_path, 'r', encoding='utf-8') as f:
                self.cat_features = json.load(f)
            logger.info(f"Загружено {len(self.cat_features)} категориальных признаков из {cat_path}")
        else:
            logger.warning(f"Файл cat_features не найден: {cat_path}")

    def _prepare_features(self, features):
        """Подготовка признаков перед предсказанием.
        
        Преобразует NaN значения в категориальных признаках в строки.
        
        Args:
            features: DataFrame с признаками.
            
        Returns:
            DataFrame с подготовленными признаками.
        """

        features_prepared = features.copy()
        
        # Обрабатываем категориальные признаки
        for col in self.cat_features:
            if col in features_prepared.columns:
                # Заполняем NaN и преобразуем в строку
                features_prepared[col] = features_prepared[col].fillna("missing").astype(str)
        
        return features_prepared

    def predict_single(self, image_bytes: bytes, dataframe_row: dict, image_filename: str = None) -> PredictionResponse:
        """Одиночное предсказание: одно изображение и одна строка данных.

        Выполняет предсказание для одного объекта на основе
        изображения и соответствующей строки данных.

        Args:
            image_bytes: Байты изображения.
            dataframe_row: Словарь с данными одной строки.
            image_filename: Имя файла изображения (необязательно).

        Returns:
            PredictionResponse с предсказанием и уверенностью.

        Raises:
            Exception: Ошибка при выполнении предсказания.
        """
        try:
            logger.info(f"Starting single prediction for image: {image_filename if image_filename else 'unknown'} and dataframe row with item_id: {dataframe_row.get('item_id', 'unknown')}")
            # Load and process image
            image = ImageProcessor.load_image_from_bytes(image_bytes)
            if not ImageProcessor.validate_image_format(image):
                raise ValueError("Invalid image format")

            # Get item_id for image mapping
            item_id = dataframe_row.get('item_id', None)
            if item_id is None:
                raise ValueError("dataframe_row must contain 'item_id'")

            # Create single-row dataframe
            df = DataFrame([dataframe_row])

            # Create image bytes dictionary for multimodal processor
            image_bytes_dict = {item_id: image_bytes}

            # Extract features using multimodal processor with bytes
            features = self.multimodal_processor.transform_with_bytes(df, image_bytes_dict)

            # Prepare features (handle NaN in categorical features)
            features = self._prepare_features(features)

            # Get prediction probabilities
            probas = self.model.predict_proba(features)
            
            # Get prediction (0 = REAL, 1 = FAKE based on classification threshold)
            pred_class = self.model.predict(features)[0]
            confidence = float(probas[0][pred_class])

            prediction = PredictionType.FAKE if pred_class == 1 else PredictionType.REAL

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
        """Пакетное предсказание: несколько изображений и несколько строк данных.

        Выполняет предсказания для набора объектов на основе
        изображений и соответствующих строк данных.

        Args:
            image_bytes_list: Список байтов изображений.
            dataframe: DataFrame с несколькими строками данных.
            image_filenames: Список имен файлов изображений (необязательно).

        Returns:
            BatchPredictionResponse с предсказаниями и уверенностями.

        Raises:
            Exception: Ошибка при выполнении пакетного предсказания.
        """
        try:
            logger.info(f"Starting batch prediction for {len(image_bytes_list)} images and dataframe with {len(dataframe)} rows")

            if len(image_bytes_list) != len(dataframe):
                raise ValueError(f"Number of images ({len(image_bytes_list)}) must match number of rows ({len(dataframe)})")

            # Validate all images and create bytes dictionary
            item_ids = dataframe['item_id'].tolist() if 'item_id' in dataframe.columns else None
            if item_ids is None:
                raise ValueError("DataFrame must contain 'item_id' column")

            image_bytes_dict = {}
            images = []
            for i, image_bytes in enumerate(image_bytes_list):
                image = ImageProcessor.load_image_from_bytes(image_bytes)
                if not ImageProcessor.validate_image_format(image):
                    raise ValueError(f"Invalid image format at index {i}")
                images.append(image)
                # Map by item_id
                if item_ids[i] is not None:
                    image_bytes_dict[item_ids[i]] = image_bytes

            # Extract features using multimodal processor with bytes
            features = self.multimodal_processor.transform_with_bytes(dataframe, image_bytes_dict)

            # Prepare features (handle NaN in categorical features)
            features = self._prepare_features(features)

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