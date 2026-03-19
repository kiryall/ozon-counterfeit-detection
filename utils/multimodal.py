# utils/multimodal.py
# Универсальный класс для объединения признаков из разных модальностей

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import core.config as config
from core.logging import setup_logging
from utils.data_utils import ImageDataset, BytesImageDataset
from utils.features import ImageFeatureExtractor, SentenceEmbedder
from utils.preprocessing import TabularPreprocessor, TextPreprocessor

# Настройка логирования
logger = setup_logging(log_file="multimodal.log", console=True, remove_file=True, logger_name="multimodal")

class MultiModalFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Класс для объединения признаков из трех модальностей:
    - Табличные данные
    - Текстовые эмбеддинги  
    - Визуальные признаки
    """

    def __init__(self, model_name: str = "resnet18"):
        self.tabular = TabularPreprocessor()
        self.text = TextPreprocessor()
        self.text_embedder = SentenceEmbedder()
        self.img_extractor = ImageFeatureExtractor(model_name=model_name)
        self.image_dataset = None
        self.is_fitted = False

    def fit(self, df, y=None):
        """
        Обучает все препроцессоры на тренировочных данных
        """
        # предобработка текста
        preprocessed_text = self.text.fit(df)

        # создание Dataset
        self.image_dataset = ImageDataset(df, config.IMG_DIR, transform=config.TRANSFORMS)

        self.tabular.fit(df)
        self.text_embedder.fit(preprocessed_text)
        self.img_extractor.fit(self.image_dataset)

        self.is_fitted = True
        logger.info("Препроцессоры обучены")
        return self

    def transform(self, df, y=None):
        """Извлечение и объединение признаков из всех модальностей.

        Выполняет преобразование данных с использованием всех
        препроцессоров и объединяет признаки в единый DataFrame.

        Args:
            df: DataFrame с данными для преобразования.
            y: Целевая переменная (не используется).

        Returns:
            DataFrame с объединенными признаками из всех модальностей.

        Raises:
            ValueError: Препроцессоры не обучены (не вызван fit()).
        """
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()!")
        
        # предобработка текста
        preprocessed_text = self.text.transform(df)
        
        # создание Dataset
        image_dataset = ImageDataset(df, config.IMG_DIR, transform=config.TRANSFORMS)

        # извлечение признаков
        tabular_features = self.tabular.transform(df)
        text_features = self.text_embedder.transform(preprocessed_text)
        img_features = self.img_extractor.transform(image_dataset)

        # объединение в один датафрейм
        comb_features = pd.concat(
            [tabular_features, text_features, img_features], axis=1
        )

        logger.info(f"Извлечены признаки: {comb_features.shape}")

        return comb_features

    def transform_with_bytes(self, df, image_bytes_dict: dict, y=None):
        """Извлечение и объединение признаков из всех модальностей с использованием bytes изображений.

        Выполняет преобразование данных с использованием всех
        препроцессоров и объединяет признаки в единый DataFrame.
        В отличие от transform() принимает словарь с байтами изображений
        вместо загрузки из файловой системы.

        Args:
            df: DataFrame с данными для преобразования.
            image_bytes_dict: Словарь {item_id: bytes} с изображениями.
            y: Целевая переменная (не используется).

        Returns:
            DataFrame с объединенными признаками из всех модальностей.

        Raises:
            ValueError: Препроцессоры не обучены (не вызван fit()).
        """
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()!")
        
        # предобработка текста
        preprocessed_text = self.text.transform(df)
        
        # создание Dataset с байтами изображений
        image_dataset = BytesImageDataset(df, image_bytes_dict, transform=config.TRANSFORMS)

        # извлечение признаков
        tabular_features = self.tabular.transform(df)
        text_features = self.text_embedder.transform(preprocessed_text)
        img_features = self.img_extractor.transform(image_dataset)

        # объединение в один датафрейм
        comb_features = pd.concat(
            [tabular_features, text_features, img_features], axis=1
        )

        logger.info(f"Извлечены признаки (из bytes): {comb_features.shape}")

        return comb_features

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def get_features(self):
        """Получение информации об извлеченных признаках.

        Returns:
            Словарь с категориальными и числовыми признаками:
            - Category: Список категориальных признаков.
            - Numeric: Список числовых признаков (изображения и текст).
        """
        return {
            "Category": self.tabular.get_cat_features(),
            "Numeric": self.img_extractor.get_img_features()
            + self.text_embedder.get_text_features(),
        }