import os
import re
import warnings
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config
from src.data_loader import ImageDataset
from src.features import ImageFeatureExtractor, SentenceEmbedder
from src.preprocessing import TabularPreprocessor, TextPreprocessor

class MultiModalFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Класс для объединения признаков из трех модальностей:
    - Табличные данные
    - Текстовые эмбеддинги  
    - Визуальные признаки
    """

    def __init__(self):
        self.tabular = TabularPreprocessor()
        self.text = TextPreprocessor()
        self.text_embedder = SentenceEmbedder()
        self.img_extractor = ImageFeatureExtractor()
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
        print("Препроцессоры обучены")
        return self

    def transform(self, df, y=None):
        """
        Извлекает и объединяет признаки из всех модальностей
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

        print(f"Извлечены признаки: {comb_features.shape}")

        return comb_features

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

    def get_features(self):
        """
        Возвращает список категориальных фич
        """
        return {
            "Category": self.tabular.get_cat_features(),
            "Numeric": self.img_extractor.get_img_features()
            + self.text_embedder.get_text_features(),
        }