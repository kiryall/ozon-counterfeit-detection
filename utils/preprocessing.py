# Классы для предобработки табличных и текстовых данных

import os
import re

import numpy as np
from bs4 import BeautifulSoup
from sklearn.impute import SimpleImputer

import core.config as config
from core.logging import setup_logging

# Настройка логирования
logger = setup_logging(log_file="preprocessing.log", console=True, remove_file=True, logger_name="preprocessing")


class TabularPreprocessor:
    """Класс для предобработки табличных данных.

    Предоставляет функционал для:
    - Определения типов признаков.
    - Обработки пропущенных значений.
    - Преобразования типов данных.
    - Автоматического определения категориальных признаков.
    """

    def __init__(
        self,
        img_dir=config.IMG_DIR,
        text_columns=[
            config.TEXT_COLUMN,
            "brand_name",
            "name_rus",
            "commercial_type_name4",
        ],
        max_unique_categorical=50,
    ):
        """Инициализация препроцессора табличных данных.

        Args:
            img_dir: Путь к директории с изображениями.
            text_columns: Список текстовых колонок для исключения.
            max_unique_categorical: Максимальное количество уникальных
                значений для автоматического определения категориального признака.
        """
        self.numeric_features = None
        self.categorical_features = None
        self.img_dir = img_dir
        self.text_columns = text_columns
        self.max_unique_categorical = max_unique_categorical
        self.auto_detected_categorical = []

    def _auto_detect_categorical(self, df):
        """Автоматическое определение категориальных признаков.

        Определяет категориальные признаки по количеству уникальных значений.
        Исключает признаки со словом 'count' в названии, так как это
        счетчики, а не категории.

        Args:
            df: DataFrame для анализа.

        Returns:
            Список названий потенциально категориальных колонок.
        """
        potential_categorical = []
        num_columns = (
            df.drop(self.text_columns, axis=1)
            .select_dtypes(include=[np.number])
            .columns
        )

        for col in num_columns:
            if col in self.text_columns or col in [config.TARGET, config.ITEM]:
                continue
            # Пропускаем признаки, содержащие 'count' в названии - это счетчики, а не категории
            if 'count' in col.lower():
                continue
            # число уникальных значений
            unique_count = df[col].nunique()

            if unique_count <= self.max_unique_categorical:
                potential_categorical.append(col)

        return potential_categorical

    def _add_indicators(self, df):
        # индикатор наличия картинки
        df["has_image"] = df["item_id"].apply(
            lambda x: int(os.path.exists(os.path.join(self.img_dir, f"{int(x)}.png")))
        )
        # индикатор наличия текста
        df["has_description"] = df[config.TEXT_COLUMN].notna().astype(int)
        return df

    def fit(self, df):
        """Запоминание статистик для нормализации и кодирования.

        Определяет категориальные и числовые признаки в данных.

        Args:
            df: DataFrame для анализа.

        Returns:
            self.
        """
        df = self._add_indicators(df.copy())
        df = df.set_index(config.ITEM, drop=True)

        # определение категориальных признаков
        auto_categorical = self._auto_detect_categorical(df)
        self.auto_detected_categorical = auto_categorical

        base_categorical = (
            df.drop(self.text_columns, axis=1)
            .select_dtypes(include=["object"])
            .columns.tolist()
        )

        self.categorical_features = list(set(base_categorical + auto_categorical))

        self.numeric_features = (
            df.drop(self.text_columns, axis=1)
            .select_dtypes(exclude=["object"])
            .columns.difference(auto_categorical)
            .tolist()
        )

        logger.info(f"Обнаружено категориальных признаков: {len(self.categorical_features)}")
        logger.info(f"Обнаружено числовых признаков: {len(self.numeric_features)}")

        return self

    def transform(self, df):
        """Преобразование табличных признаков.

        Заполняет пропущенные значения и преобразует данные
        в формат, пригодный для модели.

        Args:
            df: DataFrame для преобразования.

        Returns:
            DataFrame с преобразованными признаками.
        """
        df = self._add_indicators(df.copy())
        df = df.set_index(config.ITEM, drop=True)

        # защитимся от случая, если fit не вызывался
        num_feats = [c for c in (self.numeric_features or []) if c in df.columns]
        cat_feats = [c for c in (self.categorical_features or []) if c in df.columns]

        if num_feats:
            df[num_feats] = df[num_feats].fillna(0)

        for col in cat_feats:
            df[col] = df[col].fillna("missing")
            df[col] = df[col].astype(str)

        return df[num_feats + cat_feats]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_info(self):
        """Получение информации о признаках для отладки.

        Returns:
            Словарь с информацией о признаках:
            - numeric_features: Список числовых признаков.
            - categorical_features: Список категориальных признаков.
            - auto_detected_categorical: Автоматически определенные категориальные признаки.
            - total_features: Общее количество признаков.
        """
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "auto_detected_categorical": self.auto_detected_categorical,
            "total_features": len(self.numeric_features)
            + len(self.categorical_features),
        }

    def get_cat_features(self):
        """Получение списка категориальных признаков.

        Returns:
            Список категориальных признаков.
        """
        return self.categorical_features


class TextPreprocessor:
    """Класс для предобработки текстовых данных в DataFrame.

    Предоставляет функционал для:
    - Удаления HTML-тегов.
    - Очистки от специальных символов.
    - Приведения к нижнему регистру.
    - Замены пропущенных значений.
    """

    def __init__(
        self,
        text_columns=[
            config.TEXT_COLUMN,
            "brand_name",
            "name_rus",
            "commercial_type_name4",
        ],
    ):
        """Инициализация текстового препроцессора.

        Args:
            text_columns: Список колонок с текстовыми данными.
        """
        self.str = None
        self.imputer = SimpleImputer(
            strategy="constant", fill_value="missing_discription"
        )
        self.text_columns = text_columns

    def _clear_text(self, text, pattern=r"[^a-zA-Zа-яА-Я0-9]"):
        """Очистка текста от HTML-тегов и специальных символов.

        Args:
            text: Текст для очистки.
            pattern: Регулярное выражение для поиска символов на удаление.

        Returns:
            Очищенный текст в нижнем регистре.
        """
        self.str = BeautifulSoup(text, "html.parser").get_text()
        self.str = " ".join(re.sub(pattern, " ", self.str).split()).lower()
        return self.str

    def fit(self, df):
        """Обучение препроцессора на данных.

        Подгоняет imputer для заполнения пропущенных значений.

        Args:
            df: DataFrame с текстовыми данными.

        Returns:
            self.
        """
        self.imputer.fit(df[self.text_columns])
        return self

    def transform(self, df):
        """Преобразование текстовых данных.

        Обрабатывает текстовые колонки: заполняет пропущенные значения,
        объединяет описания и очищает текст.

        Args:
            df: DataFrame с текстовыми данными.

        Returns:
            DataFrame с обработанным текстом.
        """
        df[self.text_columns] = self.imputer.transform(df[self.text_columns])
        # объединение всех описаний в одно
        df[config.TEXT_COLUMN] = df[self.text_columns].apply(
            lambda x: " ".join(x.astype(str)), axis=1
        )
        # очистка текста
        df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].apply(self._clear_text)

        return df[[config.TEXT_COLUMN, config.ITEM]].set_index(config.ITEM)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
