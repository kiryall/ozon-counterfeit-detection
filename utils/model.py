# Универсальный классификатор для мультимодальных данных

from typing import Any, Dict, List, Optional
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    f1_score,
)

# core.config as config
import core.config as config
import pickle

from core.logging import setup_logging

logger = setup_logging(log_file="model.log", console=True, remove_file=True, logger_name="model")


class MultiModalClassifier(BaseEstimator, ClassifierMixin):
    """Универсальный классификатор для мультимодальных данных.

    Поддерживает различные алгоритмы машинного обучения (CatBoost, LightGBM, XGBoost)
    и предоставляет единый интерфейс для обучения и предсказания.
    """

    def __init__(self, 
                 algorithm: str = 'catboost',
                 cat_features: Optional[List[str]] = None,
                 model_params: Optional[Dict[str, Any]] = None,
                 classification_threshold: float = 0.5,
                 cv_folds: int = 5,
                 random_state: int = config.SEED):
        """Инициализация мультимодального классификатора.

        Args:
            algorithm: Алгоритм модели ('catboost', 'lightgbm', 'xgboost').
            cat_features: Список категориальных признаков.
            model_params: Параметры модели.
            classification_threshold: Порог классификации.
            cv_folds: Количество фолдов для кросс-валидации.
            random_state: Зерно случайности для воспроизводимости.
        """
        self.algorithm = algorithm
        self.cat_features = cat_features or []
        self.cat_feature_indices = None
        self.model_params = model_params or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.feature_importance_ = None
        self.best_score_ = None
        self.study = None
        self.classification_threshold = classification_threshold

    def _init_model(self):
        """Инициализация модели машинного обучения.

        Создает экземпляр модели на основе выбранного алгоритма.

        Returns:
            Экземпляр модели.

        Raises:
            ValueError: Алгоритм не поддерживается.
        """
        if self.algorithm == 'catboost':
            # обновляем параметры
            base_params = config.CATBOOST_PARAMS.copy()
            base_params.update(self.model_params)
            self.model = CatBoostClassifier(**base_params)
        # elif self.algorithm == 'lightgbm':
        # elif self.algorithm == 'xgboost':
        else:
            raise ValueError(f"Алгоритм {self.algorithm} не поддерживается")


    def _optimal_threshold(self, model, X_val, y_val):
        """Поиск оптимального порога классификации.

        Перебирает различные значения порога и выбирает тот,
        при котором достигается максимальное значение F1-меры.

        Args:
            model: Обученная модель.
            X_val: Валидационные данные.
            y_val: Валидационные метки.

        Returns:
            Оптимальный порог классификации.
        """
        probas = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 250)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (probas > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """Обучение модели на тренировочных данных.

        Обучает модель на переданных данных, выполняет валидацию
        и автоматически подбирает оптимальный порог классификации.

        Args:
            X: Признаки для обучения.
            y: Целевая переменная.
            X_val: Валидационные признаки (необязательно).
            y_val: Валидационная целевая переменная (необязательно).
            **kwargs: Дополнительные аргументы для метода fit модели.
        """
        self._init_model()

        if self.algorithm == 'catboost' and isinstance(X, pd.DataFrame):
            # Создаем Pool для CatBoost
            # индексы столбцов категориальных фич
            self.cat_feature_indices = [X.columns.get_loc(col) for col in self.cat_features if col in X.columns]

            train_pool = Pool(X, y, 
                            cat_features=self.cat_feature_indices)

            if X_val is not None:
                val_pool = Pool(X_val, y_val,
                              cat_features=self.cat_feature_indices)
                self.model.fit(train_pool, eval_set=val_pool, **kwargs)
                # СОХРАНЯЕМ ДЛЯ ПОДБОРА ПОРОГА
                self.X_val = X_val
                self.y_val = y_val
                # Автоматически находим оптимальный порог
                self.classification_threshold = self._optimal_threshold(self.model, X_val, y_val)
            else:
                self.model.fit(train_pool, **kwargs)


    def predict_proba(self, X, **kwargs):
        """Предсказание вероятностей классов.

        Возвращает вероятности принадлежности к каждому классу.

        Args:
            X: Признаки для предсказания.
            **kwargs: Дополнительные аргументы.

        Returns:
            Массив вероятностей классов.

        Raises:
            ValueError: Модель не обучена.
        """
        if self.model is None:
            raise ValueError("Модель не обучена!")

        if self.algorithm == "catboost" and isinstance(X, pd.DataFrame):
            # Вычисляем индексы заново для тестовых данных
            cat_indices = [
                X.columns.get_loc(col) for col in self.cat_features if col in X.columns
            ]
            test_pool = Pool(X, cat_features=cat_indices)
            return self.model.predict_proba(test_pool)
        else:
            return self.model.predict_proba(X)
        

    def predict(self, X, **kwargs):
        """Предсказание классов для входных данных.

        Использует порог классификации для определения класса.

        Args:
            X: Признаки для предсказания.
            **kwargs: Дополнительные аргументы.

        Returns:
            Массив предсказанных классов.
        """
        probas = self.predict_proba(X)

        if probas.shape[1] == 2:
            return (probas[:, 1] > self.classification_threshold).astype(int)
        else:
            return np.argmax(probas, axis=1)
        

    def tune_threshold(self, X_val, y_val):
        """Ручной запуск подбора оптимального порога классификации.

        Выполняет поиск оптимального порога на валидационных данных.

        Args:
            X_val: Валидационные данные.
            y_val: Валидационные метки.

        Returns:
            Оптимальный порог классификации.

        Raises:
            ValueError: Модель не обучена.
        """
        if self.model is None:
            raise ValueError("Сначала обучите модель!")
        
        self.classification_threshold = self._optimal_threshold(self.model, X_val, y_val)
        logger.info(f"Оптимальный порог: {self.classification_threshold:.3f}")
        return self.classification_threshold
    

    def save_model(self, filepath: str):
        """Сохранение модели в файл.

        Сохраняет модель в нативном формате (.cbm для CatBoost)
        и метаданные в отдельный файл.

        Args:
            filepath: Путь для сохранения модели.

        Raises:
            ValueError: Модель не обучена.
        """
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        # Определяем путь для модели и метаданных
        # Если передан .pkl файл, разделяем на модель и метаданные
        if filepath.endswith('.pkl'):
            model_path = filepath.replace('.pkl', '.cbm')
            metadata_path = filepath
        else:
            model_path = filepath
            metadata_path = filepath.replace('.cbm', '_metadata.pkl')
        
        # Сохраняем модель CatBoost в нативном формате .cbm
        self.model.save_model(model_path)
        
        # Сохраняем метаданные в отдельный файл
        metadata = {
            'cat_features': self.cat_features,
            'cat_feature_indices': self.cat_feature_indices,
            'classification_threshold': self.classification_threshold,
            'algorithm': self.algorithm,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'model_params': self.model_params
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load_model(cls, filepath: str):
        """Загрузка модели из файла.

        Загружает модель и метаданные из файлов.

        Args:
            filepath: Путь к файлу модели.

        Returns:
            Загруженный экземпляр MultiModalClassifier.

        Raises:
            FileNotFoundError: Файл модели или метаданных не найден.
        """
        # Определяем пути для модели и метаданных
        if filepath.endswith('.pkl'):
            model_path = filepath.replace('.pkl', '.cbm')
            metadata_path = filepath
        else:
            model_path = filepath
            metadata_path = filepath.replace('.cbm', '_metadata.pkl')
        
        # Создаем экземпляр класса
        loaded_model = cls()
        
        # Загружаем основную модель CatBoost из .cbm файла
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        loaded_model.model = CatBoostClassifier().load_model(model_path)
        
        # Загружаем метаданные
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Восстанавливаем атрибуты
        loaded_model.cat_features = metadata['cat_features']
        loaded_model.cat_feature_indices = metadata['cat_feature_indices']
        loaded_model.classification_threshold = metadata['classification_threshold']
        loaded_model.algorithm = metadata['algorithm']
        loaded_model.cv_folds = metadata['cv_folds']
        loaded_model.random_state = metadata['random_state']
        loaded_model.model_params = metadata['model_params']
        
        return loaded_model


