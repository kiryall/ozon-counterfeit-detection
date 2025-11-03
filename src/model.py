import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report
)
from catboost import CatBoostClassifier, Pool
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

import config


class MultiModalClassifier(BaseEstimator, ClassifierMixin):
    """
    Универсальный классификатор для мультимодальных данных.
    Поддерживает различные алгоритмы и оптимизацию гиперпараметров. # в разработке
    """

    def __init__(self, 
                 algorithm: str = 'catboost',
                 cat_features: Optional[List[str]] = None,
                 model_params: Optional[Dict[str, Any]] = None,
                 classification_threshold: float = 0.5,
                 cv_folds: int = 5,
                 random_state: int = config.SEED):
        """
        Parameters:
        -----------
        algorithm : str
            Алгоритм модели ('catboost', 'lightgbm', 'xgboost')
        cat_features :  list
            список категориальных фич
        model_params : dict
            Параметры модели
        cv_folds : int
            Количество фолдов для кросс-валидации
        random_state : int
            Seed для воспроизводимости
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
        """ Инициализация модели """
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
        """Функция ищет оптимальный порог классификации"""
        probas = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 100)
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
        """
        Обучение модели
        
        Parameters:
        -----------
        X : array-like
            Признаки для обучения
        y : array-like
            Целевая переменная
        X_val : array-like, optional
            Валидационные признаки
        y_val : array-like, optional
            Валидационная целевая переменная
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
        """Предсказание вероятности классов"""

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
        """Предсказание классов"""

        probas = self.predict_proba(X)

        if probas.shape[1] == 2:
            return (probas[:, 1] > self.classification_threshold).astype(int)
        else:
            return np.argmax(probas, axis=1)
        

    def tune_threshold(self, X_val, y_val):
        """Ручной вызов подбора порога"""
        if self.model is None:
            raise ValueError("Сначала обучите модель!")
        
        self.classification_threshold = self._optimal_threshold(self.model, X_val, y_val)
        print(f"Оптимальный порог: {self.classification_threshold:.3f}")
        return self.classification_threshold


