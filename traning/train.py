import json
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from core import config
from utils.data_utils import train_val_test_split
from utils.model import MultiModalClassifier

logging.basicConfig(
    filename="traning/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

def load_train_data(
    data_path: str = config.FEATURES_PATH,
    cat_features_path: str = config.CAT_FEATURES_PATH
):
    """
    Функция для загрузки данных для обучения модели
    :param data_path: Путь к файлу с фичами
    :param cat_features_path: Путь к файлу с списком категориальных признаков
    :return: Разделенные на трейн, валид, тест данные и список категориальных признаков
    """

    data = pd.read_csv(data_path)

    with open(cat_features_path, "r") as f:
        cat_features = json.load(f)

    # Разделение на трейн, валид, тест
    X_train, X_val, X_test, y_train, y_test, y_val = train_val_test_split(
        data,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.SEED
    )

    logging.info(
        f"Data split into train ({X_train.shape}), val ({X_val.shape}), test ({X_test.shape})"
    )

    return X_train, X_val, X_test, y_train, y_test, y_val, cat_features


def train(X_train, y_train, X_val, y_val, cat_features):
    """
    Функция для обучения модели

    :param X_train: Обучающие данные
    :param y_train: Целевая переменная обучающей выборки
    :param X_val: Валидационные данные
    :param y_val: Целевая переменная валидационной выборки
    :param cat_features: Список категориальных признаков
    """

    # ОБУЧЕНИЕ МОДЕЛИ
    logging.info("Training model...")
    model = MultiModalClassifier("catboost", cat_features=cat_features)
    model.fit(X_train, y_train, X_val, y_val)
    logging.info("Model trained.")

    # Сохраняем модель
    model.save_model(config.MODEL_PATH)


if __name__ == "__main__":

    # Загрузка данных для обучения
    X_train, X_val, X_test, y_train, y_test, y_val, cat_features = load_train_data()
    
    # Обучение модели
    train(X_train, y_train, X_val, y_val, cat_features)
    
    model = MultiModalClassifier.load_model(config.MODEL_PATH)
    
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # вероятности положительного класса
    
    # Вычисление метрик
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("Тестовые метрики:")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

