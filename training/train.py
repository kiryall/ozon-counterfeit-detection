# training/train.py
# Скрипт для обучения модели на извлеченных фичах

import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from core import config
from core.logging import setup_logging
from utils.model import MultiModalClassifier

logger = setup_logging(log_file="training.log", console=True, remove_file=True, logger_name="training")

def load_train_data(
    train_name: str = config.TRAIN_FEATURES_PATH,
    val_name: str = config.VAL_FEATURES_PATH,
    test_name: str = config.TEST_FEATURES_PATH,
    cat_features_path: str = config.CAT_FEATURES_PATH
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Загрузка данных для обучения модели.

    Загружает тренировочные, валидационные и тестовые данные из CSV файлов,
    а также список категориальных признаков.

    Args:
        train_name: Путь к файлу с тренировочными признаками.
        val_name: Путь к файлу с валидационными признаками.
        test_name: Путь к файлу с тестовыми признаками.
        cat_features_path: Путь к файлу со списком категориальных признаков.

    Returns:
        Кортеж из:
        - X_train: Тренировочные данные.
        - X_val: Валидационные данные.
        - X_test: Тестовые данные.
        - y_train: Целевая переменная тренировочной выборки.
        - y_val: Целевая переменная валидационной выборки.
        - y_test: Целевая переменная тестовой выборки.
        - cat_features: Список категориальных признаков.
    """
    X_train = pd.read_csv(train_name)
    X_val = pd.read_csv(val_name)
    X_test = pd.read_csv(test_name)

    y_train = X_train[config.TARGET]
    y_val = X_val[config.TARGET]
    y_test = X_test[config.TARGET]
    
    # Удаляем таргет из признаков
    X_train = X_train.drop(columns=[config.TARGET])
    X_val = X_val.drop(columns=[config.TARGET])
    X_test = X_test.drop(columns=[config.TARGET])

    with open(cat_features_path, "r") as f:
        cat_features = json.load(f)
  
    logger.info(
        f"Data split into train ({X_train.shape}), val ({X_val.shape}), test ({X_test.shape})"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, cat_features


def train(X_train, y_train, X_val, y_val, cat_features):
    """Обучение модели на тренировочных данных.

    Обучает мультимодальный классификатор на переданных данных
    и сохраняет модель в файл.

    Args:
        X_train: Обучающие данные (признаки).
        y_train: Целевая переменная обучающей выборки.
        X_val: Валидационные данные (признаки).
        y_val: Целевая переменная валидационной выборки.
        cat_features: Список категориальных признаков.

    Raises:
        Exception: Ошибка при обучении или сохранении модели.
    """
    try:
        # ОБУЧЕНИЕ МОДЕЛИ
        logger.info("Training model...")
        model = MultiModalClassifier("catboost", cat_features=cat_features)
        model.fit(X_train, y_train, X_val, y_val)
        logger.info("Model trained.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

    try:
        # Сохраняем модель
        model.save_model(config.MODEL_PATH)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def metrics_calculation(y_true, y_pred, y_pred_proba) -> dict:
    """Вычисление метрик качества модели.

    Рассчитывает основные метрики классификации на основе
    истинных и предсказанных значений.

    Args:
        y_true: Истинные метки классов.
        y_pred: Предсказанные метки классов.
        y_pred_proba: Предсказанные вероятности положительного класса.

    Returns:
        Словарь с метриками:
        - f1: F1-мера.
        - accuracy: Точность.
        - precision: Полнота.
        - recall: Полнота (recall).
        - roc_auc: ROC-AUC.
    """
    average_method = "weighted"
    
    f1 = f1_score(y_true, y_pred, average=average_method)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_method)
    recall = recall_score(y_true, y_pred, average=average_method)
    roc_auc = roc_auc_score(y_true, y_pred_proba, labels=[0, 1])
    
    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }

def main():
    parser = argparse.ArgumentParser(description="Multimodal classifier pipeline")

    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--metrics", action="store_true", help="Evaluate model")

    args = parser.parse_args()

    if args.train:
        # X_train, X_val, X_test, y_train, y_val, y_test, cat_features
        X_train, X_val, _, y_train, y_val, _, cat_features = load_train_data()

        train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features
        )

    if args.metrics:
        # X_train, X_val, X_test, y_train, y_val, y_test, cat_features
        _, _, X_test, _, _, y_test, _ = load_train_data()

        try:
            model = MultiModalClassifier.load_model(config.MODEL_PATH)
        except FileNotFoundError:
            logger.error("Model not found. Train model first.")
            raise Exception("Model not found. Train model first.")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = metrics_calculation(y_test, y_pred, y_pred_proba)
        
        logger.info(metrics)

        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        # Проверяем, есть ли доступ к графическому интерфейсу
        import platform
        
        plt.hist(y_pred_proba, bins=50)
        plt.title("Distribution of predicted probabilities")
        
        if platform.system() == 'Windows':
            # На Windows график обычно отображается нормально
            plt.show()
        else:
            # На Linux серверах может не быть GUI
            try:
                plt.show()

            except:
                # Если нет GUI, сохраняем в файл
                plt.savefig('prediction_distribution.png')
                logger.info("Plot saved to prediction_distribution.png")

        # Строим confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()


if __name__ == "__main__":
    main()