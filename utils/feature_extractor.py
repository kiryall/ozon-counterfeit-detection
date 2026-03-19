# utils/feature_extractor.py
# Скрипт для извлечения фич из сырых данных
import argparse
import json
import pickle

from pandas import DataFrame

from core import config
from core.logging import setup_logging
from utils.data_utils import load_data, train_val_test_split
from utils.multimodal import MultiModalFeatureUnion
from pathlib import Path

# Настройка логирования
logger = setup_logging(log_file="features.log", console=True, remove_file=True, logger_name="feature_extractor")


def ensure_parent_dir(path: str):
    """Создание родительской директории для указанного пути.

    Создает все необходимые родительские директории, если они не существуют.

    Args:
        path: Путь к файлу или директории.
    """
    parent = Path(path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def save_features(df: DataFrame, path: str):
    """Сохранение DataFrame в CSV файл.

    Сохраняет DataFrame в файл по указанному пути, предварительно
    создавая необходимые директории.

    Args:
        df: DataFrame для сохранения.
        path: Путь к файлу для сохранения.

    Raises:
        Exception: Ошибка при сохранении файла.
    """
    try:
        ensure_parent_dir(path)
        df.to_csv(path, index=False)
        logger.info(f"Saved features to {path}")
    except Exception:
        logger.exception(f"Failed to save features to {path}")
        raise


def feature_extractor(data_path: str = config.DATA_CSV, sample: int | None = None, model_name: str = "resnet18"):
    """Извлечение признаков из тренировочных данных.

    Выполняет полный пайплайн извлечения признаков:
    1. Загружает данные из CSV.
    2. Разделяет на тренировочную, валидационную и тестовую выборки.
    3. Извлекает мультимодальные признаки (табличные, текстовые, визуальные).
    4. Сохраняет признаки и процессор в файлы.

    Args:
        data_path: Путь к файлу с данными.
        sample: Количество строк для извлечения (для тестирования).
        model_name: Название модели для извлечения признаков изображений.
    """
    logger.info(
        f"Feature pipeline started. Data path: {data_path}, Sample size: {sample}, Model name: {model_name}"
    )

    # загрузка данных
    data = load_data(data_path)
    logger.info(f"Data loaded from {data_path}, shape: {data.shape}")

    if sample:
        data = data.sample(n=sample, random_state=config.SEED)
        logger.info(f"Data sampled to {sample} rows, new shape: {data.shape}")

    # разделение данных
    X_train, X_val, X_test, y_train, y_test, y_val = train_val_test_split(
        data,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.SEED,
    )

    logger.info(
        f"Split completed: "
        f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}"
    )

    # feature extraction
    multimodal = MultiModalFeatureUnion(model_name=model_name)

    logger.info("Fitting multimodal processor on TRAIN set")
    X_train_features = multimodal.fit_transform(X_train)

    logger.info("Transforming VAL set")
    X_val_features = multimodal.transform(X_val)

    logger.info("Transforming TEST set")
    X_test_features = multimodal.transform(X_test)

    # Не добавляем target к признакам - он будет использоваться отдельно
    # X_train_features[config.TARGET] = y_train
    # X_val_features[config.TARGET] = y_val
    # X_test_features[config.TARGET] = y_test

    logger.info(
        f"Train features shape: {X_train_features.shape}, "
        f"Val features shape: {X_val_features.shape}, "
        f"Test features shape: {X_test_features.shape}"
    )

    # сохраняем признаки
    logger.info("Saving features")

    save_features(X_train_features, config.TRAIN_FEATURES_PATH)
    save_features(X_val_features, config.VAL_FEATURES_PATH)
    save_features(X_test_features, config.TEST_FEATURES_PATH)

    # Сохранение категориальных фич
    cat_features = multimodal.get_features().get("Category")
    if cat_features is None:
        logger.warning("Key 'Category' not found in multimodal.get_features(); saving empty list")
        cat_features = []

    ensure_parent_dir(config.CAT_FEATURES_PATH)
    with open(config.CAT_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(cat_features, f, ensure_ascii=False)

    # Сохранение процессора
    ensure_parent_dir(config.MULTIMODAL_PROCESSOR_PATH)
    with open(config.MULTIMODAL_PROCESSOR_PATH, "wb") as f:
        pickle.dump(multimodal, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Multimodal processor saved")

    logger.info("Feature pipeline finished successfully")

def main():
    argparser = argparse.ArgumentParser(
        description="Feature extraction for training data"
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        default=config.DATA_CSV,
        help="Path to the input data CSV file",
    )
    argparser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of rows to sample for feature extraction (for testing purposes)",
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Name of the model to use for image feature extraction (e.g., resnet18, resnet50)",
    )
    args = argparser.parse_args()

    feature_extractor(data_path=args.data_path, sample=args.sample, model_name=args.model_name)
    
if __name__ == "__main__":
    main()
