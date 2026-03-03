import json
import logging

from pandas import DataFrame

from core import config
from utils.data_utils import load_data
from utils.multimodal import MultiModalFeatureUnion

logging.basicConfig(
    filename="data/features.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)


def save_features(df: DataFrame, path: str):
    """
    Функция для сохранения фич в csv файл
    :param df: DataFrame с фичами
    :param path: Путь для сохранения
    """

    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"Error {e}")


def feature_extractor(data_path: str = config.DATA_CSV, sample: int | None = None):
    """
    Функция для извлечения фич тренировочных данных
    :param data_path: Путь к файлу с данными
    :param sample: Количество строк для извлечения (для тестирования)
    """

    # START
    logging.info("Training pipeline started.")

    # загрузка данных
    data = load_data(data_path)
    logging.info(f"Data loaded from {data_path}, shape: {data.shape}")

    if sample:
        data = data.sample(n=sample, random_state=config.SEED)
        logging.info(f"Data sampled to {sample} rows, new shape: {data.shape}")

    # Извлекаем фичи без целевой переменной
    logging.info("Extracting multimodal features...")
    multimodal = MultiModalFeatureUnion()
    X = data.drop(config.TARGET, axis=1)
    features = multimodal.fit_transform(X)
    
    # Добавляем целевую переменную обратно
    features[config.TARGET] = data[config.TARGET].values
    logging.info(f"Features extracted and target attached, final shape: {features.shape}")

    # Получаем список категориальных фич
    cat_features = multimodal.get_features()["Category"]

    # сохраняем фичи
    logging.info("Saving features...")
    save_features(features, config.FEATURES_PATH)

    with open(config.CAT_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(cat_features, f)
    logging.info("Features secsessfuly saved")


if __name__ == "__main__":
    feature_extractor(sample=20000)