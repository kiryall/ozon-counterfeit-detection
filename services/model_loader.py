# services/model_loader.py
from utils.model import MultiModalClassifier
from core import config
from core.logging import setup_logging
import os

 # Настройка логирования
logger = setup_logging(log_file="model_loader.log", console=True, remove_file=True, logger_name="model_loader")

def load_model(path: str):
    """
    Загружает модель из указанного пути. Ожидается, что модель
    сохранена в формате .cbm с соответствующим .pkl файлом метаданных.
    param path: Путь к файлу модели (может быть .cbm или .pkl)
    return: Загруженный экземпляр MultiModalClassifier
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    model = MultiModalClassifier()
    return model.load_model(path)


def load_multimodal_processor(path: str):
    """
    Загружает препроцессор для мультимодальных данных из указанного пути.
    param path: Путь к файлу препроцессора (ожидается .pkl)
    return: Загруженный препроцессор
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Multimodal processor file not found: {path}")
    
    from utils.multimodal import MultiModalFeatureUnion
    import joblib

    try:
        processor = joblib.load(path)
        if not isinstance(processor, MultiModalFeatureUnion):
            raise ValueError(f"Loaded object is not a MultiModalFeatureUnion: {type(processor)}")
        return processor
    except Exception as e:
        logger.error(f"Error loading multimodal processor: {e}")
        raise


if __name__ == "__main__":
    try:
        model = load_model(config.MODEL_PATH)
        processor = load_multimodal_processor(config.MULTIMODAL_PROCESSOR_PATH)
        logger.info(f"Model successfully loaded from {config.MODEL_PATH}")
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")