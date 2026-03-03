# services/model_loader.py
from utils.model import MultiModalClassifier
from core import config
import os

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


if __name__ == "__main__":
    try:
        model = load_model(config.MODEL_PATH)
        print(f"Model successfully loaded from {config.MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"Error loading model: {e}")