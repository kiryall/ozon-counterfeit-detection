# tests/prediction_test.py
# Тестирование предсказаний модели на одной картинке и одной строке данных.

from core.logging import setup_logging
from services.prediction_service import PredictionProcessor
from services.model_loader import load_model, load_multimodal_processor
from core.config import MODEL_PATH, MULTIMODAL_PROCESSOR_PATH

# Настройка логирования
logger = setup_logging(log_file="prediction_test.log", console=True, remove_file=True, logger_name="prediction_test")

def predict_single_test():
    """Основной метод для тестирования предсказаний.
    """

    model = load_model(MODEL_PATH)
    multimodal_processor = load_multimodal_processor(MULTIMODAL_PROCESSOR_PATH)

    prediction_processor = PredictionProcessor(model, multimodal_processor)
    logger.info("PredictionProcessor initialized successfully")

    with open("test_data/single/78312.png", "rb") as f:
        image_bytes = f.read()

    with open("test_data/single/test_data.csv", "r") as f:
        import csv
        reader = csv.DictReader(f)
        dataframe_row = next(reader)  # Получаем первую строку данных

    prediction_processor.predict_single(
        image_bytes=image_bytes,
        dataframe_row=dataframe_row,
        image_filename="78312.png"
    )

if __name__ == "__main__":
    predict_single_test()