# models/prediction.py
from pandas import DataFrame
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from PIL import Image
from io import BytesIO
import numpy as np


class PredictionType(str, Enum):
    """Перечисление типов предсказаний.

    Определяет возможные значения результата предсказания:
    - FAKE: объект определен как фейковый
    - REAL: объект определен как реальный
    """
    FAKE = "FAKE"
    REAL = "REAL"


class SinglePredictionRequest(BaseModel):
    """Модель запроса для одиночного предсказания.

    Примечание: в FastAPI загрузка файлов обрабатывается отдельно через UploadFile.
    Эта модель используется только для валидации ответа.
    """
    pass


class BatchPredictionRequest(BaseModel):
    """Модель запроса для пакетного предсказания.

    Примечание: в FastAPI загрузка файлов обрабатывается отдельно через UploadFile.
    Эта модель используется только для валидации ответа.
    """
    pass


class PredictionResponse(BaseModel):
    """Модель ответа для одиночного предсказания.

    Содержит результат предсказания, уверенность и идентификатор объекта.
    """
    prediction: PredictionType = Field(..., description="Результат предсказания: FAKE или REAL")
    confidence: float = Field(..., ge=0, le=1, description="Оценка уверенности (0-1)")
    item_id: Optional[str] = Field(None, description="Идентификатор объекта из данных")


class BatchPredictionResponse(BaseModel):
    """Модель ответа для пакетного предсказания.

    Содержит списки результатов предсказаний, уверенностей и идентификаторов объектов.
    """
    predictions: List[PredictionType] = Field(..., description="Список результатов предсказаний")
    confidences: List[float] = Field(..., description="Список оценок уверенности")
    item_ids: Optional[List[str]] = Field(None, description="Список идентификаторов объектов из данных")
    count: int = Field(..., description="Количество предсказаний")
    message: Optional[str] = Field(None, description="Дополнительное сообщение")


class ImageProcessor:
    """Вспомогательный класс для обработки загруженных изображений.

    Предоставляет статические методы для загрузки, валидации
    и преобразования изображений.
    """

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
        """Загрузка изображения из байтов.

        Преобразует байты изображения в объект PIL Image.

        Args:
            image_bytes: Байты изображения.

        Returns:
            Объект PIL Image.

        Raises:
            ValueError: Не удалось загрузить изображение из байтов.
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from bytes: {str(e)}")

    @staticmethod
    def validate_image_format(image: Image.Image) -> bool:
        """Валидация формата изображения.

        Проверяет, поддерживается ли формат изображения.

        Args:
            image: Объект PIL Image.

        Returns:
            True, если формат поддерживается, иначе False.
        """
        valid_formats = {'JPEG', 'PNG', 'BMP', 'GIF', 'TIFF'}
        return image.format in valid_formats

    @staticmethod
    def get_image_array(image: Image.Image) -> np.ndarray:
        """Преобразование PIL Image в numpy массив.

        Args:
            image: Объект PIL Image.

        Returns:
            Numpy массив, представляющий изображение.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)


class DataFrameProcessor:
    """Вспомогательный класс для обработки загруженных файлов данных.

    Предоставляет статические методы для загрузки и валидации
    DataFrame из различных форматов файлов.
    """

    @staticmethod
    def load_dataframe_from_bytes(file_bytes: bytes, filename: str) -> DataFrame:
        """Загрузка DataFrame из загруженного файла.

        Поддерживает форматы CSV и Excel.

        Args:
            file_bytes: Байты файла.
            filename: Имя файла с расширением.

        Returns:
            Загруженный DataFrame.

        Raises:
            ValueError: Неподдерживаемый формат файла или ошибка загрузки.
        """
        try:
            if filename.endswith('.csv'):
                df = DataFrame.from_csv(BytesIO(file_bytes), index=False)
                # Handle case where pandas reads first column as index
                if df.index.name is not None:
                    df = df.reset_index()
                return df
            elif filename.endswith(('.xlsx', '.xls')):
                df = DataFrame.from_excel(BytesIO(file_bytes))
                return df
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            raise ValueError(f"Failed to load dataframe from file: {str(e)}")

    @staticmethod
    def validate_dataframe(df: DataFrame) -> bool:
        """Валидация структуры DataFrame.

        Проверяет, что DataFrame не пустой и содержит строки.

        Args:
            df: DataFrame для валидации.

        Returns:
            True, если DataFrame валиден.

        Raises:
            ValueError: DataFrame пустой или не содержит строк.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        if len(df) == 0:
            raise ValueError("DataFrame has no rows")
        return True

    @staticmethod
    def get_row_as_dict(df: DataFrame, index: int) -> dict:
        """Получение одной строки в виде словаря.

        Args:
            df: Исходный DataFrame.
            index: Индекс строки.

        Returns:
            Словарь со значениями строки.

        Raises:
            IndexError: Индекс выходит за пределы диапазона.
        """
        if index >= len(df):
            raise IndexError(f"Row index {index} out of range")
        return df.iloc[index].to_dict()