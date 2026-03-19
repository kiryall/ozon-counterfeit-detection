# utils/data_utils.py
# Утилиты для работы с данными
import multiprocessing as mp
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

import core.config as config


def to_snake_case(name: str) -> str:
    """Преобразование имени из CamelCase / camelCase в snake_case.

    Args:
        name: Строка для преобразования.

    Returns:
        Строка в формате snake_case.
    """
    if not isinstance(name, str):
        return name

    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    return name.lower()

def load_data(path: str):
    """Загрузка данных из CSV файла.

    Загружает CSV файл и преобразует названия столбцов в snake_case.

    Args:
        path: Путь к файлу data.csv.

    Returns:
        DataFrame с загруженными данными.

    Raises:
        FileNotFoundError: Файл не найден.
        ValueError: Ошибка при преобразовании названий столбцов.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
    
    df = pd.read_csv(path, index_col=0)
    df = df.copy()

    try:
        # преобразование названий столбцов в snake_case
        df.columns = [to_snake_case(col) for col in df.columns]
        
        if df.index.name:
            df.index.name = to_snake_case(df.index.name)

    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании названий столбцов to snake_case: {e}")

    return df


def train_val_test_split(
    df: pd.DataFrame, test_size: float, val_size: float, random_state: int
):
    """Разделение данных на тренировочную, валидационную и тестовую выборки.

    Выполняет стратифицированное разделение данных с сохранением
    пропорций классов.

    Args:
        df: Исходные данные.
        test_size: Доля тестовой выборки.
        val_size: Доля валидационной выборки.
        random_state: Зерно случайности для воспроизводимости.

    Returns:
        Кортеж из:
        - train_df: Тренировочный DataFrame.
        - val_df: Валидационный DataFrame.
        - test_df: Тестовый DataFrame.
        - y_train: Целевая переменная тренировочной выборки.
        - y_test: Целевая переменная тестовой выборки.
        - y_val: Целевая переменная валидационной выборки.
    """
    val_relative_size = val_size / (1 - test_size)

    # X, y
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]

    train_df, test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=df[config.TARGET]
    )

    train_df, val_df, y_train, y_val = train_test_split(
        train_df,
        y_train,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train,
    )

    # кодируем таргет
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    return train_df, val_df, test_df, y_train, y_test, y_val


class BytesImageDataset(Dataset):
    """Класс для работы с изображениями в виде байтов.

    Наследует Dataset из PyTorch для использования с DataLoader.
    Принимает словарь с байтами изображений вместо загрузки из файловой системы.
    """

    def __init__(self, df, image_bytes_dict: dict, transform=None, img_size=config.IMG_SIZE):
        """Инициализация датасета с байтами изображений.

        Args:
            df: DataFrame с идентификаторами изображений.
            image_bytes_dict: Словарь {item_id: bytes} с изображениями.
            transform: Трансформации из torchvision (необязательно).
            img_size: Размер изображения (кортеж).
        """
        self.df = df
        self.image_bytes_dict = image_bytes_dict
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row[config.ITEM]

        label = row.get("resolution", -1)

        # Пробуем получить изображение из байтов
        if img_id in self.image_bytes_dict:
            try:
                from io import BytesIO
                image = Image.open(BytesIO(self.image_bytes_dict[img_id])).convert("RGB")
            except Exception as e:
                print(f"Не удалось прочитать изображение для {img_id}: {e}")
                image = Image.fromarray(np.zeros((*self.img_size, 3), dtype=np.uint8))
        else:
            # Если нет в словаре - создаём пустое изображение
            image = Image.fromarray(np.zeros((*self.img_size, 3), dtype=np.uint8))

        # аугментации
        if self.transform:
            image = self.transform(image)

        # Возвращаем img_id как список для корректной работы DataLoader
        return image, label, [img_id]


class ImageDataset(Dataset):
    """Класс для загрузки изображений из датасета.

    Наследует Dataset из PyTorch для использования с DataLoader.
    Загружает изображения и метки из указанной директории.
    """

    def __init__(self, df, img_dir, transform=None, img_size=config.IMG_SIZE):
        """Инициализация датасета изображений.

        Args:
            df: DataFrame с идентификаторами изображений.
            img_dir: Путь к папке с изображениями.
            transform: Трансформации из torchvision (необязательно).
            img_size: Размер изображения (кортеж).
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row[config.ITEM]

        label = row.get("resolution", -1)

        # path
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        # обработка отсутствующих фото
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Не удалось прочитать файл {img_path}: {e}")
                image = Image.fromarray(np.zeros((*self.img_size, 3), dtype=np.uint8))
        else:
            image = Image.fromarray(np.zeros((*self.img_size, 3), dtype=np.uint8))

        # аугментации
        if self.transform:
            image = self.transform(image)

        return image, label, [img_id]


def preview_batch(loader, num_img=8, num_row=4):
    """Визуализация батча из даталоадера.

    Отображает сетку изображений из батча для проверки данных.

    Args:
        loader: DataLoader с изображениями.
        num_img: Количество изображений для отображения.
        num_row: Количество изображений в строке.
    """
    images, label, _ = next(iter(loader))

    grid = torchvision.utils.make_grid(images[:num_img], nrow=num_row, normalize=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


def resize_save_img(args):
    """Изменение размера и сохранение одного изображения.

    Загружает изображение, изменяет его размер до указанного
    и сохраняет в целевую папку.

    Args:
        args: Кортеж из (исходный_путь, целевой_путь, размер_изображения).

    Returns:
        True при успешном сохранении, False при ошибке.
    """
    source_path, target_path, img_size = args

    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            resize_img = img.resize(img_size, Image.Resampling.LANCZOS)

            # сохранение
            resize_img.save(target_path, "PNG", optimize=True)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении {e}")
        return False


def create_resized_dataset(
    source_path, target_path, img_size=config.IMG_SIZE, num_workers=8
):
    """Создание уменьшенной копии всего датасета изображений.

    Рекурсивно обрабатывает все PNG изображения в исходной папке,
    изменяет их размер и сохраняет в целевую папку.

    Args:
        source_path: Путь к исходной папке с изображениями.
        target_path: Путь к целевой папке для уменьшенных копий.
        img_size: Целевой размер изображений.
        num_workers: Количество процессов для параллельной обработки.

    Returns:
        Количество успешно обработанных изображений.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    img_paths = list(source_path.glob("*.png"))
    print(f"Найдено {len(img_paths)} изображений")

    tasks = []
    for img_path in img_paths:
        target = target_path / img_path.name
        tasks.append((img_path, target, img_size))

    print("Start resize...")
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(resize_save_img, tasks),
                total=len(tasks),
                desc="Resize images",
            )
        )

    success_count = sum(results)
    print(f"Успешно обработано: {success_count}/{len(tasks)}")
    return success_count
