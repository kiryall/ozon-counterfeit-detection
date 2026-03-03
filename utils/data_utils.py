# Утилиты для работы с данными
import multiprocessing as mp
import os
from pathlib import Path

import humps
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
from ydata_profiling import ProfileReport

import core.config as config

def load_data(path: str):
    """
    Загружает CSV.

    Args:
        path (str): путь к data.csv

    Returns:
        pd.DataFrame: датафрейм
    """
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
    else:
        print("Something is wrong")
        df = None
        
    try:
        # преобразование названий столбцов в snake_case
        df = df.copy()
        df.columns = [humps.decamelize(col) for col in df.columns]
        df.index.name = humps.decamelize(df.index.name)
    except Exception as e:
        print(f"Ошибка при преобразовании названий столбцов: {e}")

    return df


def train_val_test_split(
    df: pd.DataFrame, test_size: float, val_size: float, random_state: int
):
    """
    Делает стратифицированный train/valid/test split.

    Args:
        df (pd.DataFrame): исходные данные
        test_size (float): доля тестовой выборки
        val_size (float): доля валидационной выборки
        random_state (int): random seed

    Returns:
        (pd.DataFrame, pd.DataFrame): train_df, val_df, test_df
    """

    # относительный размер валидационной выборки
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


class ImageDataset(Dataset):
    """
    Класс для загрузки изображений.
    """

    def __init__(self, df, img_dir, transform=None, img_size=config.IMG_SIZE):
        """
        Args:
            df (pd.DataFrame): таблица с id
            img_dir (str): путь к папке с изображениями
            transform (callable): torchvision трансформации
            img_size (tuple): размер изображения
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

        return image, label, img_id


def exploratory_data_analysis(df):
    """
    Создает отчет EDA

    Args:
        df (pd.DataFrame): исходные данные

    """

    report = ProfileReport(df, minimal=True, plot={"dpi": 200, "image_format": "png"})

    report.to_notebook_iframe()


def preview_batch(loader, num_img=8, num_row=4):
    """
    Функция для визуализации батча даталоадера

    Args:
        loader
    """
    images, label, _ = next(iter(loader))

    grid = torchvision.utils.make_grid(images[:num_img], nrow=num_row, normalize=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


def resize_save_img(args):
    """
    функция для ресайза и сохранения изображения
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
    """
    Создает уменьшенную копию всего датасета

    Args:
        source_dir: исходная папка с изображениями
        target_dir: целевая папка для уменьшенных копий
        img_size: целевой размер
        num_workers: количество процессов
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
