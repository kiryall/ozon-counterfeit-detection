import os
import multiprocessing as mp
from PIL import Image
from pathlib import Path

from tqdm import tqdm
from src.data_loader import load_data
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings

import matplotlib.pyplot as plt
import torchvision
import config

def exploratory_data_analysis(df):
    """
    Создает отчет EDA

    Args:
        df (pd.DataFrame): исходные данные

    """

    report = ProfileReport(
        df, minimal=True, plot={"dpi": 200, "image_format": "png"}
    )

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
    plt.axis('off')
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
        print(f'Ошибка при сохранении {e}')
        return False
    
def create_resized_dataset(source_path, target_path, img_size=config.IMG_SIZE, num_workers=8):
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
    print(f'Найдено {len(img_paths)} изображений')

    tasks = []
    for img_path in img_paths:
        target = target_path / img_path.name
        tasks.append((img_path, target, img_size))

    print('Start resize...')
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(resize_save_img, tasks),
            total=len(tasks),
            desc='Resize images'
        ))

    success_count = sum(results)
    print(f"Успешно обработано: {success_count}/{len(tasks)}")
    return success_count