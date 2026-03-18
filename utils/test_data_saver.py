# utils/test_data_saver.py
import os
import shutil
import pandas as pd
from pathlib import Path
from core.config import DATA_CSV, IMG_DIR
from core.logging import setup_logging
from utils.data_utils import load_data

logger = setup_logging(log_file="test_data_saver.log", console=True, remove_file=True, logger_name="test_data_saver")

def save_single_test_data(
                          img_dir: str = IMG_DIR,
                          csv_path: str = DATA_CSV,
                          row_index: int = 0,
                          output_dir: str = "test_data/single/"):
    """
    Из data.csv сохраняет одну строку в test_data/single/test_data.csv и одну картинку в test_data/single/
    """

    os.makedirs(output_dir, exist_ok=True)

    path = Path(csv_path)

    df = load_data(path)

    try:
        df.iloc[[row_index]].to_csv(os.path.join(output_dir, "test_data.csv"))
    except Exception as e:
        raise ValueError(f"Error saving CSV data: {e}")
    
    img_filename = str(df.iloc[row_index]['item_id'])
    logger.info(f"Saving image with filename: {img_filename}")
    
    img_path = None
    extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    
    for ext in extensions:
        potential_path = os.path.join(img_dir, f"{img_filename}{ext}")
        if os.path.exists(potential_path):
            img_path = potential_path
            break
    
    if img_path is None:
        raise FileNotFoundError(f"Image file with base name '{img_filename}' not found in {img_dir}")
    
    # Копируем файл (исправлено: shutil.copy2 вместо os.copy)
    shutil.copy2(img_path, os.path.join(output_dir, os.path.basename(img_path)))
    logger.info(f"Image saved: {os.path.basename(img_path)}")
    

if __name__ == "__main__":
    save_single_test_data(img_dir = IMG_DIR,
                          csv_path = DATA_CSV,
                          row_index = 0,
                          output_dir = "test_data/single/")