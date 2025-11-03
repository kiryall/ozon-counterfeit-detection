from torchvision import transforms

# paths
DATA = "data/"
DATA_CSV = f"{DATA}data.csv"
IMG_DIR = f"{DATA}img/"
SUBMISSION_DIR = f"submission/"

# Reproducibility
SEED = 42
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

TEST_SIZE = 0.25
VAL_SIZE = 0.25

# data
TARGET = 'resolution'
ITEM = 'item_id'

# text
TEXT_COLUMN = "description"
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)
TEXT_BATCH_SIZE = 128

# IMG
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# model
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 6,
    "random_seed": SEED,
    "eval_metric": 'Logloss',
    'auto_class_weights': 'Balanced',
    "early_stopping_rounds": 100,
    'l2_leaf_reg': 3,
    "random_strength": 1,
    "bagging_temperature": 0.5,
}

PARAM_GRID_CAT = {
    'iterations': [1000],  # Количество деревьев
    'learning_rate': [0.001, 0.1],  # Скорость обучения
    'depth': [3],  # Глубина деревьев
    'l2_leaf_reg': [1, 2],  # L2-регуляризация
}

MODEL_CACHE_DIR = "model_cache"