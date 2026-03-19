# Класс для получения эмбеддингов изображений и текста

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import core.config as config
from core.logging import setup_logging
 
# Настройка логирования
logger = setup_logging(log_file="features.log", console=True, remove_file=True, logger_name="features")


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """Класс для извлечения эмбеддингов из изображений.

    Использует предобученные модели (ResNet, MobileNet) для извлечения
    визуальных признаков из изображений.
    """

    MODEL_REGISTRY = {
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        ),
    }

    def __init__(self, model_name: str = "resnet18", device=config.DEVICE, batch_size=config.BATCH_SIZE, num_workers=2):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = None
        self.img_columns = []

    def _build_backbone(self) -> nn.Module:
        """Создание backbone модели через реестр.

        Создает экземпляр модели на основе заданного имени модели.

        Returns:
            Модель без классификационного слоя.

        Raises:
            ValueError: Модель не поддерживается или неизвестная архитектура.
        """
        if self.model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model = self.MODEL_REGISTRY[self.model_name]()

        # Унификация: убираем classifier / fc
        if hasattr(model, "fc"):  # resnet
            model.fc = nn.Identity()

        elif hasattr(model, "classifier"):  # mobilenet / efficientnet
            model.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown model architecture: {self.model_name}")

        return model

    def _init_model(self):
        """Инициализация модели для извлечения признаков.

        Загружает предобученную модель и переводит её в режим оценки.

        Returns:
            Инициализированная модель.
        """
        logger.info(f"Loading model: {self.model_name}")

        model = self._build_backbone()
        model.eval()
        model.to(self.device)

        return model

    def fit(self, X=None, y=None):
        self.model = self._init_model()
        return self

    def transform(self, dataset, y=None):
        """Извлечение признаков из изображений.

        Обрабатывает датасет изображений и извлекает визуальные признаки
        с помощью предобученной модели.

        Args:
            dataset: Датасет с изображениями.
            y: Целевая переменная (не используется).

        Returns:
            DataFrame с извлеченными признаками изображений.

        Raises:
            ValueError: Модель не инициализирована (не вызван fit()).
        """
        if self.model is None:
            raise ValueError('Сначала нужно вызвать fit()')

        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)

        features = []
        all_ids = []

        logger.info("...Извлечение признаков из изображений...")
        pbar = tqdm(total=len(loader), desc="Batches", leave=False, position=0)

        with torch.no_grad():
            for images, _, ids in loader:
                images = images.to(self.device)
                feats = self.model(images)
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                features.append(feats)

                # Обработка ids: поддержка различных типов (list, tuple, tensor, scalar)
                if isinstance(ids, (list, tuple)):
                    ids_list = list(ids)
                elif hasattr(ids, 'tolist'):  # torch.Tensor
                    ids_list = ids.tolist()
                else:
                    ids_list = [ids]
                
                all_ids.extend(ids_list)

                pbar.update(1)
                pbar.set_postfix({
                    'Текущий батч': len(ids),
                    'Всего изображений': len(all_ids)
                })

        pbar.close()

        features = np.vstack(features)
        logger.info(f"Извлечено {len(features)} признаков размерностью {features.shape[1]}")

        # список колонок с фичами изображений
        self.img_columns = [f'img_emb_{i}' for i in range(features.shape[1])]

        # преобразование в датафрейм
        features_dataframe = pd.DataFrame(
            features,
            index=all_ids,
            columns=self.img_columns,
        )
        return features_dataframe

    def fit_transform(self, dataset, y=None):
        self.fit()
        return self.transform(dataset)
    
    def get_img_features(self):
        """Получение списка извлеченных признаков изображений.

        Returns:
            Список имен колонок признаков изображений.
        """
        return self.img_columns


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Класс для извлечения эмбеддингов из текста описания товара.

    Использует Sentence Transformers для получения семантических
    представлений текстовых описаний.
    """
    def __init__(self,
                 model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 device=config.DEVICE,
                 batch_size=config.TEXT_BATCH_SIZE):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.text_columns = []

    def _init_model(self):
        """Инициализация модели Sentence Transformers.

        Загружает модель из кэша или скачивает её при первом запуске.
        Сохраняет модель в кэш для последующего использования.

        Returns:
            Инициализированная модель SentenceTransformer.
        """
        model_cache_dir = Path(config.MODEL_CACHE_DIR) / self.model_name.replace("/", "_")
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        if model_cache_dir.exists() and any(model_cache_dir.iterdir()):
            logger.info(f'Загрузка модели из кэша {model_cache_dir}')
            try:
                self.model = SentenceTransformer(str(model_cache_dir), device=self.device)
                logger.info('Модель загружена из кэша')
                return self.model
            except Exception as e:
                logger.error(f'Ошибка при загрузке модели {e}')

        logger.info('Загрузка Sentence Transformer')
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info('Модель загружена')
            self.model.save(str(model_cache_dir))
            logger.info('Модель сохранена в кэш')
        except Exception as e:
            logger.error(f'Ошибка при загрузке модели {e}')
            raise

        return self.model

    def fit(self, X=None, y=None):
        self.model = self._init_model()
        return self

    def transform(self, df, y=None):
        """Извлечение эмбеддингов из текстовых описаний.

        Обрабатывает текстовые описания и преобразует их в
        семантические векторы с помощью Sentence Transformer.

        Args:
            df: DataFrame с текстовыми данными.
            y: Целевая переменная (не используется).

        Returns:
            DataFrame с извлеченными текстовыми эмбеддингами.

        Raises:
            ValueError: Модель не инициализирована (не вызван fit()).
            Exception: Ошибка при извлечении эмбеддингов.
        """
        if self.model is None:
            raise ValueError('Сначала нужно вызвать fit()')
        
        texts = df[config.TEXT_COLUMN].tolist()

        logger.info(f"Извлечение эмбеддингов для {len(texts)} текстов...")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True
            )

            # список колонок с фичами текста
            self.text_columns = [f'text_emb_{i}' for i in range(embeddings.shape[1])]

            # преобразование в датафрейм
            logger.info(f"Получены эмбеддинги: {embeddings.shape}")
            embeddings_dataframe = pd.DataFrame(
                embeddings,
                index=df.index,
                columns=self.text_columns
            )

            return embeddings_dataframe
        except Exception as e:
            print(f"Ошибка при извлечении эмбеддингов: {e}")
            raise

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)   

    def get_text_features(self):
        """Получение списка извлеченных текстовых признаков.

        Returns:
            Список имен колонок текстовых признаков.
        """
        return self.text_columns
