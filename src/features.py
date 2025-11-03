import os
import re
import warnings

from img2vec_pytorch import Img2Vec
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
import torch
from PIL import Image
import torchvision.models as models
from pathlib import Path

import config


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Класс для получения эмбеддингов изображений
    """
    def __init__(self, model_name="resnet50", device=config.DEVICE, batch_size=config.BATCH_SIZE, num_workers=2):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = None
        self.img_columns = []

    def _init_model(self):
        """
        Инициализация модели
        """
        
        # пути для моделей
        model_cache_dir = Path(config.MODEL_CACHE_DIR) / self.model_name.replace("/", "_")
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_cache_dir / "model_weights.pth"

        if model_path.exists():
            print(f'Загрузка модели из кэша {model_path}')
            try:
                if self.model_name == 'resnet50':
                    backbone = models.resnet50(pretrained=False)
                    backbone.fc = nn.Identity()
                else:
                    raise ValueError(f"Модель {self.model_name} не поддерживается")

                backbone.load_state_dict(torch.load(model_path))
                backbone.eval()
                backbone.to(self.device)
                return backbone

            except Exception as e:
                print(f'Ошибка при загрузке модели из кэша {e}')

        print(f'Загрузка {self.model_name}')
        try:
            if self.model_name == 'resnet50':
                backbone = models.resnet50(pretrained=True)
                backbone.fc = nn.Identity()
            else:
                raise ValueError(f"Модель {self.model_name} не поддерживается")

            torch.save(backbone.state_dict(), model_path)
            print(f'💾 Модель сохранена в кэш: {model_path}')

            backbone.eval()
            backbone.to(self.device)
            return backbone

        except Exception as e:
            print(f'Ошибка при загрузке модели {e}')
            raise

    def fit(self, X=None, y=None):
        self.model = self._init_model()
        return self

    def transform(self, dataset, y=None):
        if self.model is None:
            raise ValueError('Сначала нужно вызвать fit()')

        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)

        features = []
        all_ids = []

        print("...Извлечение признаков из изображений...")
        pbar = tqdm(total=len(loader), desc="Batches")

        with torch.no_grad():
            for images, _, ids in loader:
                images = images.to(self.device)
                feats = self.model(images)
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                features.append(feats)

                ids_list = ids.tolist() 
                all_ids.extend(ids_list)

                pbar.update(1)
                pbar.set_postfix({
                    'Текущий батч': len(ids),
                    'Всего изображений': len(all_ids)
                })

        pbar.close()

        features = np.vstack(features)
        print(f"Извлечено {len(features)} признаков размерностью {features.shape[1]}")

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
        """
        Возрращает список извлеченных фич
        """
        return self.img_columns


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """
    Класс для извлечения эмбеддингов из текста описания товара (sentence_transformers)
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device=config.DEVICE, batch_size=config.TEXT_BATCH_SIZE):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.text_columns = []

    def _init_model(self):
        """
        Инициализация модели
        """

        #пути для моделей
        model_cache_dir = Path(config.MODEL_CACHE_DIR) / self.model_name.replace("/", "_")
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        if model_cache_dir.exists() and any(model_cache_dir.iterdir()):
            print(f'Загрузка модели из кэша {model_cache_dir}')
            try:
                self.model = SentenceTransformer(str(model_cache_dir), device=self.device)
                print('Модель загружена из кэша')
                return self.model
            except Exception as e:
                print(f'Ошибка при загрузке модели {e}')

        print('Загрузка Sentence Transformer')
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print('Модель загружена')
            self.model.save(str(model_cache_dir))
            print('Модель сохранена в кэш')
        except Exception as e:
            print(f'Ошибка при загрузке модели {e}')
            raise

        return self.model

    def fit(self, X=None, y=None):
        self.model = self._init_model()
        return self

    def transform(self, df, y=None):
        """
        Извлечение эмбеддингов для текста
        """
        if self.model is None:
            raise ValueError('Сначала нужно вызвать fit()')
        
        texts = df[config.TEXT_COLUMN].tolist()

        print(f"Извлечение эмбеддингов для {len(texts)} текстов...")

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
            print(f"Получены эмбеддинги: {embeddings.shape}")
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
        """
        Возрращает список извлеченных фич
        """
        return self.text_columns
