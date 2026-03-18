# 🚀 Ozon Counterfeit Detection

Production-ready ML system for detecting counterfeit products using multimodal data (tabular + text + images).

## 💡 Problem

Контрафакт на маркетплейсах:

- снижает доверие пользователей
- бьёт по брендам
- создаёт финансовые потери

Задача: автоматически определять подозрительные товары.

## 🎯 Solution

Мультимодальная ML-система, которая объединяет:

- 📊 Tabular features (цена, рейтинг, отзывы)
- 📝 Text embeddings (описания)
- 🖼 Image embeddings (изображения)

→ и предсказывает вероятность контрафакта.

## 🧠 Key Features

- ✅ Multimodal ML pipeline
- ✅ REST API (FastAPI)
- ✅ Batch inference
- ✅ Feature extraction pipeline
- ✅ Reproducible training
- ✅ Logging & monitoring
- ✅ Clean project structure (production-like)

## 🏗️ Architecture (High-level)

Client → FastAPI → Prediction Service → Feature Pipeline → Model → Response

Detailed:
```
            ┌──────────────┐
            │   Client     │
            └──────┬───────┘
                   │
                   ▼
        ┌──────────────────────┐
        │      FastAPI API     │
        └─────────┬────────────┘
                  ▼
        ┌──────────────────────┐
        │ Prediction Service   │
        └─────────┬────────────┘
                  ▼
        ┌──────────────────────┐
        │ Feature Engineering  │
        │ - text embeddings    │
        │ - image embeddings   │
        │ - tabular features   │
        └─────────┬────────────┘
                  ▼
        ┌──────────────────────┐
        │   ML Model (Fusion)  │
        └─────────┬────────────┘
                  ▼
            Prediction
```

## 🤖 Model

### Multimodal Architecture

| Modality | Model |
|----------|-------|
| Tabular | CatBoost |
| Text | Sentence Transformers |
| Image | ResNet50 |

Fusion strategy:
- Early fusion (concatenation embeddings)
- Feeding into gradient boosting

## 📊 Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9208 |
| F1-score | 0.9399 |
| Precision | 0.9416 |
| Recall | 0.9384 |

## ⚙️ Tech Stack

- Python 3.11+
- FastAPI
- CatBoost
- PyTorch / torchvision
- Sentence Transformers
- Pandas / NumPy / scikit-learn
- uv (package manager)

## 📁 Project Structure (Production-oriented)

```
api/        # REST API layer
services/   # business logic
models/     # schemas
core/       # config, logging, exceptions
utils/      # feature engineering
training/   # model training
ml_models/  # saved models
```

## 🔄 ML Pipeline

### Training pipeline
```
raw data
  ↓
preprocessing
  ↓
feature extraction
  ↓
train/val split
  ↓
model training
  ↓
evaluation
  ↓
model saving
```

### Inference pipeline
```
input JSON
  ↓
validation
  ↓
feature extraction
  ↓
model inference
  ↓
response
```

## 🚀 Quick Start

```bash
git clone https://github.com/kiryall/ml-ozon-counterfeit.git
cd kachectvo-zadachi-7054

uv venv
source .venv/bin/activate

uv sync
uv run python main.py
```

## 🔌 API

### Predict
- POST /api/v1/predict

### Batch
- POST /api/v1/batch

### Health
- GET /api/v1/health

## 📦 Example Request

```json
{
  "features": {
    "price": 1500,
    "rating": 4.5
  },
  "text": "Описание товара",
  "image_url": "..."
}
```

## 🧪 Reproducibility

- фиксированные зависимости (uv.lock)
- конфиги (core/config.py)
- разделение train/inference
- сохранённые модели

## 📈 Future Improvements

- Add model versioning (MLflow)
- Add monitoring (Prometheus + Grafana)
- Improve fusion (deep multimodal model)
- Add A/B testing
- Optimize inference latency
- Add async batch processing (Celery / Kafka)