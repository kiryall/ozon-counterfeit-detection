# Ozon Counterfeit Detection

ML pipeline for detecting counterfeit products on Ozon using multimodal data (tabular, text, images).

## Project Structure
- `main.py` - основное FastAPI приложение
- `api/` - API маршруты и эндпоинты
- `core/` - конфигурация и исключения приложения
- `models/` - Pydantic модели для валидации данных
- `services/` - бизнес-логика приложения
- `utils/` - утилиты для обработки данных и моделей
- `traning/` - скрипты обучения модели
- `notebooks/` - Jupyter ноутбуки для анализа и демонстраций
- `data/` - данные (игнорируется git)
- `ml_model/` - обученные модели (игнорируется git)
- `submission/` - результаты сабмитов
- `static/` - статические файлы (HTML, CSS, JS)

## Quick Start
```bash
git clone https://github.com/kiryall/ml-ozon-counterfeit.git
cd ml-ozon-counterfeit
pip install -r requirements.txt
python main.py
```

## Структура проекта
```app/
├── main.py                     # Основное FastAPI приложение
├── static/
│   └── index.html              # HTML интерфейс приложения
├── api/
│   ├── __init__.py
│   └── v1/
│       ├── __init__.py
│       ├── router.py           # Определение маршрутов API
│       └── endpoints/
│           ├── __init__.py
│           ├── prediction.py   # Эндпоинты для предсказаний
│           ├── url_prediction.py  # Эндпоинты для предсказаний по URL
│           └── health.py       # Эндпоинты для проверки состояния
├── models/
│   ├── __init__.py
│   ├── prediction.py           # Pydantic модели для валидации данных
│   ├── url_input.py            # Модель для URL-входа
│   └── response.py             # Модели ответов API
├── services/
│   ├── __init__.py
│   ├── prediction_service.py   # Логика предсказания
│   ├── model_loader.py         # Загрузка и управление моделью
│   ├── scraping_service.py     # Веб-скрапинг
│   └── llm_parser.py           # LLM-парсинг
├── core/
│   ├── __init__.py
│   ├── config.py               # Конфигурация приложения
│   └── exceptions.py           # Обработка исключений
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Утилиты для предобработки
│   ├── data_utils.py           # Утилиты для работы с данными
│   ├── features.py             # Утилиты для извлечения признаков
│   ├── model.py                # Модель для многомодального классификатора
│   ├── multimodal.py           # Многомодальные функции
│   └── preprocessing.py        # Утилиты для предобработки данных
├── traning/
│   ├── __init__.py
│   └── train.py                # Скрипт обучения модели
├── notebooks/
│   ├── EDA.ipynb               # Исследовательский анализ данных
│   └── main.ipynb              # Основной ноутбук с анализом
├── data/
│   └── (данные игнорируются git)
├── ml_model/
│   └── (обученные модели игнорируются git)
├── submission/
│   └── (файлы сабмитов)
└── requirements.txt
```