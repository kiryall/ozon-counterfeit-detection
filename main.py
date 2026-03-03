# main.py

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, HTTPException, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# Импорт функции загрузки модели и конфигурации
from models.url_input import URLInput
from services.model_loader import load_model
from core.config import MODEL_PATH


# Настройка логирования
logging.basicConfig(
    filename='log/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Глобальная переменная для модели
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Загрузка модели при запуске приложения
    try:
        logger.info(f"Загрузка модели из {MODEL_PATH}")
        model = await asyncio.get_event_loop().run_in_executor(
            None,
            load_model,
            MODEL_PATH
        )
        logger.info("Модель успешно загружена")
        yield
    except FileNotFoundError as e:
        logger.error(f"Файл модели не найден: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise
    finally:
        logger.info("Приложение завершает работу")

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="static")


class Item(BaseModel):
    """
    Модель данных для входящего запроса.
    В данном случае ожидается URL для классификации.
    """
    url: str


# Роут для главной страницы с формой ввода URL
@app.get('/', response_class=HTMLResponse)
async def index(
    request: Request,
    result: str | None = None):

    logger.info("Запрос главной страницы")
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# Роут для обработки формы и предсказания класса (FAKE/REAL)
@app.post("/predict", tags=["prediction"])
async def predict(url: Annotated[str, Form()]):
    
    try:
        url = URLInput.validate_url(url)
    except ValueError as e:
        logger.error(f"Ошибка валидации URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    global model
    logger.info(f"Получен запрос на предсказание для URL: {url}")
    
    if model is None:
        logger.error("Попытка сделать предсказание, но модель не загружена")
        raise HTTPException(status_code=503,
                            detail="Модель не загружена. Пожалуйста, попробуйте позже.")
    
    try:
        # TODO: реализовать предсказание на основе URL
        # 1. scrapping данных с URL
        # 2. llm парсер для формирования данных для модели
        # 3. предсказание модели на основе полученных данных
        # Временная заглушка для демонстрации
        prediction = 'FAKE' if url == 'test' else 'REAL'
        
        logger.info(f"Предсказание для {url}: {prediction}")
        
    except Exception as e:
        logger.error(f"Ошибка во время предсказания для URL {url}: {e}")
        raise HTTPException(status_code=500,
                            detail="Ошибка во время предсказания")
    
    return RedirectResponse(url=f'/?result={prediction}', status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск приложения")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)