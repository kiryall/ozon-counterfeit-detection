# main.py

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# Импорт функции загрузки модели и конфигурации
from services.model_loader import load_model, load_multimodal_processor
from core.config import MODEL_PATH, MULTIMODAL_PROCESSOR_PATH
from core.logging import setup_logging
from api.v1.router import router as api_router

# Настройка логирования
logger = setup_logging(log_file="main_app.log", console=True, remove_file=True, logger_name="main")

# Глобальная переменная для модели
model = None
multimodal_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения FastAPI.

    Загружает модель и мультимодальный процессор при запуске приложения
    и освобождает ресурсы при завершении работы.

    Args:
        app: Экземпляр приложения FastAPI.

    Yields:
        Управление передается приложению.

    Raises:
        FileNotFoundError: Файл модели не найден.
        Exception: Ошибка загрузки модели.
    """
    global model, multimodal_processor
    # Загрузка модели при запуске приложения
    try:
        logger.info(f"Загрузка модели из {MODEL_PATH}")
        model = await asyncio.get_event_loop().run_in_executor(
            None,
            load_model,
            MODEL_PATH
        )
        multimodal_processor = await asyncio.get_event_loop().run_in_executor(
            None,
            load_multimodal_processor,
            MULTIMODAL_PROCESSOR_PATH
        )
        logger.info("Модель и процессор успешно загружены")
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

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Регистрируем маршруты API
app.include_router(api_router, prefix="/api/v1")

# Роут для главной страницы с формой ввода URL
@app.get('/', response_class=HTMLResponse)
async def index(
    request: Request,
    result: str | None = None):
    """Отображение главной страницы с формой ввода URL.

    Возвращает HTML-шаблон главной страницы с формой для загрузки
    изображения и данных для предсказания.

    Args:
        request: Объект запроса FastAPI.
        result: Результат предсказания для отображения на странице.

    Returns:
        HTML-ответ с шаблоном страницы.
    """
    logger.info("Запрос главной страницы")
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск приложения")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)