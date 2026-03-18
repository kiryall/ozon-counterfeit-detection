# core/logging.py
# Модуль для настройки логирования в приложении
import logging
import os

from .config import LOG_DIR


def setup_logging(
    log_file: str = "app.log",
    level=logging.INFO,
    console=True,
    remove_file=False,
    logger_name: str = None,
):
    """
    Настройка логирования с выводом в файл и опционально в консоль.

    :param log_file: Имя файла лога
    :param level: Уровень логирования
    :param console: Выводить ли логи в консоль
    :param remove_file: Удалить ли существующий файл лога перед настройкой
    :param logger_name: Имя логгера (если None, используется глобальный логгер)
    :return: Объект logger
    """
    log_path = os.path.join(LOG_DIR, log_file)

    if remove_file and os.path.exists(log_path):
        try:
            os.remove(log_path)
        except PermissionError:
            # File is locked by another process, continue without removing
            pass

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # Запретить распространение в родительские логгеры

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(
        log_path, mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger