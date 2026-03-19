# models/response.py
from pydantic import BaseModel
from typing import Optional

class APIResponse(BaseModel):
    """Базовая модель ответа API.

    Содержит информацию об успешности операции, сообщение
    и дополнительные данные.
    """
    success: bool
    message: str
    data: Optional[dict] = None