# models/url_input.py
from pydantic import BaseModel

class URLInput(BaseModel):
    url: str
    # URL должен содержать "ozon.ru" для валидации
    @classmethod
    def validate_url(cls, url: str):
        if "ozon.ru" not in url:
            raise ValueError("неверный URL: должен содержать 'ozon.ru'")
        return url