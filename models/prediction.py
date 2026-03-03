# models/prediction.py
from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    url: str
    prediction: str  # FAKE или REAL
    confidence: Optional[float] = None