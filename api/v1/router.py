# api/v1/router.py
from fastapi import APIRouter
from .endpoints import health, prediction, batch_prediction

router = APIRouter()
router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(prediction.router, prefix="/predict", tags=["prediction"])
router.include_router(batch_prediction.router, prefix="/predict", tags=["prediction"])
