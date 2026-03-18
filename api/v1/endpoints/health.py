# api/v1/endpoints/health.py
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse


router = APIRouter(
    prefix="/health",
)

@router.get("")
async def health_check():
    """Health check endpoint"""
    from main import model, multimodal_processor
    
    is_healthy = model is not None and multimodal_processor is not None
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "model_loaded": model is not None,
            "processor_loaded": multimodal_processor is not None
        }
    )