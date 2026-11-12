from fastapi import APIRouter
from app.api.v1.endpoints.documents import router as documents_router

api_router = APIRouter(prefix="/v1")
api_router.include_router(documents_router)
