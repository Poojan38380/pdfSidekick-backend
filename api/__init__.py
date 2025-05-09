from fastapi import APIRouter
# # from .users import router as users_router
from .pdfs import router as pdfs_router

api_router = APIRouter()

# # api_router.include_router(users_router, prefix="/users", tags=["Users"])
api_router.include_router(pdfs_router, prefix="/pdfs", tags=["PDFs"])

__all__ = ["api_router"] 