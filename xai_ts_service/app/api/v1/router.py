from fastapi import APIRouter
from app.api.v1 import lime_routes, shap_routes

router = APIRouter()

router.include_router(lime_routes.router, tags=["lime"])
router.include_router(shap_routes.router, tags=["shap"])
