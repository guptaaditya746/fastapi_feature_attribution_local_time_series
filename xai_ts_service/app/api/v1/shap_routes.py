from fastapi import APIRouter, Depends
from app.schemas.shap_schema import ShapRequest, ShapResponse
from app.services.factory import ExplainerFactory

router = APIRouter()
factory = ExplainerFactory()

@router.post("/shap/explain", response_model=ShapResponse)
async def shap_explain(request: ShapRequest):
    explainer = factory.get_explainer("shap")
    result = explainer.explain(request)
    return ShapResponse(**result)
