from fastapi import APIRouter, Depends
from app.schemas.lime_schema import LimeRequest, LimeResponse
from app.services.factory import ExplainerFactory

router = APIRouter()
factory = ExplainerFactory()

@router.post("/lime/explain", response_model=LimeResponse)
async def lime_explain(request: LimeRequest):
    explainer = factory.get_explainer("lime")
    result = explainer.explain(request)
    return LimeResponse(**result)
