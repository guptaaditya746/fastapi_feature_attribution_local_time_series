from pydantic import BaseModel
from typing import List, Dict, Any
from app.schemas.common import BaseRequest, BaseResponse

class LimeRequest(BaseRequest):
    data: List[List[float]]
    model_input: Dict[str, Any]
    explainer_type: str = "lime_tabular"
    
class LimeResponse(BaseResponse):
    explanation: Dict[str, Any]
    feature_importance: List[Dict[str, float]]
