from pydantic import BaseModel
from typing import List, Dict, Any
from app.schemas.common import BaseRequest, BaseResponse

class ShapRequest(BaseRequest):
    data: List[List[float]]
    model_input: Dict[str, Any]
    shap_type: str = "window_shap"
    
class ShapResponse(BaseResponse):
    shap_values: List[List[float]]
    base_value: float
    feature_names: List[str]
