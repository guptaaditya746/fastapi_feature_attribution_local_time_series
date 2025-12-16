from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class BaseRequest(BaseModel):
    timestamp: Optional[datetime] = None
    
class BaseResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    timestamp: datetime = datetime.utcnow()
