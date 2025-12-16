from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "XAI Time-Series Service"
    api_key: Optional[str] = None
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
