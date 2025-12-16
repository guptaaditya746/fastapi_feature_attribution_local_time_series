from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass
