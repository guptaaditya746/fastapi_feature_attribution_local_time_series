from app.services.base_explainer import BaseExplainer
from typing import Dict, Any

class WindowShapExplainer(BaseExplainer):
    def explain(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Window SHAP for time-series
        return {
            "shap_values": [[0.1, 0.2]],
            "base_value": 0.0,
            "feature_names": ["feature1", "feature2"]
        }
