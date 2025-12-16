from abc import ABC, abstractmethod
from typing import Dict, Any
from app.services.base_explainer import BaseExplainer
from app.services.explainers import lime_impl, window_shap_impl, shats_impl, etsc_impl

class ExplainerFactory:
    _explainers: Dict[str, BaseExplainer] = {}
    
    def __init__(self):
        self._explainers["lime"] = lime_impl.LimeExplainer()
        self._explainers["window_shap"] = window_shap_impl.WindowShapExplainer()
        self._explainers["shats"] = shats_impl.ShatsExplainer()
        self._explainers["etsc"] = etsc_impl.EtscExplainer()
    
    def get_explainer(self, explainer_type: str) -> BaseExplainer:
        return self._explainers.get(explainer_type)
