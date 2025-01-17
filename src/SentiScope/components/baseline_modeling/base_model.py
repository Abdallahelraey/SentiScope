from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseModel(ABC):
    """Abstract base class defining the interface for all models"""
    
    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        """Train the model on the given data"""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the model's current parameters"""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> None:
        """Set the model's parameters"""
        pass