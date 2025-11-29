"""Base model interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for forecasting models"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize forecaster
        
        Args:
            model_config: Model configuration dictionary
        """
        self.model_config = model_config
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray = None
    ) -> 'BaseForecaster':
        """Fit the model
        
        Args:
            y_train: Training target values
            X_train: Training exogenous features
            
        Returns:
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        n_periods: int,
        X: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous features for prediction
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    def save(self, path: str):
        """Save model to disk
        
        Args:
            path: Path to save model
        """
        import joblib
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'BaseForecaster':
        """Load model from disk
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model
        """
        import joblib
        return joblib.load(path)


class ModelArtifacts:
    """Container for model artifacts"""
    
    def __init__(
        self,
        model: Any,
        scaler: Any = None,
        feature_names: list = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize model artifacts
        
        Args:
            model: Trained model
            scaler: Feature scaler
            feature_names: List of feature names
            metadata: Additional metadata
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.metadata = metadata or {}
    
    def save(self, path: str):
        """Save artifacts to disk
        
        Args:
            path: Path to save artifacts
        """
        import joblib
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'ModelArtifacts':
        """Load artifacts from disk
        
        Args:
            path: Path to saved artifacts
            
        Returns:
            Loaded artifacts
        """
        import joblib
        return joblib.load(path)

