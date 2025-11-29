"""SARIMAX forecasting model"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pmdarima.arima import auto_arima

from src.models.base_model import BaseForecaster
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SARIMAXForecaster(BaseForecaster):
    """SARIMAX forecasting model"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize SARIMAX forecaster
        
        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(
        self,
        y_train: np.ndarray,
        X_train: np.ndarray = None
    ) -> 'SARIMAXForecaster':
        """Fit SARIMAX model with automatic order selection
        
        Args:
            y_train: Training target values
            X_train: Training exogenous features
            
        Returns:
            Fitted model
        """
        logger.info("Fitting SARIMAX model")
        
        # Scale exogenous features
        X_scaled = None
        if X_train is not None:
            X_scaled = self.scaler.fit_transform(X_train)
            logger.info(f"Scaled {X_train.shape[1]} exogenous features")
        
        # Get seasonal periods to try
        seasonal_periods = self.model_config.get('seasonal_periods', [12])
        
        best_model = None
        best_aic = float('inf')
        best_m = None
        
        # Try different seasonal periods
        for m in seasonal_periods:
            try:
                logger.info(f"Trying seasonal period m={m}")
                
                model_try = auto_arima(
                    y=y_train,
                    X=X_scaled,
                    seasonal=True,
                    m=m,
                    start_p=1,
                    start_q=1,
                    max_p=self.model_config.get('max_p', 5),
                    max_q=self.model_config.get('max_q', 5),
                    start_P=0,
                    start_Q=0,
                    max_P=self.model_config.get('max_P', 5),
                    max_Q=self.model_config.get('max_Q', 5),
                    d=None,
                    D=None,
                    stepwise=self.model_config.get('stepwise', True),
                    trace=False,
                    suppress_warnings=self.model_config.get('suppress_warnings', True),
                    error_action='ignore',
                    with_intercept='auto',
                    information_criterion=self.model_config.get('information_criterion', 'aicc')
                )
                
                current_aic = model_try.aic()
                
                if current_aic < best_aic:
                    best_model = model_try
                    best_aic = current_aic
                    best_m = m
                    logger.info(f"New best model found: m={m}, AIC={current_aic:.2f}")
                
            except Exception as e:
                logger.warning(f"Model fitting failed for m={m}: {e}")
        
        if best_model is None:
            raise ValueError("No valid SARIMAX model could be fitted")
        
        self.model = best_model
        self.is_fitted = True
        
        logger.info(
            f"Best SARIMAX model: m={best_m}, "
            f"order={best_model.order}, "
            f"seasonal_order={best_model.seasonal_order}, "
            f"AIC={best_aic:.2f}"
        )
        
        return self
    
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale exogenous features
        X_scaled = None
        if X is not None:
            X_scaled = self.scaler.transform(X)
        
        # Make predictions
        preds, conf_int = self.model.predict(
            n_periods=n_periods,
            X=X_scaled,
            return_conf_int=True
        )
        
        logger.info(f"Made predictions for {n_periods} periods")
        
        return preds, conf_int
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if not self.is_fitted:
            return {}
        
        return {
            'order': self.model.order,
            'seasonal_order': self.model.seasonal_order,
            'aic': self.model.aic(),
            'aicc': self.model.aicc(),
            'bic': self.model.bic()
        }

