"""Hybrid SARIMAX + Prophet model with optimized weights"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.sarimax_model import SARIMAXForecaster
from src.models.prophet_model import ProphetForecaster
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridForecaster:
    """Weighted hybrid model combining SARIMAX and Prophet"""
    
    def __init__(
        self,
        sarimax_config: Dict[str, Any],
        prophet_config: Dict[str, Any],
        hybrid_config: Dict[str, Any]
    ):
        """Initialize hybrid forecaster
        
        Args:
            sarimax_config: SARIMAX model configuration
            prophet_config: Prophet model configuration
            hybrid_config: Hybrid optimization configuration
        """
        self.sarimax_model = SARIMAXForecaster(sarimax_config)
        self.prophet_model = ProphetForecaster(prophet_config)
        self.hybrid_config = hybrid_config
        
        self.weights = np.array([0.5, 0.5])  # Initial weights
        self.is_fitted = False
        self.prophet_available = False  # Track if Prophet is working
    
    def fit(
        self,
        y_train: np.ndarray,
        X_train: pd.DataFrame,
        dates_train: pd.DatetimeIndex,
        y_val: np.ndarray = None,
        X_val: pd.DataFrame = None,
        dates_val: pd.DatetimeIndex = None
    ) -> 'HybridForecaster':
        """Fit hybrid model and optimize weights
        
        Args:
            y_train: Training target values
            X_train: Training exogenous features
            dates_train: Training dates
            y_val: Validation target values (for weight optimization)
            X_val: Validation exogenous features
            dates_val: Validation dates
            
        Returns:
            Fitted hybrid model
        """
        logger.info("Fitting hybrid model")
        
        # Fit SARIMAX model
        logger.info("Training SARIMAX component")
        self.sarimax_model.fit(y_train, X_train.values)
        
        # Try to fit Prophet model (optional - may fail on Windows)
        prophet_available = False
        try:
            logger.info("Training Prophet component")
            self.prophet_model.fit(y_train, X_train, dates_train)
            prophet_available = True
            logger.info("Prophet training successful")
        except Exception as e:
            logger.warning(f"Prophet training failed: {str(e)}")
            logger.warning("Continuing with SARIMAX-only model (this is fine!)")
            prophet_available = False
        
        # Optimize weights if validation data provided and Prophet is available
        if prophet_available and y_val is not None and X_val is not None and dates_val is not None:
            logger.info("Optimizing hybrid weights")
            try:
                self._optimize_weights(y_val, X_val, dates_val)
            except Exception as e:
                logger.warning(f"Weight optimization failed: {e}")
                logger.warning("Using SARIMAX-only (weights: [1.0, 0.0])")
                self.weights = np.array([1.0, 0.0])
                prophet_available = False
        elif not prophet_available:
            logger.info("Using SARIMAX-only model (Prophet unavailable on Windows)")
            self.weights = np.array([1.0, 0.0])  # 100% SARIMAX
        else:
            logger.warning("No validation data provided. Using equal weights (0.5, 0.5)")
        
        self.is_fitted = True
        self.prophet_available = prophet_available
        
        if prophet_available:
            logger.info(
                f"Hybrid model fitted. Weights: SARIMAX={self.weights[0]:.4f}, "
                f"Prophet={self.weights[1]:.4f}"
            )
        else:
            logger.info("SARIMAX-only model fitted (Prophet not available)")
        
        return self
    
    def _optimize_weights(
        self,
        y_val: np.ndarray,
        X_val: pd.DataFrame,
        dates_val: pd.DatetimeIndex
    ):
        """Optimize hybrid model weights using validation data
        
        Args:
            y_val: Validation target values
            X_val: Validation exogenous features
            dates_val: Validation dates
        """
        n_periods = len(y_val)
        
        # Get predictions from both models
        sarimax_preds, _ = self.sarimax_model.predict(n_periods, X_val.values)
        prophet_preds, _ = self.prophet_model.predict(n_periods, X_val, dates_val)
        
        # Calculate logarithmic errors (as per the paper)
        e_sarimax = np.log(y_val) - np.log(sarimax_preds)
        e_prophet = np.log(y_val) - np.log(prophet_preds)
        
        # Create error matrix
        error_matrix = np.vstack([e_sarimax, e_prophet]).T
        
        def objective_function(weights):
            """Minimize sum of absolute weighted errors"""
            weighted_errors = np.dot(error_matrix, weights)
            return np.sum(np.abs(weighted_errors))
        
        # Initial weights
        initial_weights = self.hybrid_config.get('initial_weights', [0.5, 0.5])
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = self.hybrid_config.get('bounds', [(0, 1), (0, 1)])
        
        # Optimize
        result = minimize(
            objective_function,
            initial_weights,
            method=self.hybrid_config.get('optimization_method', 'SLSQP'),
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.weights = result.x
            logger.info(f"Weight optimization successful: {self.weights}")
        else:
            logger.warning(f"Weight optimization failed: {result.message}")
    
    def predict(
        self,
        n_periods: int,
        X: pd.DataFrame,
        dates: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Make hybrid predictions
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous features for prediction
            dates: Future dates
            
        Returns:
            Tuple of (hybrid_predictions, individual_predictions_dict)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from SARIMAX (always available)
        sarimax_preds, sarimax_conf = self.sarimax_model.predict(n_periods, X.values)
        
        # Try to get predictions from Prophet (may not be available)
        if hasattr(self, 'prophet_available') and self.prophet_available:
            try:
                prophet_preds, prophet_conf = self.prophet_model.predict(n_periods, X, dates)
                
                # Combine using logarithmic weighted average
                log_sarimax = np.log(sarimax_preds)
                log_prophet = np.log(prophet_preds)
                
                log_hybrid = self.weights[0] * log_sarimax + self.weights[1] * log_prophet
                hybrid_preds = np.exp(log_hybrid)
                
                logger.info(f"Made hybrid predictions for {n_periods} periods")
                
                individual_preds = {
                    'sarimax': sarimax_preds,
                    'prophet': prophet_preds,
                    'sarimax_conf': sarimax_conf,
                    'prophet_conf': prophet_conf
                }
            except Exception as e:
                logger.warning(f"Prophet prediction failed: {e}. Using SARIMAX-only.")
                hybrid_preds = sarimax_preds
                individual_preds = {
                    'sarimax': sarimax_preds,
                    'prophet': sarimax_preds,  # Fallback to SARIMAX
                    'sarimax_conf': sarimax_conf,
                    'prophet_conf': sarimax_conf
                }
        else:
            # Prophet not available, use SARIMAX only
            hybrid_preds = sarimax_preds
            logger.info(f"Made SARIMAX-only predictions for {n_periods} periods")
            
            individual_preds = {
                'sarimax': sarimax_preds,
                'prophet': sarimax_preds,  # Same as SARIMAX when Prophet unavailable
                'sarimax_conf': sarimax_conf,
                'prophet_conf': sarimax_conf
            }
        
        return hybrid_preds, individual_preds
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'sarimax_params': self.sarimax_model.get_params(),
            'prophet_params': self.prophet_model.get_params(),
            'weights': {
                'sarimax': float(self.weights[0]),
                'prophet': float(self.weights[1])
            }
        }
    
    def save(self, path: str):
        """Save hybrid model
        
        Args:
            path: Base path for saving (will create multiple files)
        """
        import joblib
        from pathlib import Path
        
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        joblib.dump(self.sarimax_model, base_path / 'sarimax_model.pkl')
        joblib.dump(self.prophet_model, base_path / 'prophet_model.pkl')
        
        # Save weights and config
        hybrid_data = {
            'weights': self.weights,
            'hybrid_config': self.hybrid_config
        }
        joblib.dump(hybrid_data, base_path / 'hybrid_data.pkl')
        
        logger.info(f"Saved hybrid model to {path}")
    
    @staticmethod
    def load(path: str) -> 'HybridForecaster':
        """Load hybrid model
        
        Args:
            path: Base path for loading
            
        Returns:
            Loaded hybrid model
        """
        import joblib
        from pathlib import Path
        
        base_path = Path(path)
        
        # Load individual models
        sarimax_model = joblib.load(base_path / 'sarimax_model.pkl')
        prophet_model = joblib.load(base_path / 'prophet_model.pkl')
        
        # Load weights and config
        hybrid_data = joblib.load(base_path / 'hybrid_data.pkl')
        
        # Reconstruct hybrid model
        hybrid = HybridForecaster(
            sarimax_config=sarimax_model.model_config,
            prophet_config=prophet_model.model_config,
            hybrid_config=hybrid_data['hybrid_config']
        )
        hybrid.sarimax_model = sarimax_model
        hybrid.prophet_model = prophet_model
        hybrid.weights = hybrid_data['weights']
        hybrid.is_fitted = True
        
        logger.info(f"Loaded hybrid model from {path}")
        
        return hybrid

