"""Prophet forecasting model"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.feature_selection import SelectKBest, f_regression

from src.models.base_model import BaseForecaster
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet forecasting model"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize Prophet forecaster
        
        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.feature_names = None
        self.selected_features = None
        self.max_features = 5  # Limit to 5 features for Windows stability
    
    def fit(
        self,
        y_train: np.ndarray,
        X_train: pd.DataFrame = None,
        dates: pd.DatetimeIndex = None
    ) -> 'ProphetForecaster':
        """Fit Prophet model
        
        Args:
            y_train: Training target values
            X_train: Training exogenous features (as DataFrame)
            dates: Date index for training data
            
        Returns:
            Fitted model
        """
        logger.info("Fitting Prophet model")
        
        # Prepare data in Prophet format
        if dates is None:
            raise ValueError("Prophet requires dates to be provided")
        
        # Prepare Prophet dataframe (matching notebook approach)
        df_prophet = pd.DataFrame({
            'ds': dates,
            'y': y_train
        })
        
        # Add all exogenous features (notebook uses all features)
        if X_train is not None:
            self.feature_names = X_train.columns.tolist()
            for col in self.feature_names:
                df_prophet[col] = X_train[col].values
        
        # Initialize Prophet model
        self.model = Prophet(
            seasonality_mode=self.model_config.get('seasonality_mode', 'multiplicative'),
            interval_width=self.model_config.get('interval_width', 0.95)
        )
        logger.info(f"Prophet model created (has stan_backend: {hasattr(self.model, 'stan_backend')})")
        
        # Add country holidays (optional)
        try:
            country = self.model_config.get('country_holidays')
            if country:
                self.model.add_country_holidays(country_name=country)
                logger.info(f"Added holidays for {country}")
        except Exception as e:
            logger.warning(f"Could not add country holidays: {e}")
        
        # Add simplified custom seasonalities (reduced fourier orders)
        try:
            custom_seasonalities = self.model_config.get('custom_seasonalities', [])
            for seasonality in custom_seasonalities:
                # Reduce fourier order for stability
                fourier_order = min(seasonality['fourier_order'], 3)
                self.model.add_seasonality(
                    name=seasonality['name'],
                    period=seasonality['period'],
                    fourier_order=fourier_order
                )
                logger.info(f"Added custom seasonality: {seasonality['name']} (order={fourier_order})")
        except Exception as e:
            logger.warning(f"Could not add custom seasonalities: {e}")
        
        # Add exogenous regressors (matching notebook: multiplicative mode)
        if self.feature_names:
            regressor_mode = self.model_config.get('regressor_mode', 'multiplicative')
            for feature in self.feature_names:
                try:
                    self.model.add_regressor(feature, mode=regressor_mode)
                except Exception as e:
                    logger.warning(f"Could not add regressor {feature}: {e}")
            
            logger.info(f"Added {len(self.feature_names)} regressors")
        
        # Fit the model
        try:
            logger.info("Fitting Prophet model...")
            self.model.fit(df_prophet)
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
        except Exception as e:
            logger.error(f"Prophet fitting failed: {str(e)}")
            raise RuntimeError(
                f"Prophet model fitting failed. This may be due to:\n"
                f"  1. Too many features (reduce max_features)\n"
                f"  2. Data issues (check for NaN/inf values)\n"
                f"  3. Windows compatibility (try on Linux/Docker)\n"
                f"Original error: {str(e)}"
            )
        
        return self
    
    def predict(
        self,
        n_periods: int,
        X: pd.DataFrame = None,
        dates: pd.DatetimeIndex = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions
        
        Args:
            n_periods: Number of periods to forecast
            X: Exogenous features for prediction (as DataFrame)
            dates: Future dates for prediction
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare future dataframe
        if dates is None:
            raise ValueError("Prophet requires dates to be provided for prediction")
        
        future_df = pd.DataFrame({'ds': dates})
        
        # Add exogenous features (only the selected ones)
        if X is not None and self.feature_names:
            # Only use features that were selected during training
            for col in self.feature_names:
                if col in X.columns:
                    future_df[col] = X[col].values
                else:
                    logger.warning(f"Feature {col} not found in prediction data, using 0")
                    future_df[col] = 0
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        predictions = forecast['yhat'].values
        conf_int = forecast[['yhat_lower', 'yhat_upper']].values
        
        logger.info(f"Made predictions for {n_periods} periods")
        
        return predictions, conf_int
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if not self.is_fitted:
            return {}
        
        return {
            'seasonality_mode': self.model.seasonality_mode,
            'interval_width': self.model.interval_width,
            'changepoint_prior_scale': self.model.changepoint_prior_scale,
            'seasonality_prior_scale': self.model.seasonality_prior_scale,
            'holidays_prior_scale': self.model.holidays_prior_scale
        }

