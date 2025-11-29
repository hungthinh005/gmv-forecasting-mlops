"""API request and response schemas"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date


class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    
    city: str = Field(..., description="City name (e.g., 'Ho Chi Minh', 'Hanoi')")
    forecast_horizon: int = Field(4, ge=1, le=52, description="Number of weeks to forecast")
    exogenous_features: Dict[str, List[float]] = Field(
        ...,
        description="Dictionary of exogenous features with lists of values"
    )
    start_date: Optional[date] = Field(None, description="Start date for forecast")
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "Ho Chi Minh",
                "forecast_horizon": 4,
                "exogenous_features": {
                    "x1": [100000, 105000, 110000, 115000],
                    "x2": [50000, 55000, 60000, 65000],
                    "x3": [30000, 32000, 34000, 36000],
                    "x4": [20000, 21000, 22000, 23000],
                    "x5": [10000, 11000, 12000, 13000],
                    "x6": [5000, 5500, 6000, 6500],
                    "x7": [2000, 2200, 2400, 2600]
                },
                "start_date": "2025-07-01"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    
    city: str = Field(..., description="City name")
    forecast_horizon: int = Field(..., description="Number of forecasted periods")
    predictions: List[float] = Field(..., description="Forecasted GMV values")
    dates: List[str] = Field(..., description="Forecast dates")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    individual_predictions: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Individual model predictions"
    )
    
    class Config:
        protected_namespaces = ()  # Allow model_info field
        json_schema_extra = {
            "example": {
                "city": "Ho Chi Minh",
                "forecast_horizon": 4,
                "predictions": [950000.0, 980000.0, 1000000.0, 1020000.0],
                "dates": ["2025-07-07", "2025-07-14", "2025-07-21", "2025-07-28"],
                "model_info": {
                    "model_type": "hybrid",
                    "sarimax_weight": 0.4332,
                    "prophet_weight": 0.5668
                },
                "individual_predictions": {
                    "sarimax": [940000.0, 970000.0, 990000.0, 1010000.0],
                    "prophet": [960000.0, 990000.0, 1010000.0, 1030000.0]
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    
    cities: List[str] = Field(..., description="List of city names")
    forecast_horizon: int = Field(4, ge=1, le=52, description="Number of weeks to forecast")
    exogenous_features: Dict[str, Dict[str, List[float]]] = Field(
        ...,
        description="Dictionary mapping city names to their exogenous features"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="List of loaded models")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    
    city: str = Field(..., description="City name")
    model_type: str = Field(..., description="Model type")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")
    performance: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")

