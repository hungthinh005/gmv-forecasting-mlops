"""FastAPI application for GMV forecasting service"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn
import time

# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, HealthResponse, ModelInfoResponse
)
from src.models.hybrid_model import HybridForecaster
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.api.model_manager import ModelManager

logger = setup_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['city', 'model_type']
)

# Load configuration
config = load_config("config/config.yaml")

# Initialize FastAPI app
app = FastAPI(
    title="GMV Forecasting API",
    description="Production-ready API for GMV forecasting using Hybrid SARIMAX + Prophet model",
    version="1.0.0"
)

# Configure CORS
if config['api']['cors']['enabled']:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config['api']['cors']['allow_origins'],
        allow_credentials=True,
        allow_methods=config['api']['cors']['allow_methods'],
        allow_headers=config['api']['cors']['allow_headers']
    )

# Prometheus middleware to track all requests
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# Initialize model manager
model_manager = ModelManager(config)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting GMV Forecasting API")
    try:
        model_manager.load_all_models()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "GMV Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    loaded_models = model_manager.get_loaded_models()
    
    return HealthResponse(
        status="healthy" if loaded_models else "degraded",
        version="1.0.0",
        models_loaded=loaded_models
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Make GMV forecast for a specific city
    
    Args:
        request: Prediction request with city, horizon, and features
        
    Returns:
        Forecast predictions with model information
    """
    try:
        logger.info(f"Prediction request for {request.city}, horizon={request.forecast_horizon}")
        
        # Get model for city
        model = model_manager.get_model(request.city)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for city: {request.city}"
            )
        
        # Prepare exogenous features
        X_future = pd.DataFrame(request.exogenous_features)
        
        # Validate feature dimensions
        if len(X_future) != request.forecast_horizon:
            raise HTTPException(
                status_code=400,
                detail=f"Feature length ({len(X_future)}) must match forecast_horizon ({request.forecast_horizon})"
            )
        
        # Generate dates
        if request.start_date:
            dates = pd.date_range(
                start=request.start_date,
                periods=request.forecast_horizon,
                freq='W'
            )
        else:
            # Use current date as starting point
            dates = pd.date_range(
                start=pd.Timestamp.now(),
                periods=request.forecast_horizon,
                freq='W'
            )
        
        # Make predictions
        predictions, individual_preds = model.predict(
            n_periods=request.forecast_horizon,
            X=X_future,
            dates=dates
        )
        
        # Get model info
        model_params = model.get_params()
        
        response = PredictionResponse(
            city=request.city,
            forecast_horizon=request.forecast_horizon,
            predictions=predictions.tolist(),
            dates=[d.strftime('%Y-%m-%d') for d in dates],
            model_info={
                "model_type": "hybrid",
                "sarimax_weight": model_params['weights']['sarimax'],
                "prophet_weight": model_params['weights']['prophet']
            },
            individual_predictions={
                "sarimax": individual_preds['sarimax'].tolist(),
                "prophet": individual_preds['prophet'].tolist()
            }
        )
        
        logger.info(f"Prediction completed for {request.city}")
        
        # Increment Prometheus counter
        PREDICTION_COUNT.labels(
            city=request.city,
            model_type="hybrid"
        ).inc()
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Make GMV forecasts for multiple cities
    
    Args:
        request: Batch prediction request
        
    Returns:
        Dictionary of predictions for each city
    """
    try:
        logger.info(f"Batch prediction request for {len(request.cities)} cities")
        
        results = {}
        
        for city in request.cities:
            # Get model
            model = model_manager.get_model(city)
            if model is None:
                logger.warning(f"Model not found for city: {city}")
                results[city] = {"error": "Model not found"}
                continue
            
            # Get features for this city
            city_features = request.exogenous_features.get(city)
            if city_features is None:
                logger.warning(f"No features provided for city: {city}")
                results[city] = {"error": "No features provided"}
                continue
            
            # Prepare data
            X_future = pd.DataFrame(city_features)
            dates = pd.date_range(
                start=pd.Timestamp.now(),
                periods=request.forecast_horizon,
                freq='W'
            )
            
            # Make predictions
            predictions, individual_preds = model.predict(
                n_periods=request.forecast_horizon,
                X=X_future,
                dates=dates
            )
            
            results[city] = {
                "predictions": predictions.tolist(),
                "dates": [d.strftime('%Y-%m-%d') for d in dates],
                "individual_predictions": {
                    "sarimax": individual_preds['sarimax'].tolist(),
                    "prophet": individual_preds['prophet'].tolist()
                }
            }
        
        logger.info("Batch prediction completed")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{city}", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(city: str):
    """Get information about a specific model
    
    Args:
        city: City name
        
    Returns:
        Model information and parameters
    """
    try:
        model = model_manager.get_model(city)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for city: {city}"
            )
        
        params = model.get_params()
        
        return ModelInfoResponse(
            city=city,
            model_type="hybrid_sarimax_prophet",
            parameters=params
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models"""
    return {
        "models": model_manager.get_loaded_models(),
        "total": len(model_manager.get_loaded_models())
    }


@app.post("/models/reload", tags=["Models"])
async def reload_models(background_tasks: BackgroundTasks):
    """Reload all models (useful after model updates)"""
    try:
        background_tasks.add_task(model_manager.load_all_models)
        return {"message": "Model reload initiated"}
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers'],
        reload=config['api']['reload']
    )

