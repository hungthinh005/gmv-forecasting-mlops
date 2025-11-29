"""Model training pipeline with MLflow tracking"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader, create_train_test_sets
from src.models.hybrid_model import HybridForecaster
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error

logger = setup_logger(__name__)


def train_city_model(
    city: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Dict,
    model_config: Dict
) -> HybridForecaster:
    """Train hybrid model for a specific city
    
    Args:
        city: City name
        train_data: Training dataset
        test_data: Test/validation dataset
        config: Main configuration
        model_config: Model-specific configuration
        
    Returns:
        Trained hybrid model
    """
    logger.info(f"Training model for {city}")
    
    # Prepare data
    target_col = config['data']['target_column']
    city_col = config['data']['city_column']
    exog_features = config['data']['exogenous_features']
    
    # Extract features and target
    y_train = train_data[target_col].values
    X_train = train_data[exog_features]
    dates_train = train_data.index
    
    y_test = test_data[target_col].values
    X_test = test_data[exog_features]
    dates_test = test_data.index
    
    logger.info(
        f"Training data: {len(y_train)} samples, "
        f"Test data: {len(y_test)} samples, "
        f"Features: {len(exog_features)}"
    )
    
    # Get model configurations
    city_key = city.replace(" ", "_")
    city_model_config = model_config['models'].get(city_key, {})
    
    sarimax_config = config['training']['sarimax'].copy()
    sarimax_config.update(city_model_config.get('sarimax', {}))
    
    prophet_config = config['training']['prophet'].copy()
    prophet_config.update(city_model_config.get('prophet', {}))
    
    hybrid_config = config['training']['hybrid'].copy()
    hybrid_config.update(city_model_config.get('hybrid', {}))
    
    # Initialize hybrid model
    model = HybridForecaster(
        sarimax_config=sarimax_config,
        prophet_config=prophet_config,
        hybrid_config=hybrid_config
    )
    
    # Train model with validation data for weight optimization
    model.fit(
        y_train=y_train,
        X_train=X_train,
        dates_train=dates_train,
        y_val=y_test,
        X_val=X_test,
        dates_val=dates_test
    )
    
    logger.info(f"Model training completed for {city}")
    
    return model


def evaluate_model(
    model: HybridForecaster,
    test_data: pd.DataFrame,
    config: Dict
) -> Dict[str, float]:
    """Evaluate model performance
    
    Args:
        model: Trained model
        test_data: Test dataset
        config: Configuration
        
    Returns:
        Dictionary of metrics
    """
    target_col = config['data']['target_column']
    exog_features = config['data']['exogenous_features']
    
    y_true = test_data[target_col].values
    X_test = test_data[exog_features]
    dates_test = test_data.index
    
    # Make predictions
    y_pred, individual_preds = model.predict(
        n_periods=len(y_true),
        X=X_test,
        dates=dates_test
    )
    
    # Calculate metrics
    metrics = {
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'sarimax_mape': mean_absolute_percentage_error(y_true, individual_preds['sarimax']),
        'prophet_mape': mean_absolute_percentage_error(y_true, individual_preds['prophet'])
    }
    
    return metrics


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train GMV forecasting models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated list of cities to train (default: all from config)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    logger.info("Loading configurations")
    config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Get cities to process
    if args.cities:
        cities = [c.strip() for c in args.cities.split(',')]
    else:
        cities = config['data']['cities']
    
    logger.info(f"Training models for cities: {cities}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    experiment_name = args.experiment_name or config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)
    
    # Load data
    logger.info("Loading data")
    data_loader = DataLoader(
        data_path=config['data']['raw_data_path'],
        date_column=config['data']['date_column'],
        target_column=config['data']['target_column'],
        city_column=config['data']['city_column']
    )
    
    data_loader.load_data()
    data_loader.validate_data()
    df = data_loader.preprocess_data()
    
    # Create train/test splits
    train_sets, test_sets = create_train_test_sets(
        df=df,
        cities=cities,
        split_date=config['data']['test_split_date'],
        start_date=config['data']['train_start_date'],
        city_column=config['data']['city_column']
    )
    
    # Train models for each city
    models_dir = Path(config['models']['output_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for city in cities:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {city}")
        logger.info(f"{'='*60}")
        
        with mlflow.start_run(run_name=f"hybrid_model_{city.replace(' ', '_').lower()}"):
            # Log parameters
            mlflow.log_params({
                'city': city,
                'model_type': 'hybrid_sarimax_prophet',
                'train_start': config['data']['train_start_date'],
                'test_split': config['data']['test_split_date']
            })
            
            # Train model
            model = train_city_model(
                city=city,
                train_data=train_sets[city],
                test_data=test_sets[city],
                config=config,
                model_config=model_config
            )
            
            # Log model parameters
            model_params = model.get_params()
            mlflow.log_params({
                'sarimax_weight': model_params['weights']['sarimax'],
                'prophet_weight': model_params['weights']['prophet']
            })
            
            # Evaluate model
            metrics = evaluate_model(model, test_sets[city], config)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            logger.info(f"Model Performance for {city}:")
            logger.info(f"  Hybrid MAPE: {metrics['mape']:.2%}")
            logger.info(f"  Hybrid MAE: {metrics['mae']:.2f}")
            logger.info(f"  Hybrid RMSE: {metrics['rmse']:.2f}")
            logger.info(f"  SARIMAX MAPE: {metrics['sarimax_mape']:.2%}")
            logger.info(f"  Prophet MAPE: {metrics['prophet_mape']:.2%}")
            
            # Save model
            city_safe = city.lower().replace(" ", "_")
            model_path = models_dir / f"hybrid_{city_safe}"
            model.save(str(model_path))
            
            # Log model to MLflow
            mlflow.log_artifacts(str(model_path))
            
            logger.info(f"Model saved to {model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

