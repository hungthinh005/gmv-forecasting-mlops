"""Batch prediction script for generating forecasts"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.hybrid_model import HybridForecaster
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_predictions(
    model: HybridForecaster,
    features: pd.DataFrame,
    dates: pd.DatetimeIndex,
    city: str
) -> pd.DataFrame:
    """Generate predictions for a city
    
    Args:
        model: Trained model
        features: Exogenous features
        dates: Prediction dates
        city: City name
        
    Returns:
        DataFrame with predictions
    """
    predictions, individual_preds = model.predict(
        n_periods=len(dates),
        X=features,
        dates=dates
    )
    
    results = pd.DataFrame({
        'city': city,
        'date': dates,
        'prediction_hybrid': predictions,
        'prediction_sarimax': individual_preds['sarimax'],
        'prediction_prophet': individual_preds['prophet']
    })
    
    return results


def main():
    """Main batch prediction script"""
    parser = argparse.ArgumentParser(description="Batch prediction for GMV forecasting")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        required=True,
        help="CSV file with future exogenous features"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/predictions/batch_predictions.csv",
        help="Output file for predictions"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Forecast horizon (number of periods)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration")
    config = load_config(args.config)
    
    # Load feature data
    logger.info(f"Loading features from {args.features_file}")
    features_df = pd.read_csv(args.features_file)
    
    # Setup
    models_dir = Path(args.models_dir or config['models']['output_dir'])
    cities = config['data']['cities']
    exog_features = config['data']['exogenous_features']
    
    all_predictions = []
    
    # Generate predictions for each city
    for city in cities:
        logger.info(f"Generating predictions for {city}")
        
        # Load model
        city_safe = city.lower().replace(" ", "_")
        model_path = models_dir / f"hybrid_{city_safe}"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        model = HybridForecaster.load(str(model_path))
        
        # Get features for this city
        city_features = features_df[features_df['city'] == city][exog_features]
        
        if len(city_features) == 0:
            logger.warning(f"No features found for {city}")
            continue
        
        # Take last N rows based on horizon
        city_features = city_features.tail(args.horizon)
        
        # Generate dates
        dates = pd.date_range(
            start=pd.Timestamp.now(),
            periods=args.horizon,
            freq='W'
        )
        
        # Generate predictions
        predictions = generate_predictions(model, city_features, dates, city)
        all_predictions.append(predictions)
        
        logger.info(f"Generated {len(predictions)} predictions for {city}")
    
    # Combine all predictions
    if all_predictions:
        results_df = pd.concat(all_predictions, ignore_index=True)
        
        # Save to file
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("Batch Prediction Summary")
        print("="*60)
        print(results_df.groupby('city')['prediction_hybrid'].describe())
        print("="*60)
    else:
        logger.error("No predictions generated")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

