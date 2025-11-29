"""Data preparation pipeline script"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader, create_train_test_sets
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main data preparation pipeline"""
    parser = argparse.ArgumentParser(description="Prepare data for GMV forecasting")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration")
    config = load_config(args.config)
    
    # Initialize data loader
    data_loader = DataLoader(
        data_path=config['data']['raw_data_path'],
        date_column=config['data']['date_column'],
        target_column=config['data']['target_column'],
        city_column=config['data']['city_column']
    )
    
    # Load and validate data
    logger.info("Loading and validating data")
    data_loader.load_data()
    validation_results = data_loader.validate_data()
    
    if not all(validation_results.values()):
        logger.error("Data validation failed!")
        return 1
    
    # Preprocess data
    logger.info("Preprocessing data")
    df = data_loader.preprocess_data(fill_na=True)
    
    # Create train/test splits
    logger.info("Creating train/test splits")
    train_sets, test_sets = create_train_test_sets(
        df=df,
        cities=config['data']['cities'],
        split_date=config['data']['test_split_date'],
        start_date=config['data']['train_start_date'],
        city_column=config['data']['city_column']
    )
    
    # Save processed data
    output_dir = args.output_dir or config['data']['processed_data_path']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_path}")
    
    # Save full processed dataset
    df.to_csv(output_path / "processed_data.csv")
    
    # Save train/test sets for each city
    for city in config['data']['cities']:
        city_safe = city.lower().replace(" ", "_")
        train_sets[city].to_csv(output_path / f"train_{city_safe}.csv")
        test_sets[city].to_csv(output_path / f"test_{city_safe}.csv")
        logger.info(f"Saved train/test sets for {city}")
    
    logger.info("Data preparation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

