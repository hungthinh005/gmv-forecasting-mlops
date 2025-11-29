"""Data loading and validation module"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load and validate GMV forecasting data"""
    
    def __init__(
        self,
        data_path: str,
        date_column: str = "week_date",
        target_column: str = "gmv",
        city_column: str = "city_name"
    ):
        """Initialize data loader
        
        Args:
            data_path: Path to data file
            date_column: Name of date column
            target_column: Name of target column
            city_column: Name of city column
        """
        self.data_path = Path(data_path)
        self.date_column = date_column
        self.target_column = target_column
        self.city_column = city_column
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file
        
        Returns:
            Loaded dataframe
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
        
        return self.data
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate loaded data
        
        Returns:
            Dictionary of validation results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        validations = {}
        
        # Check required columns
        required_cols = [self.date_column, self.target_column, self.city_column]
        for col in required_cols:
            validations[f"has_{col}"] = col in self.data.columns
            if not validations[f"has_{col}"]:
                logger.error(f"Missing required column: {col}")
        
        # Check for missing values in target
        validations["no_missing_target"] = not self.data[self.target_column].isna().any()
        if not validations["no_missing_target"]:
            logger.warning(f"Missing values found in {self.target_column}")
        
        # Check date format
        try:
            pd.to_datetime(self.data[self.date_column])
            validations["valid_dates"] = True
        except Exception as e:
            validations["valid_dates"] = False
            logger.error(f"Invalid date format: {e}")
        
        # Check for duplicates
        validations["no_duplicates"] = not self.data.duplicated(
            subset=[self.date_column, self.city_column]
        ).any()
        if not validations["no_duplicates"]:
            logger.warning("Duplicate records found")
        
        # Log validation summary
        passed = sum(validations.values())
        total = len(validations)
        logger.info(f"Validation: {passed}/{total} checks passed")
        
        return validations
    
    def preprocess_data(self, fill_na: bool = True) -> pd.DataFrame:
        """Preprocess loaded data
        
        Args:
            fill_na: Whether to fill missing values
            
        Returns:
            Preprocessed dataframe
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Preprocessing data")
        df = self.data.copy()
        
        # Convert date column to datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Fill missing values
        if fill_na:
            df = df.fillna(0)
            logger.info("Filled missing values with 0")
        
        # Sort by date and city
        df = df.sort_values([self.city_column, self.date_column])
        
        # Set index
        df = df.set_index(self.date_column)
        
        logger.info("Data preprocessing completed")
        return df
    
    def get_city_data(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """Extract data for a specific city
        
        Args:
            df: Input dataframe
            city: City name
            
        Returns:
            City-specific dataframe
        """
        city_df = df[df[self.city_column] == city].copy()
        logger.info(f"Extracted {len(city_df)} rows for {city}")
        return city_df


def train_test_split_time_series(
    df: pd.DataFrame,
    split_date: str,
    start_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data into train and test sets
    
    Args:
        df: Input dataframe with datetime index
        split_date: Date to split at
        start_date: Optional start date for training
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_date = pd.to_datetime(split_date)
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        train_df = df[(df.index >= start_date) & (df.index < split_date)]
    else:
        train_df = df[df.index < split_date]
    
    test_df = df[df.index >= split_date]
    
    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    
    return train_df, test_df


def create_train_test_sets(
    df: pd.DataFrame,
    cities: List[str],
    split_date: str,
    start_date: Optional[str] = None,
    city_column: str = "city_name"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Create train/test sets for multiple cities
    
    Args:
        df: Input dataframe
        cities: List of city names
        split_date: Date to split at
        start_date: Optional start date for training
        city_column: Name of city column
        
    Returns:
        Tuple of (train_sets, test_sets) dictionaries
    """
    train_sets = {}
    test_sets = {}
    
    for city in cities:
        city_df = df[df[city_column] == city].copy()
        train_df, test_df = train_test_split_time_series(
            city_df, split_date, start_date
        )
        train_sets[city] = train_df
        test_sets[city] = test_df
    
    logger.info(f"Created train/test sets for {len(cities)} cities")
    
    return train_sets, test_sets

