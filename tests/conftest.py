"""Pytest configuration and shared fixtures"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'data': {
            'date_column': 'week_date',
            'target_column': 'gmv',
            'city_column': 'city_name',
            'cities': ['City A', 'City B'],
            'exogenous_features': ['x1', 'x2', 'x3']
        },
        'models': {
            'output_dir': 'models/'
        }
    }


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing"""
    np.random.seed(42)
    n = 100
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='W')
    
    df = pd.DataFrame({
        'week_date': dates,
        'city_name': (['City A'] * 50) + (['City B'] * 50),
        'gmv': np.random.uniform(100000, 200000, n),
        'x1': np.random.uniform(10000, 20000, n),
        'x2': np.random.uniform(5000, 10000, n),
        'x3': np.random.uniform(2000, 5000, n)
    })
    
    return df

