"""Model loading and management"""

from pathlib import Path
from typing import Dict, List, Optional
from src.models.hybrid_model import HybridForecaster
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelManager:
    """Manage model loading and caching"""
    
    def __init__(self, config: Dict):
        """Initialize model manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models: Dict[str, HybridForecaster] = {}
        self.models_dir = Path(config['models']['output_dir'])
    
    def load_model(self, city: str) -> Optional[HybridForecaster]:
        """Load a model for a specific city
        
        Args:
            city: City name
            
        Returns:
            Loaded model or None if not found
        """
        city_safe = city.lower().replace(" ", "_")
        model_path = self.models_dir / f"hybrid_{city_safe}"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            logger.info(f"Loading model for {city}")
            model = HybridForecaster.load(str(model_path))
            self.models[city] = model
            logger.info(f"Model loaded successfully for {city}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model for {city}: {e}")
            return None
    
    def load_all_models(self):
        """Load all available models"""
        cities = self.config['data']['cities']
        
        logger.info(f"Loading models for {len(cities)} cities")
        
        for city in cities:
            self.load_model(city)
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def get_model(self, city: str) -> Optional[HybridForecaster]:
        """Get a loaded model
        
        Args:
            city: City name
            
        Returns:
            Model if loaded, otherwise tries to load it
        """
        if city in self.models:
            return self.models[city]
        
        # Try to load if not in cache
        return self.load_model(city)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names
        
        Returns:
            List of city names with loaded models
        """
        return list(self.models.keys())
    
    def unload_model(self, city: str):
        """Unload a model from memory
        
        Args:
            city: City name
        """
        if city in self.models:
            del self.models[city]
            logger.info(f"Unloaded model for {city}")
    
    def unload_all_models(self):
        """Unload all models"""
        self.models.clear()
        logger.info("All models unloaded")

