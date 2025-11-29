"""Model evaluation and comparison module"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.hybrid_model import HybridForecaster
from src.data.data_loader import DataLoader, create_train_test_sets
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluate and compare forecasting models"""
    
    def __init__(self, config: Dict):
        """Initialize evaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = []
    
    def evaluate_model(
        self,
        model: HybridForecaster,
        test_data: pd.DataFrame,
        city: str
    ) -> Dict:
        """Evaluate a single model
        
        Args:
            model: Trained model
            test_data: Test dataset
            city: City name
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model for {city}")
        
        target_col = self.config['data']['target_column']
        exog_features = self.config['data']['exogenous_features']
        
        y_true = test_data[target_col].values
        X_test = test_data[exog_features]
        dates_test = test_data.index
        
        # Get predictions from hybrid and individual models
        y_pred_hybrid, individual_preds = model.predict(
            n_periods=len(y_true),
            X=X_test,
            dates=dates_test
        )
        
        # Calculate metrics for each model
        results = {
            'city': city,
            'test_samples': len(y_true)
        }
        
        for model_name, predictions in [
            ('Hybrid', y_pred_hybrid),
            ('SARIMAX', individual_preds['sarimax']),
            ('Prophet', individual_preds['prophet'])
        ]:
            metrics = self._calculate_metrics(y_true, predictions)
            for metric_name, value in metrics.items():
                results[f'{model_name}_{metric_name}'] = value
        
        # Store predictions
        results['y_true'] = y_true
        results['y_pred_hybrid'] = y_pred_hybrid
        results['y_pred_sarimax'] = individual_preds['sarimax']
        results['y_pred_prophet'] = individual_preds['prophet']
        results['dates'] = dates_test
        
        # Model weights
        params = model.get_params()
        results['sarimax_weight'] = params['weights']['sarimax']
        results['prophet_weight'] = params['weights']['prophet']
        
        self.results.append(results)
        
        logger.info(f"Evaluation completed for {city}")
        logger.info(f"  Hybrid MAPE: {results['Hybrid_mape']:.2%}")
        logger.info(f"  Hybrid MAE: {results['Hybrid_mae']:.2f}")
        logger.info(f"  Hybrid RMSE: {results['Hybrid_rmse']:.2f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': root_mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def create_comparison_plots(self, output_dir: str = "outputs/plots"):
        """Create visualization plots
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualization plots in {output_dir}")
        
        for result in self.results:
            city = result['city']
            city_safe = city.lower().replace(" ", "_")
            
            # 1. Time series plot
            self._plot_time_series(result, output_path / f"{city_safe}_forecast.png")
            
            # 2. Model comparison plot
            self._plot_model_comparison(result, output_path / f"{city_safe}_comparison.png")
        
        # 3. Overall metrics comparison
        self._plot_overall_comparison(output_path / "overall_comparison.png")
        
        logger.info("Visualization plots created successfully")
    
    def _plot_time_series(self, result: Dict, output_file: Path):
        """Plot time series forecast
        
        Args:
            result: Evaluation result
            output_file: Output file path
        """
        fig, ax = plt.subplots(figsize=(15, 6))
        
        dates = result['dates']
        ax.plot(dates, result['y_true'], label='Actual', color='green', linewidth=2)
        ax.plot(dates, result['y_pred_hybrid'], label='Hybrid', color='red', linewidth=2)
        ax.plot(dates, result['y_pred_sarimax'], label='SARIMAX', 
                color='blue', alpha=0.6, linestyle='--')
        ax.plot(dates, result['y_pred_prophet'], label='Prophet', 
                color='orange', alpha=0.6, linestyle='--')
        
        ax.set_title(f"GMV Forecast - {result['city']}", fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('GMV', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, result: Dict, output_file: Path):
        """Plot model metrics comparison
        
        Args:
            result: Evaluation result
            output_file: Output file path
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = ['SARIMAX', 'Prophet', 'Hybrid']
        metrics = ['mape', 'mae', 'rmse']
        metric_labels = ['MAPE (%)', 'MAE', 'RMSE']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [result[f'{model}_{metric}'] for model in models]
            if metric == 'mape':
                values = [v * 100 for v in values]
            
            ax = axes[idx]
            bars = ax.bar(models, values, color=['blue', 'orange', 'red'], alpha=0.7)
            ax.set_title(label, fontsize=12)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f"Model Comparison - {result['city']}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_comparison(self, output_file: Path):
        """Plot overall comparison across cities
        
        Args:
            output_file: Output file path
        """
        # Prepare data for plotting
        data = []
        for result in self.results:
            for model in ['SARIMAX', 'Prophet', 'Hybrid']:
                data.append({
                    'City': result['city'],
                    'Model': model,
                    'MAPE': result[f'{model}_mape'] * 100,
                    'MAE': result[f'{model}_mae'],
                    'RMSE': result[f'{model}_rmse']
                })
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(['MAPE', 'MAE', 'RMSE']):
            ax = axes[idx]
            
            # Create grouped bar plot
            cities = df['City'].unique()
            x = np.arange(len(cities))
            width = 0.25
            
            for i, model in enumerate(['SARIMAX', 'Prophet', 'Hybrid']):
                values = df[df['Model'] == model].groupby('City')[metric].mean().values
                offset = (i - 1) * width
                bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('City')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(cities, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Overall Model Performance Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = "outputs/evaluation_report.csv"):
        """Generate evaluation report
        
        Args:
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        report_data = []
        for result in self.results:
            for model in ['SARIMAX', 'Prophet', 'Hybrid']:
                report_data.append({
                    'City': result['city'],
                    'Model': model,
                    'MAPE': result[f'{model}_mape'],
                    'MAE': result[f'{model}_mae'],
                    'RMSE': result[f'{model}_rmse'],
                    'R2': result[f'{model}_r2'],
                    'Test_Samples': result['test_samples']
                })
        
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(output_file, index=False)
        
        logger.info(f"Evaluation report saved to {output_file}")
        
        return df_report


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate GMV forecasting models")
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
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for evaluation outputs"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration")
    config = load_config(args.config)
    
    # Load data
    logger.info("Loading data")
    data_loader = DataLoader(
        data_path=config['data']['raw_data_path'],
        date_column=config['data']['date_column'],
        target_column=config['data']['target_column'],
        city_column=config['data']['city_column']
    )
    
    data_loader.load_data()
    df = data_loader.preprocess_data()
    
    # Create train/test splits
    cities = config['data']['cities']
    train_sets, test_sets = create_train_test_sets(
        df=df,
        cities=cities,
        split_date=config['data']['test_split_date'],
        start_date=config['data']['train_start_date'],
        city_column=config['data']['city_column']
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Load and evaluate models
    models_dir = Path(args.models_dir or config['models']['output_dir'])
    
    for city in cities:
        logger.info(f"\nEvaluating model for {city}")
        
        city_safe = city.lower().replace(" ", "_")
        model_path = models_dir / f"hybrid_{city_safe}"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        # Load model
        model = HybridForecaster.load(str(model_path))
        
        # Evaluate
        evaluator.evaluate_model(model, test_sets[city], city)
    
    # Generate outputs
    output_dir = Path(args.output_dir)
    
    # Create plots
    evaluator.create_comparison_plots(str(output_dir / "plots"))
    
    # Generate report
    report = evaluator.generate_report(str(output_dir / "evaluation_report.csv"))
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    print(report.to_string(index=False))
    
    logger.info("\nEvaluation completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

