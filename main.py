"""
HA-Dopa Hydrogel Machine Learning Analysis - Main Entry Point
==============================================================
This module provides the main entry point for the complete machine 
learning analysis pipeline for HA-Dopa hydrogel property prediction.

Author: Research Team
License: MIT
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import project modules
from data_preprocessing import (
    DataConfig, 
    PreprocessingPipeline, 
    create_default_pipeline
)
from model_training import (
    ModelConfig, 
    ModelType, 
    TrainingPipeline, 
    MultiTargetTrainer,
    TrainingResult
)
from shap_analysis import (
    SHAPConfig, 
    ExplainerType, 
    SHAPAnalysisPipeline,
    MultiTargetSHAPAnalyzer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Master configuration for the complete analysis pipeline."""
    
    # Data paths
    input_data_path: str
    output_dir: str
    
    # Feature configuration
    feature_columns: list
    target_columns: list
    categorical_columns: list
    
    # Model configuration
    model_type: ModelType = ModelType.GRADIENT_BOOSTING
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # SHAP configuration
    run_shap_analysis: bool = True
    shap_explainer_type: ExplainerType = ExplainerType.TREE
    
    # Output configuration
    save_models: bool = True
    save_plots: bool = True
    save_results: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        raise NotImplementedError("YAML loading not yet implemented")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'input_data_path': self.input_data_path,
            'output_dir': self.output_dir,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'categorical_columns': self.categorical_columns,
            'model_type': self.model_type.value,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'cv_folds': self.cv_folds,
            'run_shap_analysis': self.run_shap_analysis
        }


class AnalysisPipeline:
    """Main analysis pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._data = None
        self._preprocessed_data = None
        self._models: Dict[str, Any] = {}
        self._training_results: Dict[str, TrainingResult] = {}
        self._shap_results: Dict[str, Any] = {}
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> 'AnalysisPipeline':
        """Load data from the specified path."""
        import pandas as pd
        
        logger.info(f"Loading data from {self.config.input_data_path}")
        self._data = pd.read_csv(self.config.input_data_path)
        logger.info(f"Loaded {len(self._data)} samples with {len(self._data.columns)} columns")
        
        return self
    
    def preprocess(self) -> 'AnalysisPipeline':
        """Run data preprocessing pipeline."""
        logger.info("Starting data preprocessing")
        
        data_config = DataConfig(
            input_path=self.config.input_data_path,
            output_path=os.path.join(self.config.output_dir, 'processed_data.csv'),
            feature_columns=self.config.feature_columns,
            target_columns=self.config.target_columns,
            categorical_columns=self.config.categorical_columns
        )
        
        pipeline = create_default_pipeline(data_config)
        self._preprocessed_data = pipeline.fit_transform(self._data)
        
        logger.info("Data preprocessing completed")
        return self
    
    def train_models(self) -> 'AnalysisPipeline':
        """Train models for all target variables."""
        logger.info("Starting model training")
        
        trainer = MultiTargetTrainer(
            targets=self.config.target_columns,
            feature_columns=self.config.feature_columns,
            model_type=self.config.model_type
        )
        
        self._training_results = trainer.train_all(
            data=self._preprocessed_data,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Store trained models
        for target in self.config.target_columns:
            self._models[target] = trainer._pipelines[target].model
        
        logger.info("Model training completed")
        return self
    
    def run_shap_analysis(self) -> 'AnalysisPipeline':
        """Run SHAP analysis for model interpretability."""
        if not self.config.run_shap_analysis:
            logger.info("SHAP analysis skipped (disabled in config)")
            return self
        
        logger.info("Starting SHAP analysis")
        
        analyzer = MultiTargetSHAPAnalyzer(
            targets=self.config.target_columns,
            feature_columns=self.config.feature_columns
        )
        
        X = self._preprocessed_data[self.config.feature_columns].values
        
        # Get underlying sklearn models
        models = {
            target: self._models[target].model 
            for target in self.config.target_columns
        }
        
        self._shap_results = analyzer.analyze_all(
            models=models,
            X_train=X,
            X_explain=X
        )
        
        logger.info("SHAP analysis completed")
        return self
    
    def save_results(self) -> 'AnalysisPipeline':
        """Save all results to the output directory."""
        import pandas as pd
        import json
        
        logger.info(f"Saving results to {self.config.output_dir}")
        
        # Save training results
        if self.config.save_results:
            results_df = pd.DataFrame([
                result.to_dict() for result in self._training_results.values()
            ])
            results_df.to_csv(
                os.path.join(self.config.output_dir, 'training_results.csv'),
                index=False
            )
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info("Results saved successfully")
        return self
    
    def generate_report(self) -> str:
        """Generate a summary report of the analysis."""
        raise NotImplementedError("Report generation not yet implemented")
    
    def run(self) -> 'AnalysisPipeline':
        """Execute the complete analysis pipeline."""
        logger.info("="*60)
        logger.info("Starting HA-Dopa Hydrogel ML Analysis Pipeline")
        logger.info("="*60)
        
        self.load_data()
        self.preprocess()
        self.train_models()
        self.run_shap_analysis()
        self.save_results()
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully")
        logger.info("="*60)
        
        return self


def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration."""
    return PipelineConfig(
        input_data_path='data/ha_dopa_dataset.csv',
        output_dir='results',
        feature_columns=['HA_C', 'HA_MW', 'DS', 'CL_C', 'pH', 'SP_C', 'SP_Type'],
        target_columns=['AS', 'EM'],
        categorical_columns=['SP_Type'],
        model_type=ModelType.GRADIENT_BOOSTING,
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        run_shap_analysis=True
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HA-Dopa Hydrogel Machine Learning Analysis Pipeline'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/ha_dopa_dataset.csv',
        help='Path to input data file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['rf', 'gbr', 'mlp'],
        default='gbr',
        help='Model type to use'
    )
    
    parser.add_argument(
        '--no-shap',
        action='store_true',
        help='Skip SHAP analysis'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    model_type_map = {
        'rf': ModelType.RANDOM_FOREST,
        'gbr': ModelType.GRADIENT_BOOSTING,
        'mlp': ModelType.MLP
    }
    
    config = PipelineConfig(
        input_data_path=args.input,
        output_dir=args.output,
        feature_columns=['HA_C', 'HA_MW', 'DS', 'CL_C', 'pH', 'SP_C', 'SP_Type'],
        target_columns=['AS', 'EM'],
        categorical_columns=['SP_Type'],
        model_type=model_type_map[args.model],
        random_state=args.seed,
        run_shap_analysis=not args.no_shap
    )
    
    # Run pipeline
    pipeline = AnalysisPipeline(config)
    pipeline.run()
    
    logger.info("Analysis completed successfully!")


if __name__ == '__main__':
    main()
