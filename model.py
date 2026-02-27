"""
HA-Dopa Hydrogel Machine Learning Model Training Module
========================================================
This module provides a comprehensive framework for training and 
evaluating machine learning models for hydrogel property prediction.

Author: Research Team
License: MIT
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""
    RANDOM_FOREST = 'rf'
    GRADIENT_BOOSTING = 'gbr'
    MLP = 'mlp'
    SVR = 'svr'
    XGBOOST = 'xgb'
    LIGHTGBM = 'lgbm'


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    model_type: ModelType
    target_column: str
    feature_columns: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    scoring_metric: str = 'r2'
    
    def __post_init__(self):
        if not self.hyperparameters:
            self.hyperparameters = self._get_default_hyperparameters()
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for the specified model type."""
        defaults = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 1.0
            },
            ModelType.MLP: {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 500
            },
            ModelType.SVR: {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1
            }
        }
        return defaults.get(self.model_type, {})


@dataclass
class TrainingResult:
    """Container for training results and metrics."""
    
    model_name: str
    target: str
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    cv_scores: Optional[List[float]] = None
    feature_importances: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            'model_name': self.model_name,
            'target': self.target,
            'train_r2': self.train_r2,
            'test_r2': self.test_r2,
            'train_rmse': self.train_rmse,
            'test_rmse': self.test_rmse,
            'cv_mean': np.mean(self.cv_scores) if self.cv_scores else None,
            'cv_std': np.std(self.cv_scores) if self.cv_scores else None,
            'training_time': self.training_time
        }


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._is_trained = False
        self._feature_importances = None
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build and return the underlying model object."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Fit the model to the training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        """Return feature importances if available."""
        return self._feature_importances
    
    def _validate_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input data dimensions and types."""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or pandas DataFrame")
        
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")


class RandomForestModel(BaseModel):
    """Random Forest regression model wrapper."""
    
    def _build_model(self) -> Any:
        """Build Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            random_state=self.config.random_state,
            **self.config.hyperparameters
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Fit the Random Forest model."""
        self._validate_data(X, y)
        self.model = self._build_model()
        self.model.fit(X, y)
        
        self._feature_importances = dict(zip(
            self.config.feature_columns,
            self.model.feature_importances_
        ))
        self._is_trained = True
        
        logger.info("RandomForestModel training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting regression model wrapper."""
    
    def _build_model(self) -> Any:
        """Build Gradient Boosting model."""
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            random_state=self.config.random_state,
            **self.config.hyperparameters
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingModel':
        """Fit the Gradient Boosting model."""
        self._validate_data(X, y)
        self.model = self._build_model()
        self.model.fit(X, y)
        
        self._feature_importances = dict(zip(
            self.config.feature_columns,
            self.model.feature_importances_
        ))
        self._is_trained = True
        
        logger.info("GradientBoostingModel training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)


class MLPModel(BaseModel):
    """Multi-Layer Perceptron regression model wrapper."""
    
    def _build_model(self) -> Any:
        """Build MLP model."""
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            random_state=self.config.random_state,
            **self.config.hyperparameters
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPModel':
        """Fit the MLP model."""
        self._validate_data(X, y)
        self.model = self._build_model()
        self.model.fit(X, y)
        self._is_trained = True
        
        logger.info("MLPModel training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)


class ModelFactory:
    """Factory class for creating model instances."""
    
    _model_registry: Dict[ModelType, type] = {
        ModelType.RANDOM_FOREST: RandomForestModel,
        ModelType.GRADIENT_BOOSTING: GradientBoostingModel,
        ModelType.MLP: MLPModel,
    }
    
    @classmethod
    def create(cls, config: ModelConfig) -> BaseModel:
        """Create a model instance based on the configuration."""
        model_class = cls._model_registry.get(config.model_type)
        
        if model_class is None:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        return model_class(config)
    
    @classmethod
    def register(cls, model_type: ModelType, model_class: type) -> None:
        """Register a new model type."""
        cls._model_registry[model_type] = model_class


class ModelEvaluator:
    """Evaluate model performance with various metrics."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ['r2', 'rmse', 'mae', 'mape']
        self._metric_functions = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, Callable]:
        """Initialize metric calculation functions."""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        return {
            'r2': r2_score,
            'rmse': lambda y, p: np.sqrt(mean_squared_error(y, p)),
            'mae': mean_absolute_error,
            'mape': lambda y, p: np.mean(np.abs((y - p) / y)) * 100
        }
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all specified metrics."""
        results = {}
        
        for metric_name in self.metrics:
            if metric_name in self._metric_functions:
                results[metric_name] = self._metric_functions[metric_name](y_true, y_pred)
        
        return results
    
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                       cv: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation and return fold-wise scores."""
        raise NotImplementedError("Cross-validation not yet implemented")


class HyperparameterOptimizer:
    """Optimize model hyperparameters using grid or random search."""
    
    def __init__(self, model_config: ModelConfig, search_space: Dict[str, List[Any]],
                 method: str = 'grid', n_iter: int = 50):
        self.model_config = model_config
        self.search_space = search_space
        self.method = method
        self.n_iter = n_iter
        self._best_params = None
        self._best_score = None
        self._search_results = []
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                 cv: int = 5) -> Dict[str, Any]:
        """Find optimal hyperparameters."""
        raise NotImplementedError("Hyperparameter optimization not yet implemented")
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Return the best hyperparameters found."""
        return self._best_params
    
    def get_search_results(self) -> List[Dict[str, Any]]:
        """Return all search results."""
        return self._search_results


class TrainingPipeline:
    """Orchestrate the complete model training workflow."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Optional[BaseModel] = None
        self.evaluator = ModelEvaluator()
        self._training_history: List[TrainingResult] = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> TrainingResult:
        """Train the model and evaluate performance."""
        import time
        
        start_time = time.time()
        
        # Create and train model
        self.model = ModelFactory.create(self.config)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Evaluate
        train_metrics = self.evaluator.evaluate(y_train, train_pred)
        test_metrics = self.evaluator.evaluate(y_test, test_pred)
        
        training_time = time.time() - start_time
        
        result = TrainingResult(
            model_name=self.config.model_type.value,
            target=self.config.target_column,
            train_r2=train_metrics['r2'],
            test_r2=test_metrics['r2'],
            train_rmse=train_metrics['rmse'],
            test_rmse=test_metrics['rmse'],
            feature_importances=self.model.get_feature_importances(),
            training_time=training_time
        )
        
        self._training_history.append(result)
        logger.info(f"Training completed: Test R² = {result.test_r2:.4f}")
        
        return result
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> TrainingResult:
        """Perform cross-validation training."""
        raise NotImplementedError("Cross-validation training not yet implemented")
    
    def get_training_history(self) -> List[TrainingResult]:
        """Return training history."""
        return self._training_history


class MultiTargetTrainer:
    """Train models for multiple target variables."""
    
    def __init__(self, targets: List[str], feature_columns: List[str],
                 model_type: ModelType = ModelType.GRADIENT_BOOSTING):
        self.targets = targets
        self.feature_columns = feature_columns
        self.model_type = model_type
        self._pipelines: Dict[str, TrainingPipeline] = {}
        self._results: Dict[str, TrainingResult] = {}
    
    def train_all(self, data: pd.DataFrame, test_size: float = 0.2,
                  random_state: int = 42) -> Dict[str, TrainingResult]:
        """Train models for all target variables."""
        from sklearn.model_selection import train_test_split
        
        X = data[self.feature_columns].values
        
        for target in self.targets:
            logger.info(f"Training model for target: {target}")
            
            y = data[target].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            config = ModelConfig(
                model_type=self.model_type,
                target_column=target,
                feature_columns=self.feature_columns,
                random_state=random_state
            )
            
            pipeline = TrainingPipeline(config)
            result = pipeline.train(X_train, y_train, X_test, y_test)
            
            self._pipelines[target] = pipeline
            self._results[target] = result
        
        return self._results
    
    def get_results_summary(self) -> pd.DataFrame:
        """Return a summary of all training results."""
        records = [result.to_dict() for result in self._results.values()]
        return pd.DataFrame(records)


if __name__ == '__main__':
    # Example usage
    config = ModelConfig(
        model_type=ModelType.GRADIENT_BOOSTING,
        target_column='AS',
        feature_columns=['HA_C', 'HA_MW', 'DS', 'CL_C', 'pH', 'SP_C', 'SP_Type'],
        hyperparameters={'n_estimators': 100, 'max_depth': 4}
    )
    
    pipeline = TrainingPipeline(config)
    logger.info("Training pipeline created successfully")
