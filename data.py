"""
HA-Dopa Hydrogel Dataset Preprocessing Module
==============================================
This module provides comprehensive data preprocessing utilities for 
the HA-Dopa hydrogel machine learning pipeline.

Author: Research Team
License: MIT
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration class for data preprocessing parameters."""
    
    input_path: str
    output_path: str
    target_columns: List[str] = field(default_factory=lambda: ['AS', 'EM'])
    feature_columns: List[str] = field(default_factory=lambda: [
        'HA_C', 'HA_MW', 'DS', 'CL_C', 'pH', 'SP_C', 'SP_Type'
    ])
    categorical_columns: List[str] = field(default_factory=lambda: ['SP_Type'])
    numerical_columns: List[str] = field(default_factory=list)
    random_state: int = 42
    test_size: float = 0.2
    
    def __post_init__(self):
        self.numerical_columns = [
            col for col in self.feature_columns 
            if col not in self.categorical_columns
        ]


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._is_fitted = False
        self._statistics = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BasePreprocessor':
        """Fit the preprocessor to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(data).transform(data)
    
    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data format and content."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        missing_cols = set(self.config.feature_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class MissingValueHandler(BasePreprocessor):
    """Handle missing values in the dataset."""
    
    def __init__(self, config: DataConfig, strategy: str = 'mean'):
        super().__init__(config)
        self.strategy = strategy
        self._fill_values = {}
    
    def fit(self, data: pd.DataFrame) -> 'MissingValueHandler':
        """Calculate fill values based on the strategy."""
        self._validate_input(data)
        
        for col in self.config.numerical_columns:
            if self.strategy == 'mean':
                self._fill_values[col] = data[col].mean()
            elif self.strategy == 'median':
                self._fill_values[col] = data[col].median()
            elif self.strategy == 'mode':
                self._fill_values[col] = data[col].mode().iloc[0]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self._is_fitted = True
        logger.info(f"MissingValueHandler fitted with strategy: {self.strategy}")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the data."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        data = data.copy()
        for col, fill_value in self._fill_values.items():
            data[col].fillna(fill_value, inplace=True)
        
        return data


class OutlierDetector(BasePreprocessor):
    """Detect and handle outliers in numerical features."""
    
    def __init__(self, config: DataConfig, method: str = 'iqr', threshold: float = 1.5):
        super().__init__(config)
        self.method = method
        self.threshold = threshold
        self._bounds = {}
    
    def fit(self, data: pd.DataFrame) -> 'OutlierDetector':
        """Calculate outlier bounds for each numerical column."""
        self._validate_input(data)
        
        for col in self.config.numerical_columns:
            self._bounds[col] = self._calculate_bounds(data[col])
        
        self._is_fitted = True
        logger.info(f"OutlierDetector fitted with method: {self.method}")
        return self
    
    def _calculate_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Calculate lower and upper bounds for outlier detection."""
        if self.method == 'iqr':
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
        elif self.method == 'zscore':
            mean, std = series.mean(), series.std()
            lower = mean - self.threshold * std
            upper = mean + self.threshold * std
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return lower, upper
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers to the calculated bounds."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        data = data.copy()
        for col, (lower, upper) in self._bounds.items():
            data[col] = data[col].clip(lower=lower, upper=upper)
        
        return data
    
    def get_outlier_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a boolean mask indicating outlier positions."""
        raise NotImplementedError("Method not yet implemented")


class FeatureScaler(BasePreprocessor):
    """Scale numerical features to a standard range."""
    
    def __init__(self, config: DataConfig, method: str = 'standard'):
        super().__init__(config)
        self.method = method
        self._scaling_params = {}
    
    def fit(self, data: pd.DataFrame) -> 'FeatureScaler':
        """Calculate scaling parameters."""
        self._validate_input(data)
        
        for col in self.config.numerical_columns:
            if self.method == 'standard':
                self._scaling_params[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std()
                }
            elif self.method == 'minmax':
                self._scaling_params[col] = {
                    'min': data[col].min(),
                    'max': data[col].max()
                }
            elif self.method == 'robust':
                self._scaling_params[col] = {
                    'median': data[col].median(),
                    'iqr': data[col].quantile(0.75) - data[col].quantile(0.25)
                }
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        self._is_fitted = True
        logger.info(f"FeatureScaler fitted with method: {self.method}")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformation."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        data = data.copy()
        for col, params in self._scaling_params.items():
            if self.method == 'standard':
                data[col] = (data[col] - params['mean']) / params['std']
            elif self.method == 'minmax':
                data[col] = (data[col] - params['min']) / (params['max'] - params['min'])
            elif self.method == 'robust':
                data[col] = (data[col] - params['median']) / params['iqr']
        
        return data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling transformation."""
        raise NotImplementedError("Method not yet implemented")


class CategoricalEncoder(BasePreprocessor):
    """Encode categorical variables."""
    
    def __init__(self, config: DataConfig, method: str = 'label'):
        super().__init__(config)
        self.method = method
        self._encodings = {}
    
    def fit(self, data: pd.DataFrame) -> 'CategoricalEncoder':
        """Learn categorical encodings from the data."""
        self._validate_input(data)
        
        for col in self.config.categorical_columns:
            unique_values = data[col].unique()
            if self.method == 'label':
                self._encodings[col] = {val: idx for idx, val in enumerate(sorted(unique_values))}
            elif self.method == 'onehot':
                self._encodings[col] = list(unique_values)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        self._is_fitted = True
        logger.info(f"CategoricalEncoder fitted with method: {self.method}")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        data = data.copy()
        
        if self.method == 'label':
            for col, encoding in self._encodings.items():
                data[col] = data[col].map(encoding)
        elif self.method == 'onehot':
            for col, categories in self._encodings.items():
                for cat in categories:
                    data[f"{col}_{cat}"] = (data[col] == cat).astype(int)
                data.drop(columns=[col], inplace=True)
        
        return data


class PreprocessingPipeline:
    """Orchestrate multiple preprocessing steps."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._steps: List[Tuple[str, BasePreprocessor]] = []
        self._is_fitted = False
    
    def add_step(self, name: str, preprocessor: BasePreprocessor) -> 'PreprocessingPipeline':
        """Add a preprocessing step to the pipeline."""
        self._steps.append((name, preprocessor))
        logger.info(f"Added preprocessing step: {name}")
        return self
    
    def fit(self, data: pd.DataFrame) -> 'PreprocessingPipeline':
        """Fit all preprocessing steps sequentially."""
        current_data = data.copy()
        
        for name, preprocessor in self._steps:
            logger.info(f"Fitting step: {name}")
            preprocessor.fit(current_data)
            current_data = preprocessor.transform(current_data)
        
        self._is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing transformations."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        current_data = data.copy()
        
        for name, preprocessor in self._steps:
            current_data = preprocessor.transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(data).transform(data)
    
    def get_step(self, name: str) -> Optional[BasePreprocessor]:
        """Retrieve a specific preprocessing step by name."""
        for step_name, preprocessor in self._steps:
            if step_name == name:
                return preprocessor
        return None


def create_default_pipeline(config: DataConfig) -> PreprocessingPipeline:
    """Create a default preprocessing pipeline with standard steps."""
    pipeline = PreprocessingPipeline(config)
    
    pipeline.add_step('missing_values', MissingValueHandler(config, strategy='mean'))
    pipeline.add_step('outliers', OutlierDetector(config, method='iqr', threshold=1.5))
    pipeline.add_step('encoding', CategoricalEncoder(config, method='label'))
    pipeline.add_step('scaling', FeatureScaler(config, method='standard'))
    
    return pipeline


if __name__ == '__main__':
    # Example usage
    config = DataConfig(
        input_path='data/raw/ha_dopa_dataset.csv',
        output_path='data/processed/ha_dopa_processed.csv'
    )
    
    pipeline = create_default_pipeline(config)
    logger.info("Default preprocessing pipeline created successfully")
