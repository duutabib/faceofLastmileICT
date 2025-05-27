import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable, Dict, Any, Union, Type, Optional
import importlib
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import get_scorer

# Import configuration
from .config import (
    DEFAULT_TERM_FUNCTIONS,
    DEFAULT_MODELS,
    DEFAULT_METRICS,
    DATA_VALIDATION
)

ModelType = Union[BaseEstimator, Any]

@pd.api.extensions.register_dataframe_accessor("analyzer")
class Analyzer:
    """Class representing data analyzer
    It takes a pandas DataFrame and fits configured models to the data.
    """
    # Use term functions from config
    _TERM_FUNCTIONS: Dict[str, Callable[[NDArray, NDArray], NDArray]] = DEFAULT_TERM_FUNCTIONS.copy()

    @classmethod
    def add_term_function(cls, term: str, func: Callable[[NDArray, NDArray], NDArray]):
        """Add custom term to TERM_FUNCTIONS"""
        cls._TERM_FUNCTIONS[term] = func

    @staticmethod
    def compute_term(term: str, x0: NDArray, x1: NDArray) -> NDArray:
        """
        Compute a term for the design matrix

        Args:
            term: String representing the term (e.g., 'x0^2')
            x0: First feature array
            x1: Second feature array

        Returns:
            Computed term as NumPy array.
        """
        if term not in Analyzer._TERM_FUNCTIONS:
            raise ValueError(f"Unknown term: {term}")
        return Analyzer._TERM_FUNCTIONS[term](x0, x1)

    def __init__(self, pandas_obj: pd.DataFrame):
        """Initialize the analyzer with a pandas DataFrame."""
        if not isinstance(pandas_obj, pd.DataFrame):
            raise ValueError("pandas_obj must be a pandas DataFrame")
        if pandas_obj.empty:
            raise ValueError("pandas_obj must not be empty")
        self._pandas_obj = pandas_obj
        self._models = self._initialize_models()

    def _initialize_models(self) -> Dict[str, BaseEstimator]:
        """Initialize models from configuration."""
        models = {}
        for name, cfg in DEFAULT_MODELS.items():
            module_name, class_name = cfg['class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            models[name] = model_class(**(cfg.get('params', {})))
        return models

    @dataclass
    class FitResult:
        """Container for model fitting results."""
        model_name: str
        score: float
        mse: float
        predictions: NDArray
        params: Optional[Dict[str, Any]] = None

    def _make_fit_result(self, model_name: str, score: float, mse: float, 
                        predictions: NDArray, params: Dict[str, Any] = None) -> 'Analyzer.FitResult':
        """Create a FitResult instance."""
        return self.FitResult(
            model_name=model_name,
            score=score,
            mse=mse,
            predictions=predictions,
            params=params
        )

    def _fit_model(self, model: BaseEstimator, X: NDArray, y: NDArray, 
                  model_name: str = 'model') -> FitResult:
        """
        Fit a model and compute metrics.
        
        Args:
            model: The model to fit
            X: Feature matrix
            y: Target values
            model_name: Name of the model for identification
            
        Returns:
            FitResult containing model metrics and predictions
        """
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = float(np.mean((y - y_pred) ** 2))
        score = model.score(X, y) if hasattr(model, 'score') else None
        
        params = None
        if hasattr(model, 'get_params'):
            params = model.get_params()
            
        return self._make_fit_result(
            model_name=model_name,
            score=score,
            mse=mse,
            predictions=y_pred,
            params=params
        )

    def analyze(self, feature_columns: list, target_column: str, 
               models: Optional[Dict[str, BaseEstimator]] = None) -> Dict[str, FitResult]:
        """
        Analyze data using configured models.
        
        Args:
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            models: Optional dictionary of models to use instead of defaults
            
        Returns:
            Dictionary of FitResult objects keyed by model name
        """
        # Data validation
        if len(feature_columns) != 2:
            raise ValueError("Exactly two feature columns must be specified")
            
        missing_cols = [col for col in feature_columns + [target_column] 
                      if col not in self._pandas_obj.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
        # Prepare data
        X = self._pandas_obj[feature_columns].values
        y = self._pandas_obj[target_column].values
        
        # Use provided models or defaults
        models_to_fit = models or self._models
        
        # Fit models and collect results
        results = {}
        for name, model in models_to_fit.items():
            results[name] = self._fit_model(model, X, y, model_name=name)
            
        return results
