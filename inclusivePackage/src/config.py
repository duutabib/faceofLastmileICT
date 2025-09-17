"""Configuration settings for the data analyzer."""
from typing import Dict, Callable
import numpy as np
from numpy.typing import NDArray


# Default term functions for the Analyzer class
DEFAULT_TERM_FUNCTIONS: Dict[str, Callable[[NDArray, NDArray], NDArray]] = {
    '1': lambda x0, x1: np.ones_like(x0),
    'x0': lambda x0, x1: x0,
    'x1': lambda x0, x1: x1,
    'x0^2': lambda x0, x1: np.square(x0),
    'x0*x1': lambda x0, x1: np.multiply(x0, x1),
    'x1^2': lambda x0, x1: np.square(x1),
    'x0^3': lambda x0, x1: np.power(x0, 3),
    'sin(x0)': lambda x0, x1: np.sin(x0),
}

# Default models configuration
DEFAULT_MODELS = {
    'linear_regression': {
        'class': 'sklearn.linear_model.LinearRegression',
        'params': {}
    },
    'decision_tree': {
        'class': 'sklearn.tree.DecisionTreeRegressor',
        'params': {
            'max_depth': 3,
            'random_state': 42
        }
    }
}

# Default model metrics to compute
DEFAULT_METRICS = [
    'r2_score',
    'mean_squared_error',
    'mean_absolute_error'
]

# Default settings for data validation
DATA_VALIDATION = {
    'required_columns': [],  # Add required column names here
    'min_rows': 1,
    'max_nan_ratio': 0.5
}

# LabFit Constants
LABFIT_CONSTANTS = {
    'A': 0.21334,
    'B': 0.14933,
    'C': 0.033616,
    'D': 0.023546,
    'output_lpm': 100000,  # atm pressure in Pa
    'Tsb': 30.66,
    'KCorrection': 273.15,  # Kelvin temp correction
    'Tn': 293.15,  # standard temperature in Kelvin (20°C)
    'A1': 0.0005366622509, # diameter 1
    'A2': 0.0002010619298, # diameter 2
    'r0': 1.184,
    'epsilon': 0.1
}