import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable, Dict

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


@pd.api.extensions.register_dataframe_accessor("analyzer")
class Analyzer:
    """Class representing data analyzer
        it takes an a pandas DataFrame, and fits three default models 
        to data, but can be configured to handle other models.
    """
    # Define term functions as a class-level dispatch table
    _TERM_FUNCTIONS: Dict[str, Callable[[NDArray, NDArray], NDArray]] = {
        '1': lambda x0, x1: np.ones_like(x0),
        'x0': lambda x0, x1: x0,
        'x1': lambda x0, x1: x1,
        'x0^2': lambda x0, x1: np.square(x0),
        'x0*x1':lambda x0, x1: np.multiply(x0, x1),
        'x1^2': lambda x0, x1: np.square(x1),
        # Add more terms as needed, e.g.:
        'x0^3': lambda x0, x1: np.power(x0, 3),
        'sin(x0)': lambda x0, x1: np.sin(x0),
    }


    @classmethod
    def add_term_function(cls, term:str, func:Callable[[NDArray, NDArray], NDArray]):
        """Add custom term to TERMS_FUNCTIONS"""
        cls._TERM_FUNCTIONS[term] = func

    @staticmethod
    def compute_term(term:str, x0:NDArray, x1:NDArray) -> NDArray:
        """
        Compute a term for the design matrix

        Args:
            term: String representing the term (e.g x0^2)
            x0: first feature
            x1: second feature
        

        Returns:
            compute term as Numpy Array.
        Raises:
            Value Error
    """
    if term not in Analyzer.TERMS_FUNCTIONS:
        raise ValueError(f'Unsupported term {term}. Supported terms {list(Analyzer.TERMS_FUNCTIONS.keys())}')
    result = Analyzer._TERM_FUNCTIONS[term](x0, x1)
    if result.shape !=x0.shape:
        raise ValueError(f"Term {term} produced shape {result.shape}, expected {x0.shape}")
    return np.asarry(result, dtype=x0.dtype)

    def __init__(self, pandas_obj: pd.DataFrame):
        self._pandas_obj = pandas_obj

    class FitResult:
        """Class representing the FitResult
            This is a wrapper function that collects the results 
            for model fitting.      
        """
        def __init__(self, score: float, mse: float, predictions: NDArray):
            self.score = score
            self.mse = mse
            self.predictions = predictions

    def _make_fit_result(self, score: float, mse: float, predictions: NDArray) -> tuple:
        "returns an set of model metrics; including predictions"
        return self.FitResult(score, mse, predictions)

    def _fit_model(self, model, X: NDArray, y: NDArray) -> "FitResult":
        """Compute model score, mean squure error and predictions

        Args:
            model_obj: model object, initialized model object
            X (NDArray): (n, 2) n instances of 2 features from
            y (NDArray): (n,) instances of dependent variable

        Returns:
            an ordered tuple, score, mse, y_pred
        """
        model.fit(X, y)
        score = model.score(X, y)
        y_pred = model.predict(X)
        mse = self.compute_mse(y, y_pred)
        return self.FitResult(score, mse, y_pred)

    @staticmethod
    def compute_mse(
        y_actual: NDArray[np.float64], y_predicted: NDArray[np.float64]
    ) -> float:
        """Compute mean squared error between actual and predicted values.
        Return mse, residuals  for model predictions.
        """
        residuals = y_actual - y_predicted
        return np.mean(residuals**2)

    @staticmethod
    def x_design(X: NDArray, terms: list[str]) -> NDArray:
        """Creates design matrix X based on terms.
        Args:
            X (NDArray[n, 2]): shape (n, 2) or (number of instances, number of features)
            terms List[str] : form of the function. Default is f(x0, x1)=a* x0^2 + b* x0 + c* x0*x1 + d* x1 + e*x1^2 + f
        Returns:
            an array design X for which is of the form
        """
        if X.shape[1] != 2:
            raise ValueError("X must have exactly 2 features")
        x0, x1= X[:, 0], X[:, 1]  # static, differential pressure
        default_terms = ["1", "x0", "x1", "x0^2", "x0*x1", "x1^2"]
        terms = terms or default_terms
        design = [Analyzer.compute_term(t, x0, x1) for t in terms]
        return np.vstack(design).T

    def fit_data(
        self,
        x_cols: list[str],
        y_col: str,
        models: dict[str] = None,
        random_state: int = 42,
    ) -> dict:
        """Fit multiple models to data.
        Args:
            x_cols (list[str]) : list of cols of x.
            y_col: (n, ) instances of response variable
            models: dict of models
            random_state (int): integer to set the random state of model for reproducibility.d

        Returns:
            return a dictionary `fitDict` of multiple fits for data... including:
            polynomial fit, decision trees, regresssion

        """
        default_models = {
            "lm_model": LinearRegression(),
            "dTree_model": DecisionTreeRegressor(random_state=random_state),
        }

        X = self._pandas_obj[x_cols].to_numpy()
        y = self._pandas_obj[y_col].to_numpy()

        # init, fit and predict linear regression  model
        # define helper to compute model stats...
        # call to and add metrics as

        models = models or default_models
        fit_dict = {"y_actual": self._make_fit_result(None, None, y)}
        for name, model in models.items():
            fit_dict[name] = self._fit_model(model, X, y)
        if "poly_model" not in models:
            x_design = self.x_design(X, terms=None)
            coef, *_ = np.linalg.lstsq(x_design, y)
            y_pred = x_design @ coef
            poly_mse = self.compute_mse(y, y_pred)
            fit_dict["poly_model"] = self._make_fit_result(None, poly_mse, y_pred)

        return fit_dict
