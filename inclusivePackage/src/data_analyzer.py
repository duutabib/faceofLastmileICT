import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data_utils import compute_residuals
from sklearn.linear_models import LinearRegression
from sklearn.tree import DecisionTreeRegressor


@pd.api.extensions.register_dataframe_accessor("analyzer")
class DataAnalyzer:
    def __init__(self, data_Object: pd.DataFrame):
        self.data_Object = data_Object
  
    @staticmethod 
    def add_model_stats_to_set(model_score: float, model_mse: float, model_predictions: NDArray) -> tuple:
        "returns an set of model metrics; including predictions"
        return (model_score, model_mse, model_predictions)
    
    
    # consider rewriting model init...
    # to make for easier call of models and generalization of methods
    def _compute_model_stats(model:str, ) -> tuple:
        
        mse
        return (model.score, mse, predictions)
        ...

    def _compute_residuals(y_actual: NDArray[np.float64], y_predicted: NDArray[np.float64]):
        return (y_actual - y_predicted)

    @staticmethod
    def compute_mse(y_actual: NDArray[np.float64], y_predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        """
            Return mse, residuals  for model predictions.
        """
        return _compute_residuals(y_actual, y_predictions)**2


    @staticmethod 
    def design_X(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
            Args:
                X: shape (n, 2) or (number of instances, number of features) 
            return an array design X for which is of the form 
            f(x0, x1)=a* x0^2 + b* x0 + c* x0*x1 + d* x1 + e*x1^2 + f
            
            for later: allow user specified design matrix
        """
        x0 = X[:, 0]  # static pressure
        x1 = X[:, 1]  # differential pressure 
        X_design = np.vstack((np.ones_like(x0), x0, x1, x0**2, x0*x1, x1**2)).T
        return  X_design


    def fit_data(self, X: NDArray[np.float64], y: NDArray[np.float64], **kwargs, random_state: int = 42) -> dict:
        """s
            return a dictionary `fitDict` of multiple fits for data... including:
            polynomial fit, decision trees, regresssion 
            
            model_values: takes an order pair, model_score, model_mse, and predicted model values
            but poly 2D which takes the coeffs, model_mse, and model_predicted 

            Args:
            X: shape (n, 2) or (number of instances, number of features) 
            y: (n, ) instances of response variable
            **kwargs: keyword arguments
            random_state: ran

        """

        # init, fit and predict linear regression  model
        # define helper to compute model stats... 
        # call to and add metrics as
        lm_model = LinearRegression(random_state=42)
        
        lm_score = lm_model.score(X, y)
        lm_y  = lm_model.predict(X)
        lm_mse = compute_residuals(y - lm_y)

        lm_values = make_set_of_metrics(lm_score, lm_mse, lm_y)
        
        
        # init, fit and predict decision tree
        dTree_model = DecisionTreeRegressor(random_state=42)
        
        dTree_score = dTree_model.score(X, y)
        dTree_mse = compute_residuals(y - dTree_y)
        dTree_y  = dTree_model.predict(X)

        dTree_values = make_set_of_metrics(dTree_score, dTree_mse, dTree_y)
       
        
        # designX, fit and predict 2 order polynomial 
        X_design = design_X(X)
        
        coef, _, _, _= np.linalg.lstsq(X_design, y)
        poly_y = X_design @ coef
        poly_mse = compute_residuals(y - poly_y)

        poly_values = make_set_of_metrics(None, poly_mse, poly_y)
        
        # set dict values        
        return { 'y_actual': (None, None, y) , 'lm_model':lm_values, 'dTree_model':dTree_values, 'poly_model':poly_values} 


