import numpy as np
from numpy.typing import NDArray

from data_utils import compute_residuals
from sklearn.linear_models import LinearRegression
from sklearn.tree import DecisionTreeRegressor


@pd.api.extensions.register_dataframe_accessor("analyzer")
class DataAnalyzer:
    def __init__(self, data_Object):
        self.data_Object = data_Object
  
    @staticmethod 
    def make_set_of_metrics(model_score, model_mse, model_predictions) -> tuple:
        "returns an set of model metrics; including predictions"
        return (model_score, model_mse, model_predictions)

    @staticmethod
    def compute_residuals(y_actual, y_predicted) -> NDArray:
        """
            Return residuals for model predictions.
        """
        return np.mean((y_acutal - y_predicted)**2)


    @staticmethod 
    def design_X(X) -> NDArray:
        """
            return an array design X for which is of the form 
            f(x0, x1)=a* x0^2 + b* x0 + c* x0*x1 + d* x1 + e*x1^2 + f
            
            for later: allow user specified design matrix
        """
        x0 = X[:, 0]  # static pressure
        x1 = X[:, 1]  # differential pressure 
        X_design = np.vstack((np.ones_like(x0), x0, x1, x0**2, x0*x1, x1**2)).T
        return  X_design


    def fit_data(self, X, y, **kwargs, random_state=42) -> dict:
        """"
        return a dictionary `fitDict` of multiple fits for data... including:
        polynomial fit, decision trees, regresssion 
        
        model_values: takes an order pair, model_score, model_mse, and predicted model values
         but poly 2D which takes the coeffs, model_mse, and model_predicted 
        """

        # init, fit and predict linear regression  model
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

        poly_values = make_set_of_metrics(coef, poly_mse, poly_y)
        
        # set dict values        
        return { 'y_actual': y , 'lm_model':lm_values, 'dTree_model':dTree_values, 'poly_model':poly_values} 


