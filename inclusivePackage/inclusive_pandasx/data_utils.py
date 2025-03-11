import numpy as np


def compute_residuals(y_actual, y_predicted):
    """
    Return residuals for model predictions.
    """
    return np.mean((y_acutal - y_predicted) ** 2)


def make_set_of_metrics(model_score, model_mse, model_predictions):
    return (model_score, model_mse, model_predictions)
