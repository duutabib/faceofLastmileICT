import numpy as np
from functools import cache


def compute_residuals(y_actual, y_predicted):
    """
    Return residuals for model predictions.
    """
    return np.mean((y_acutal - y_predicted) ** 2)


def make_set_of_metrics(model_score, model_mse, model_predictions):
    return (model_score, model_mse, model_predictions)


def compute_data_stats(data):
    """
       return a dict of data stats, mean, std, max and the min 
    """
    return {"mean": data.mean(), "std": data.std(), "max": data.max(), "min": data.min()}


def check_data_quality(data_stats, pop_stats):

    def unpack_stats(data_stats):
        mu, std, max_, min_ = data_stats["mean"], data_stats["std"], data_stats["max"], data_stats["min"]
        return mu, std, max_, min_
        
    mu, std, max_, min_ = unpack_stats(data_stats)
    pop_mu, pop_std, pop_max, pop_min = unpack_stats(pop_stats)

    is_data_good = (mu - pop_mu0) < tolearance  and (std - pop_std) and (max_ - pop_max)  and (min_ - pop_min)    

    return is_data_good
    

def assert_feature_cols(cols):
    COLS = ['Static_Pa', "Differential_Pa", "cd"]
    assert COLS == cols, "Incorrect feature cols"
    return 0


@cache
def monitoring_data_stats(data, tolearance=0.00001):
    """
        data should be shape (n, 3)
        monitor data distribution and rasie alert 
        monitor the feature vars before normalization []
        cache the last state if distribution is good...
    """
    curr_data=data 
    best_data=None 
    cols = curr_data.columns 

    # assert the cols... 
    assert_feature_cols(cols)
    
    for col in cols:
        col_data = data[col] 
        data_stats = compute_data_stats(col_data)
        is_good_data  = check_data_quality(data_stats)
        if not is_good_data:
            raise "Alert: Check sensor, readings are off..."
            ## fallback previous good data version (for all cols...)
        else:
            if best_version is None:
    ## need to workout how to implement the fallback
    ## how to set the first best state, perhaps manually and update
    ## how write to cache the best version of these     

    return 0
