"""Module for testing the analyzer..."""
from src.data_analyzer import Analyzer


def test_fit_data(pandas_obj, x_cols= ['Static_Pa', 'Differential_Pa'], y_col='cd', models=None):
    """Test fit data, ensure the return type is right..."""
    assert  'True'== isinstance(Analyzer(pandas_obj).fit_data(x_cols, y_col),  dict)
