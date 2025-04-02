"""Module for testing transformer class"""
import pytest
from src.data_transformer import Transformer


@pytest.mark.parametrize(
    'scaling_type',
    [
        'min-max', 
        'standardized',
        'normalized',
    ],
)
def test_transform_variable(pandas_obj, scaling_type):
    """test for transformer class, ensures that the data we
        getting the right out put...

    """
    data = Transformer(pandas_obj)
    data = data.transform_variable(scaling_type)
    assert data.iloc[:, 0].max() <= 2
    assert data.iloc[:, 0].min() >= -2

    