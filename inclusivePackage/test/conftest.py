import pytest 
from inclusive_pandasx import DataManager



@pytest.fixtures
def test_data_init(filename, usecols, **kwargs):
    return DataManager(filename, usecols).data