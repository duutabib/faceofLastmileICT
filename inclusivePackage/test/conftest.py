import pytest 
from data_manager import DataManager



@pytest.fixtures
def test_data_init(filename, usecols, **kwargs):
    return DataManager(filename, usecols).data