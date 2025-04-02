import pytest
from src.data_reader import Reader


@pytest.fixture
def pandas_obj():
    """test for reader class, ensures that the data we 
        getting the right out put...
        
    """
    filename = "/Users/duuta/inclusiveEnergy-/labFit/training_data_dc.csv"
    return Reader(filename, usecols=None).read_csv()
