import pytest
from src.data_reader import Reader

 
def test_data_reader():
    """test for reader class, ensures that the data we 
        getting the right out put...
    """
    filename = "/Users/duuta/inclusiveEnergy-/labFit/training_data_dc.csv"
    default_cols = [
            'Static Pressure', 
            'LPM Rota', 
            'temp_mcu', 
            'Static_Pa', 
            'SP_mV', 
            'DP_mV', 
            'Differential_Pa', 
            'Flow_lph',
            'ideal flow', 
            'cd',
    ]
    assert set(default_cols) == set(Reader(filename, usecols=None).read_csv().columns.values)
    assert 246 == Reader(filename, usecols=None).read_csv().shape[0]


def test_read_csv_file_not_found():
    """Test read_csv for file not found..."""
    filename = "/Users/duuta/inclusiveEnergy-/LabFit0/training_data_dc.csv"
    reader = Reader(filename, usecols=None)
    with pytest.raises(FileNotFoundError) as exception_info:
        reader.read_csv()
    assert exception_info.type is FileNotFoundError
    assert exception_info.value.args[0] == f"File {filename} does not exist."
    


def test_reader_read_csv_no_filename():
    """test for reader class, ensures that the data we 
        getting the right out put...
    """
    filename = None
    reader = Reader(filename, usecols=None)
    with pytest.raises(ValueError) as exception_info:
        reader.read_csv()
    assert exception_info.type is ValueError
    assert exception_info.value.args[0] == "Cannot initialize Reader, filename not set or specified."
