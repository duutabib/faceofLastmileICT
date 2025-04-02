""" Module providing testing for Manager class, ensuring expected and consistent behaviour"""
from src.data_manager import Manager


def test_manager_get_col(pandas_obj):
    """test for data manager class, ensures that the data we
    getting the right out put...
    """
    assert Manager(pandas_obj).get_col(["Static Pressure", "ideal flow"]).shape == (
        246,
        2,
    )

def test_manager_get_row(pandas_obj):
    """test for get_row , ensures that the data we
    getting the right out put...
    """
    assert Manager(pandas_obj).get_row(1).shape == (1, 10)

def test_manager_count_nrows(pandas_obj):
    """test for get_row , ensures that the data we
    getting the right out put...
    """
    assert Manager(pandas_obj).count_nrows() == 246

def test_manager_deduplicate(pandas_obj):

    """test manager deduplicate, ensures that the data we
    getting the right out put...
    """
    assert Manager(pandas_obj).deduplicate().shape == (210, 10)


def test_manager_count_ncols(pandas_obj):
    """test for count_ncols, ensures that the data we
    getting the right out put...
    """
    assert Manager(pandas_obj).count_ncols() == 10
