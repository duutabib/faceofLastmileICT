import os
import pickle
import typing

import pandas as pd

from typing import Union, Any, Optional, List

# I have two design choices to make, either I remove the Pandas decorator...
# which expects a pandas object 
# or keep it and abstract the reading of the data from the manager, so that the manager
# expects a pandas object and not a filename name
# so I could have a separate class that depends on that inherits from the pandas read data class
# since the ddataManager depends on pandas functionality it might make sense to do the former...
# otherwise, I could create my own removing the dependence on pandas, but this might be too costly
# remove read functionality from data manager class...


@pd.api.extensions.register_dataframe_accessor("manager")
class DataManager:
    # I'm thinking of implementing usecols differently, what do you think? for example defining as a constant,
    # embedded in an assert statement
    # add handle invoking with missing filenames...
    # update: might be better to do the handling in pandas now...
    def __init__(self, pandas_obj):
        """ Create a new Data Manager instance"""
        self._data  = pandas_obj
    
    def get_col(self, col: Union[str List[str]] ) -> Union[pd.Series, pd.DataFrame]:
        'Implement one or many cols of data...'
        if isinstance(col, str):
            col = [col]
        if not all(item in self._data.columns.values for item in col):
            raise KeyError(f"Columns {col} not in data columns.")
        return self._data.loc[:, col] 

    def get_row(self, row: Union[str, int]) -> Union[pd.Series, pd.DataFrame]:
        'Implement for one or many row...'
        if isinstance(row, [str, int]):
            row = [row]
        if not all(item in self.data.index for item in row):
            raise KeyError(f"Rows {row} not in data rows")
        return self._data.loc[row, :]
    
    def count_nrows(self) -> int:
        """
            returns the number of rows for data 
        """
        return len(self._data.index)

    def count_ncols(self) -> int:
        "returns the number of cols for data"
        return len(self._data.columns)       

    def deduplicate(self, inplace=False) -> pd.Series:
        "return data without duplicates..."
        if inplace:
            self._data.drop_duplicates(inplace=True)
            return None
        return self._data.drop_duplicates()

      