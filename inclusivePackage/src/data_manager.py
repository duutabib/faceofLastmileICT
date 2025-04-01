import os
import pickle

from typing import List, Union

import pandas as pd

@pd.api.extensions.register_dataframe_accessor("manager")
class DataManager:
    """Class representing DataManager"""

    def __init__(self, pandas_obj):
        """Create a new Data Manager instance"""
        self._data = pandas_obj

    def get_col(self, col: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        "Implement one or many cols of data..."
        if isinstance(col, str):
            col = [col]
        if not all(item in self._data.columns.values for item in col):
            raise KeyError(f"Columns {col} not in data columns.")
        return self._data.loc[:, col]

    def get_row(self, row: Union[str, int]) -> Union[pd.Series, pd.DataFrame]:
        "Implement for one or many row..."
        if isinstance(row, Union[str, int]):
            row = [row]
        if not all(item in self._data.index for item in row):
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

    def save_csv(self, filename: Union[str, os.PathLike]) -> None:
        """save data in csv format..."""
        with open(filename, "wb") as canvas:
            self._data.to_csv(canvas)

    def save_excel(self, filename: Union[str, os.PathLike]) -> None:
        """save data in excel format"""
        with open(filename, "wb") as canvas:
            self._data.to_excel(canvas)

    def write_curr_best_data_to_file(
        self, filename: Union[str, os.PathLike] = "curr_best.pickle"
    ) -> None:
        """write best version of data in monitoring utility"""
        with open(filename, "wb") as canvas:
            pickle.dump(self._data, canvas, protocol=pickle.HIGHEST_PROTOCOL)

