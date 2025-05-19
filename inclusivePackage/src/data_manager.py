import os
import pickle

from typing import List, Union, Optional
from pathlib import Path

import pandas as pd

@pd.api.extensions.register_dataframe_accessor("manager")
class Manager:
    """Class representing DataManager"""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        """Create a new Data Manager instance
        Args:
            pandas_obj (pd.DataFrame): pandas DataFrame
        
        Raises:
            ValueError: if pandas_obj is not a pandas DataFrame
            ValueError: if pandas_obj is empty
        
        Returns:
            None
        """
        if not isinstance(pandas_obj, pd.DataFrame):
            raise ValueError("pandas_obj must be a pandas DataFrame")
        if pandas_obj.empty:
            raise ValueError("pandas_obj must not be empty")
        self._data = pandas_obj

    def get_col(self, col: Union[str, List[str]]) -> Union[pd.Series, pd.DataFrame]:
        "Implement one or many cols of data..."
        if isinstance(col, str):
            col = [col]
        if not all(item in self._data.columns.values for item in col):
            raise KeyError(f"Columns {col} not in data columns.")
        return self._data.loc[:, col]

    def get_row(self, row: List[Union[str, int]]) -> Union[pd.Series, pd.DataFrame]:
        """Get one or more rows from the data.
        Args:
            row (List[Union[str, int]]): list of row indices
        Returns:
            pd.Series if single row, pd.DataFrame: selected rows
        Raises:
            KeyError: if row not in data rows
        """
        "Implement for one or many row..."
        if isinstance(row, (str, int)):
            row = [row]
        if not row:
            return pd.DataFrame(columns=self._data.columns)
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

    def deduplicate(self, subset: List[str] = None, keep: str = "first", inplace=False) -> Optional[pd.DataFrame]:
        "return data without duplicates..."
        result = self._data.drop_duplicates(subset=subset, keep=keep)
        if inplace:
            self._data = result
            return None
        return result

    def save_csv(self, filename: Union[str, Path], **kwargs) -> None:
        """save data in csv format..."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._data.to_csv(filename, **kwargs)
        except Exception as e:
            raise OSError(f"Failed to save data to {filename}: {e}")

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

    def get_stats(self) -> pd.DataFrame:
        """get stats of data"""
        return self._data.describe()
    
    def get_missing_data(self) -> pd.DataFrame:
        """Get count of missing data
        Returns:
            pd.DataFrame: count of missing data
        """
        return self._data.isnull().sum()

