import os
import pickle
import typing

import pandas as pd

from typing import path, Any, optional


@pd.api.extensions.register_dataframe_accessor("manager")
class DataManager(self):
    # I'm thinking of implementing usecols differently, what do you think? for example defining as a constant,
    # embedded in an assert statement
    # add handle invoking with missing filenames...
    def __init__(self, filename: typing.,  usecols=['Static Pressure', 'LPM Rota', 'temp_mcu', 'Static_Pa', 'SP_mV',
       'DP_mV', 'Differential_Pa', 'Flow_lph', 'ideal flow ', 'cd',], **kwargs,):
        self.filename = filename
        self.usecols = usecols

    def __call__(self) -> pd.DataFrame:
        return self.data = pd.read_csv(self.filename, usecols=self.usecols, **kwargs)

    def get_col(self, col: list ) -> pd.Series:
        'Implement one or many cols of data...'
        if not all(item in self.data.columns.values for item in col):
            raise KeyError(f"{col} not in data columns")
        return self.data.loc[:, col] 

    def get_row(self, row: typing.Union[str, int]) -> pd.Series:
        'Implement for one or many row...'
        if not all(item in self.data.columns.values for item in row):
            raise KeyError(f"{row} not in data rows")
        return self.data.loc[row,:]
    
    def count_nrows(self) -> int:
        """
            returns the number of rows for data 
        """
        return self.data.count()

    def count_ncols(self) -> int:
        "returns the number of cols for data"
        return len(self.data.columns)       

    def deduplicate(self, inplace=False) -> pd.Series:
        "return data without duplicates..."
        if inplace:
            self.data = self.data.drop_duplicates
        return self.data.drop_duplicates


    def save_csv(self, filename: typing.Union[str, os.PathLike]):
        "save data in csv format..."
        self.filename = filename
        self.data.to_csv(self.filename)

    def save_excel(self, filename: typing.Union[str, os.PathLike]):
        'save data in excel format'
        self.filename = filename
        self.data.to_excel(self.filename)


    def write_curr_best_data_to_file(self) -> None:
        pickle.dump(self.data, file="curr_best.pickle", pickle.HIGHEST_PROTOCOL)
        