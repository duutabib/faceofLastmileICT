import os
import pickle
import typing

import pandas as pd

from typing import Union, Any, Optional, List



class DataWriter:
    def __init__(self, pandas_obj):
        """ Create a new Data Manager instance"""
        self._data  = pandas_obj
    
    def save_csv(self, filename: Union[str, os.PathLike]) -> None:
        """save data in csv format..."""
        with  open(filename, "wb") as canvas:
            self._data.to_csv(canvas)

    def save_excel(self, filename: Union[str, os.PathLike]) -> None:
        """save data in excel format"""
        with open(filename, "wb") as canvas:
            self._data.to_excel(canvas)

    def write_curr_best_data_to_file(self, filename: Union[str, os.PathLike] = "curr_best.pickle") -> None:
        """Write best version of data in monitoring utility"""
        with open(filename, "wb") as canvas:
            pickle.dump(self._data, canvas,  protocol=pickle.HIGHEST_PROTOCOL)
        