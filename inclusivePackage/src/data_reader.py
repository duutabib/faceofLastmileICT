import os
from typing import Optional
import pandas as pd



class Reader:
    """Class representing the data reader...
        allows for user configured cols, defaults_cols when usecols is None.
    """
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

    def __init__(self, filename: Optional[str],  usecols: list[str], **kwargs):
        self.filename = filename
        self.usecols = usecols
        self.kwargs = kwargs
        
    def read_csv(self):
        """Reads CSV and returns DataFrame..."""
        if not self.usecols:
            self.usecols = Reader.default_cols
        if not self.filename:
            raise ValueError("Cannot initialize Reader, filename not set or specified.")
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist.")
        return  pd.read_csv(self.filename, usecols=self.usecols, **self.kwargs)

       