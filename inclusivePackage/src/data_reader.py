import os
import pickle
import typing

import pandas as pd

from typing import path, Any, Optional

# I have two design choices to make, either I remove the Pandas decorator...
# which expects a pandas object 
# or keep it and abstract the reading of the data from the manager, so that the manager
# expects a pandas object and not a filename name
# so I could have a separate class that depends on that inherits from the pandas read data class
# since the ddataManager depends on pandas functionality it might make sense to do the former...
# otherwise, I could create my own removing the dependence on pandas, but this might be too costly
# remove read functionality from data manager class...


class DataReader:
    # I'm thinking of implementing usecols differently, what do you think? for example defining as a constant,
    # embedded in an assert statement
    # add handle invoking with missing filenames...
    # update: might be better to do the handling in pandas now...
    DEFAULT_USECOLS = [
        'Static Pressure', 
        'LPM Rota', 
        'temp_mcu', 
        'Static_Pa', 
        'SP_mV', 
        'DP_mV', 
        'Differential_Pa', 
        'Flow_lph', 
        'ideal flow ', 
        'cd',
    ]
        
    def read_csv(self, filename: Optional[str],  usecols: List[str], **kwargs,):
        """ Create a new Data Manager instance"""
        if not usecols:
            usecols = DEFAULT_USECOLS
        if not filename:
            raise ValueError("Cannot initialize DataManager, filename not set or specified.")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        return  pd.read_csv(self.filename, usecols=usecols, **kwargs)

       