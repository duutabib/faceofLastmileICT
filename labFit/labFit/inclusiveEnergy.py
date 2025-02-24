# class to get data, transform and export labfit data

import csv 
import pandas as pd
from pathlib import Path
from sklearn.linear_models import  LinearRegression
from sklearn.tree import DecisionTreeRegressor



# Extending pandas for inclusive energy in house model 
# data model for analysis; allows quick data perparation for down stream analysis
# Perhaps implement analysis as part of the sys 
# 


@pd.api.extensions.register_dataframe_accessor("inclusiveEnergy")
class inclusiveEnergyDataObject(object):
    def __init__(self, filename, output_lpm, Tn, kCorrection, **kwargs,):
        self.filename = filename
        self.output_lpm = output_lpm
        self.Tn = Tn
        self.kCorrection = kCorrection
        self.data = pd.read_csv(self.filename, usecols=['Static Pressure', 'LPM Rota', 'temp_mcu', 'Static_Pa', 'SP_mV',
       'DP_mV', 'Differential_Pa', 'Flow_lph', 'ideal flow ', 'cd',], **kwargs)


    
    Scaling = staticmethod(lambda x: (x - min(x)) / (max(x) - min(x))) 
    LPM_rota_normalization = staticmethod(lambda row: row.LPM_Rota *(row.Static_Pa + self.output_lpm)/ output_lpm*(Tn/(row.temp_mcu + KCorrection)))
    SB_lpm = staticmethod(lambda row: row.Flow_lph/60)


    def __get__(self, col):
        """returns a cols from the dataObject 
           cols should be a list, althouh could pass one
           element list   
        """
        if not all(item in self.data.columns.values for item in col):
            raise KeyError(f"{col} not in data columns...")
        return self.data[col]     
    
   
    def nrows(self):
        """
            returns the number of rows for data 
        """
        return self.data.count()


    def ncols(self):
        """
            returns the number of columns for data
        """
        return len(self.data.columns)


    def remove_duplicates(self, inplace=True, **kwargs):
        """
        return data without duplicates 
        """
        return self.data.drop_duplicates(inplace=inplace, **kwargs)
        
    
    def transform_desired_variables(self, func, axis=1, **kwargs):
        """
        return scaled versions of desired variables [typically scales to (0, 1)]
        """
        return self.data.apply(func, axis=1, **kwargs)
    
    
    def export_labfit_data(self, mylist, outputFilename, **kwargs):
        '''get transformed data specific variables for labfit
        mylist: [static pressure, differential pressure, discharge coefficient(cd (yes it's confusing)]
        mylist: [var1, var2, var3,...]
        var1: one of elements in 
        var2: one of elements in list, different from var1
        var3: one of elements in list, different from var1, var2
        '''
        return self.__get__(mylist).to_csv(outputFilename, **kwargs)

    
    def fit_data(self, **kwargs):
        """"
        return multiple fits for data... including:
        polynomial fit, decision trees, regresssion 
        """
        pass 
    
    
    def plot_data(self):
        pass 

    def plot_fit(self):
        pass 

    
    
