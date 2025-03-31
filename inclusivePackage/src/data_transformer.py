import numpy as np
import pandas as pd
from typing import Optional, Any, List

# consider making DataTransformer more general, so I can inherit Flow transformer.
#
@pd.api.extensions.register_dataframe_accessor("transformer")
class Transformer:
    def __init__(self, pandas_obj: pd.DataFrame,  **kwargs):
        self._data= pandas_obj

    def transform_variable(self, scaling_type: str = 'min-max', epsilon: float=10E-4, axis: int = 1, **kwargs) -> pd.DataFrame:
        """
        return scaled versions of desired variables [typically scales to (0, 1)]
        """
        df = self._data.copy()
        for col in df:
            df[col] = self._scale(df[col], scaling_type, epsilon)
        return df

    def _scale(self, col_data: pd.Series, scaling_type: str, epsilon:float = 1e-10) -> pd.Series:
        if scaling_type in ('min-max' | 'normalization'):
            max_value, min_value  = col_data.max(), col_data.min()
            return (col_data - min_value)/ (max_value - min_value + epsilon)
                
        elif scaling_type == 'standardized':
           mean, std = col_data.mean(), col_data.std() + epsilon 
           return  (col_data - mean)/std
        
        raise ValueError(f"Unknown scaling_type:{scaling_type}") 
    
    def is_data_transformed(self, scaling_type: str, tolerance: float) -> bool:
        """
        Check if the dataset is appropriately scaled.

        Parameters:
        - df: The DataFrame containing the data
        - scaling_type: 'standardized' or 'normalized' or 'min-max' to check the scaling type
        - tolerance: A tolerance value for mean, std, min, max to allow small deviations
    
        Returns:
        - bool, indicating whether the data is transformed 
        """
    
        for col in self._data.columns:
            col_data = self._data[col]
            
            if scaling_type == 'standardized':
                # Check mean and std for standardized data (mean ≈ 0, std ≈ 1)
                mean, std_dev = col_data.mean(), col_data.std()
                if not (abs(mean) < tolerance) and (abs(std_dev - 1) < tolerance):
                    return False

            elif scaling_type in ('normalized', 'min-max'):
                # Check min and max for normalized data (min ≈ 0, max ≈ 1)
                min_val, max_val = col_data.min(), col_data.max()
                if not (abs(min_val) < tolerance) and (abs(max_val - 1) < tolerance):
                    return False
            else: 
                raise ValueError("Invalid scaling_type. Use 'standardized' or 'normalized', 'min-max'.")
        return True


class FlowTransformer:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def convert_to_flow_lph(self, col: str = 'Flow_lph') -> pd.Series:
        """Convert flow from lph to lpm"""
        if col not in self._df:
            raise KeyError(f"Column {col} not found.")
        return self._df[col]* 0.0166

    def apply_lpm_rota_normalization(self, output_lpm: float, Tn: float, KCorrection: float) -> pd.DataFrame:
        """Normalize lpm rotation ..."""
        required_cols = ["LPM_Rota", 'Static_Pa', 'temp_mcu']
        if not all(col in self._df for col in required_cols):
            raise KeyError(f"Missing required columns {required_cols}")
        return self._df.LPM_Rota *(self._df.Static_Pa + output_lpm)/ output_lpm*(Tn/(self._df.temp_mcu + KCorrection))

