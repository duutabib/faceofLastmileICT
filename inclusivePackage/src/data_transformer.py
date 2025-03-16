import numpy as np

@pd.api.extensions.register_dataframe_accessor("transformer")
class DataTransformer:
    def __init__(self, data_Object,  **kwargs):
        self.data_Object=data_Object 

    
    def transform_variable(self, scaling_type='min-max', epsilon, axis=1, **kwargs):
        """
        return scaled versions of desired variables [typically scales to (0, 1)]
        """
        for col in self.data_Object:
            col_data = self.data_Object[col]

            match scaling_type:
                case 'min-max' | 'normalization':
                    max_value = col_data.max()
                    min_value = col_data.min()
                    col_transformed = (col_data - min_value)/ (max_value - min_value)
                
                case 'standardized':
                    mean = col_data.mean()
                    std = col_data.std() + epsilon 
                    col_transformed = (col_data - mean)/std
                
                case _:
                    max_value = col_data.max()
                    min_value = col_data.min()
                    col_transformed = (col_data - min_value)/ (max_value - min_value)

            self.data_Object[col] = col_transformed  
        return data_Object 


    def is_data_transformed(self, scaling_type, tolerance):
        """
        Check if the dataset is appropriately scaled.

        Parameters:
        - df: The DataFrame containing the data
        - scaling_type: 'standardized' or 'normalized' or 'min-max' to check the scaling type
        - tolerance: A tolerance value for mean, std, min, max to allow small deviations
    
        Returns:
        - bool, indicating whether the data is transformed 
        """
    
        for column in self.data.columns:
            column_data = self.data[column]
            
            match scaling_type:
                case 'standardized':
                    # Check mean and std for standardized data (mean ≈ 0, std ≈ 1)
                    mean = column_data.mean()
                    std_dev = column_data.std()
                    is_scaled = (abs(mean) < tolerance) and (abs(std_dev - 1) < tolerance)
                
                    # need to handle what happens if is_scaled is false 

                case 'normalized':
                    # Check min and max for normalized data (min ≈ 0, max ≈ 1)
                    min_val = column_data.min()
                    max_val = column_data.max()
                    is_scaled = (abs(min_val) < tolerance) and (abs(max_val - 1) < tolerance)
                
                    # need to handle what happens if is_scaled is false 
            
                case 'min-max':
                    # Check min and max for min-max normalized data (min ≈ 0 , max ≈ 1 )
                    min_val = column_data.min()
                    max_val = column_data.max()
                    is_scaled = (abs(min_val) < tolerance ) and (abs(max_val - 1) < tolerance )
                    # need to handle what happens if is_scaled is false 

                case _:
                    raise ValueError("Invalid scaling_type. Use 'standardized' or 'normalized'.")
    
        return is_scaled


        def convert_to_flow_lph(self, col):
            if not col:
                return self.data['Flow_lph']/60
            else:
                return self.data[col]/60 


        def apply_lpm_rota_normalization(self, output_lpm, Tn, KCorrection):
            data = self.data_Object
            return data.LPM_Rota *(data.Static_Pa + output_lpm)/ output_lpm*(Tn/(data.temp_mcu + KCorrection))


