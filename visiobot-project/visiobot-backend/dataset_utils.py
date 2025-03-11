import pandas as pd
import numpy as np

def determine_dimension(data):
    """Determines if the dataset is 1D, 2D, ND, or Hierarchical."""
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return "1D"
        elif data.ndim == 2:
            return "2D"
        else:
            return "ND"
    elif isinstance(data, list):
        if all(isinstance(i, list) for i in data):
            return "Hierarchical"
        else:
            return "1D"
    else:
        return "Unknown"

def detect_primary_variable(df):
    """
    Determines the primary variable type in the dataset by analyzing feature types.
    Categories: ordinal, categorical, continuous, geographical, or other.
    """
    feature_types = {}

    for col in df.columns:
        unique_values = df[col].nunique()
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_values < 10:  
                feature_types[col] = "ordinal"
            else:
                feature_types[col] = "continuous"

        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            if any(keyword in col.lower() for keyword in ["country", "state", "city", "region", "latitude", "longitude"]):
                feature_types[col] = "geographical"
            else:
                feature_types[col] = "categorical"

        else:
            feature_types[col] = "other"

    primary_variable = max(feature_types, key=lambda x: df[x].nunique(), default=None)
    return primary_variable, feature_types.get(primary_variable, "unknown")

def extract_dataset_info(file_path):
    """Extracts dataset properties as required by the trained model."""
    df = pd.read_csv(file_path)

    data_dimensions = determine_dimension(df.to_numpy())

    primary_variable, primary_variable_type = detect_primary_variable(df)

    dataset_info = {
        "Data_Dimensions": data_dimensions,
        "No_of_Attributes": df.shape[1],  
        "No_of_Records": df.shape[0],  
        "Primary_Variable (Data Type)": primary_variable_type,  
    }

    return dataset_info
