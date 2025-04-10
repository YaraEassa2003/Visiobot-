import pandas as pd
import numpy as np
from gpt_gateway import handle_chat


def determine_dimension(data, df=None, ask_gpt_for_hierarchy=False):
    """
    Decide if data is 1D, 2D, ND, or Hierarchical.
    If ask_gpt_for_hierarchy=True, we'll attempt a GPT classification
    using a structural summary of the CSV (df).
    """

    # If data is a NumPy array (typical path when CSV -> DataFrame -> .to_numpy()):
    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 1:
            dimension = "1D"
        elif data.ndim == 1:
            dimension = "1D"
        elif data.ndim == 2:
            # We treat up to 3 columns as '2D'
            dimension = "2D" if data.shape[1] <= 3 else "ND"
        else:
            dimension = "ND"

    # If data is a list (the old code path):
    elif isinstance(data, list):
        if all(isinstance(i, list) for i in data):
            dimension = "Hierarchical"
        else:
            dimension = "1D"
    else:
        dimension = "Unknown"

    # Optional step: Ask GPT if we want to see if the CSV is "Hierarchical."
    # This only makes sense if we have the DataFrame 'df'.
    if ask_gpt_for_hierarchy and df is not None and (dimension == "ND" or dimension == "2D"): 
        if is_one_to_many_hierarchy(df):       # Summarize CSV structure for GPT:
            print("[DEBUG] Invoking maybe_refine_to_hierarchical_with_gpt()...")
            dimension_before = dimension
            dimension = maybe_refine_to_hierarchical_with_gpt(dimension, df)
            print(f"[DEBUG] dimension after GPT call: was '{dimension_before}', now '{dimension}'")
        else:
        # It's not hierarchical if there's no one-to-many structure
            pass

    return dimension

def is_one_to_many_hierarchy(df):
    """
    Determines whether the DataFrame exhibits a one-to-many (hierarchical) structure.
    Returns True only if there are at least 3 columns and at least one of the potential
    parent columns contains duplicate values (indicating repetition).
    """
    # If there are fewer than 3 columns, we cannot really have a multi-level hierarchy.
    if len(df.columns) < 3:
        return False

    # Check each column except the last one. If any column has duplicate values,
    # that indicates a potential one-to-many relationship.
    for col in df.columns[:-1]:
        if df[col].duplicated().any():
            return True
    return False



def maybe_refine_to_hierarchical_with_gpt(current_dimension, df):
    """
    If current_dimension is '2D' or 'ND', we can ask GPT whether it might be hierarchical
    based on structural patterns in the DataFrame.
    Return the final dimension after GPT's response: either "Hierarchical" or original dimension.
    """

    # 1. Collect some structure info about the DataFrame
    columns = df.columns.tolist()
    # e.g. unique counts per column
    cardinalities = {col: df[col].nunique() for col in columns}
    # Sample first few rows (just to show GPT the shape)
    sample_rows = df.head(10).to_dict(orient="records")  # A small sample

    # 2. Build a prompt for GPT
    prompt = f"""
We have a CSV with {len(columns)} columns and {len(df)} rows (flattened table).
Columns: {columns}

Unique value counts:
{cardinalities}

Here is a sample of the first 10 rows (key-value pairs):
{sample_rows}

Only respond "Hierarchical" if there is a clear parent->child or multi-level structure 
(e.g., Region->Country->City). If you are uncertain or it does not look strictly hierarchical, 
respond "NotHierarchical". 
Return one word only: "Hierarchical" or "NotHierarchical".
"""

    try:
        gpt_response = handle_chat(prompt)
        # Normalize GPT's response
        response_lower = gpt_response.strip().lower()
        if "hierarchical" in response_lower:
            return "Hierarchical"
        else:
            return current_dimension
    except Exception as e:
        print(f"GPT error: {e}")
        # Fallback: keep original dimension
        return current_dimension

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

def extract_dataset_info(file_path, use_gpt_for_hierarchy=False):
    """Extracts dataset properties as required by the trained model."""
    df = pd.read_csv(file_path)

    # Now we pass both the np.array AND the df, so determine_dimension can do GPT logic
    data_dimensions = determine_dimension(
        df.to_numpy(),
        df=df,
        ask_gpt_for_hierarchy=use_gpt_for_hierarchy
    )

    primary_variable, primary_variable_type = detect_primary_variable(df)

    dataset_info = {
        "Data_Dimensions": data_dimensions,
        "No_of_Attributes": df.shape[1],
        "No_of_Records": df.shape[0],
        "Primary_Variable (Data Type)": primary_variable_type,
    }

    return dataset_info

