import pandas as pd
import numpy as np
from gpt_gateway import handle_chat


def determine_dimension(data, df=None, ask_gpt_for_hierarchy=False):
    """
    Decide if data is 1D, 2D, ND, or Hierarchical.
    If ask_gpt_for_hierarchy=True, we'll attempt a GPT classification
    using a structural summary of the CSV (df).
    """

    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 1:
            dimension = "1D"
        elif data.ndim == 1:
            dimension = "1D"
        elif data.ndim == 2:
            dimension = "2D" if data.shape[1] <= 3 else "ND"
        else:
            dimension = "ND"

    elif isinstance(data, list):
        if all(isinstance(i, list) for i in data):
            dimension = "Hierarchical"
        else:
            dimension = "1D"
    else:
        dimension = "Unknown"

    if ask_gpt_for_hierarchy and df is not None and (dimension == "ND" or dimension == "2D" and len(df.columns) >= 3): 
        if is_one_to_many_hierarchy(df):       # Summarize CSV structure for GPT:
            print("[DEBUG] Invoking maybe_refine_to_hierarchical_with_gpt()...")
            dimension_before = dimension
            dimension = maybe_refine_to_hierarchical_with_gpt(dimension, df)
            print(f"[DEBUG] dimension after GPT call: was '{dimension_before}', now '{dimension}'")
        else:
            pass

    return dimension

def is_one_to_many_hierarchy(df):
    """
    Determines whether the DataFrame exhibits a one-to-many (hierarchical) structure.
    Returns True only if there are at least 3 columns and at least one of the potential
    parent columns contains duplicate values (indicating repetition).
    """
    if len(df.columns) < 3:
        return False
    relevant_columns = [col for col in df.columns[:-1] if "unnamed" not in col.lower()]
    for col in relevant_columns:
        if df[col].duplicated().any():
            return True
    return False




def maybe_refine_to_hierarchical_with_gpt(current_dimension, df):
    """
    If current_dimension is '2D' or 'ND', we can ask GPT whether it might be hierarchical
    based on structural patterns in the DataFrame.
    Return the final dimension after GPT's response: either "Hierarchical" or original dimension.
    """

    columns = df.columns.tolist()
    cardinalities = {col: df[col].nunique() for col in columns}
    sample_rows = df.head(10).to_dict(orient="records")  # A small sample

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
        response_lower = gpt_response.strip().lower()
        if response_lower == "hierarchical":
            return "Hierarchical"
        else:
            return current_dimension
    except Exception as e:
        print(f"GPT error: {e}")
        return current_dimension
import numpy as np
import pandas as pd

def is_truly_ordinal(series, threshold=10, tolerance=1e-6):
    """
    Determines whether a numeric series is truly ordinal.
    
    Parameters:
      series (pd.Series): The series to check.
      threshold (int): Maximum number of unique values to consider ordinal.
      tolerance (float): Tolerance for comparing differences.
    
    Returns:
      bool: True if the series is considered ordinal, otherwise False.
    """
    unique_vals = np.sort(series.dropna().unique())
    
    if len(unique_vals) > threshold:
        return False
    
    if not np.all(np.abs(unique_vals - unique_vals.astype(int)) < tolerance):
        return False
    
    differences = np.diff(unique_vals)
    
    if len(differences) == 0:
        return True
    
    if np.all(np.abs(differences - differences[0]) < tolerance):
        return True
    
    return False

def detect_primary_variable(df):
    """
    Determines the primary variable type in the dataset by analyzing feature types.
    Categories: ordinal, categorical, continuous, geographical, time, or other.
    """
    feature_types = {}

    for col in df.columns:
        unique_values = df[col].nunique()
        dtype = df[col].dtype

        if pd.api.types.is_string_dtype(df[col]):
            try:
                parsed_dates = pd.to_datetime(df[col], errors="coerce")
                if parsed_dates.notna().sum() / len(parsed_dates) > 0.8:
                    feature_types[col] = "time"
                else:
                    if any(keyword in col.lower() for keyword in ["country", "state", "city", "region", "latitude", "longitude"]):
                        feature_types[col] = "geographical"
                    else:
                        feature_types[col] = "categorical"
            except Exception:
                feature_types[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if is_truly_ordinal(df[col]):
                feature_types[col] = "ordinal"
            else:
                feature_types[col] = "continuous"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types[col] = "time"
        else:
            feature_types[col] = "other"

    primary_variable = max(feature_types, key=lambda x: df[x].nunique(), default=None)
    primary_type = feature_types.get(primary_variable, "unknown")
    if primary_type == "time":
        primary_type = "continuous"
    return primary_variable, primary_type

def extract_dataset_info(file_path, use_gpt_for_hierarchy=False):
    """Extracts dataset properties as required by the trained model."""
    df = pd.read_csv(file_path)

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

