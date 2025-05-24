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
        if is_one_to_many_hierarchy(df):      
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
    Determines the primary variable (column with highest cardinality)
    and classifies its type as one of:
      ordinal, categorical, continuous, geographical, or time.
    """
    feature_types = {}

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series) or "date" in col.lower():
            feature_types[col] = "time"

        elif pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().any():
                feature_types[col] = "time"
            elif any(k in col.lower() for k in
                     ["country", "state", "city", "region", "latitude", "longitude"]):
                feature_types[col] = "geographical"
            else:
                feature_types[col] = "categorical"

        elif pd.api.types.is_numeric_dtype(series):
            if is_truly_ordinal(series):
                feature_types[col] = "ordinal"
            else:
                feature_types[col] = "continuous"

        else:
            feature_types[col] = "other"

    primary_var = max(feature_types, key=lambda c: df[c].nunique(), default=None)
    primary_type = feature_types.get(primary_var, "unknown")
    if primary_type == "time":
        primary_type = "continuous"

    return primary_var, primary_type


def extract_dataset_info(file_path, use_gpt_for_hierarchy=False):
    """Extracts dataset properties as required by the trained model."""
    cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    date_cols = [c for c in cols if "date" in c.lower()]
    if date_cols:
        df = pd.read_csv(file_path, parse_dates=date_cols)
    else:
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

