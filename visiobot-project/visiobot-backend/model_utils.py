import os
import numpy as np
from flask import request
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from openai import OpenAI
import joblib
import geopandas as gpd
import networkx as nx
import squarify
import gpt_gateway



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SHAPEFILE_PATH = os.path.join(DATA_DIR, "ne_110m_admin_0_countries.shp")
print("Loading shapefile from:", SHAPEFILE_PATH)
world = gpd.read_file(SHAPEFILE_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, "generated_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "visiobot_model.keras")
PIPELINE_PATH = os.path.join(BASE_DIR, "saved_models", "preprocessing_pipeline.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

preprocessing_pipeline = joblib.load(PIPELINE_PATH)
scaler = preprocessing_pipeline["scaler"]
feature_columns = preprocessing_pipeline["feature_columns"]
one_hot_columns = preprocessing_pipeline["one_hot_columns"]

chart_type_mapping = {
    1: "Histogram",
    2: "Line Chart",
    3: "Linked Graph",
    4: "Map",
    5: "Parallel Coordinates",
    6: "Pie Chart",
    7: "Scatter Plot",
    8: "Treemap"
}


def preprocess_input(user_input):
    """Prepares user input using the saved preprocessing pipeline."""

    standardized_values = scaler.transform(
        pd.DataFrame([[user_input["No_of_Attributes"], user_input["No_of_Records"]]],
                    columns=["No_of_Attributes", "No_of_Records"])
    )
    print("Training Mean & Std Dev:", scaler.mean_, scaler.scale_)

    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0  # Initialize all values to zero

    input_df["No_of_Attributes"] = standardized_values[0, 0]
    input_df["No_of_Records"] = standardized_values[0, 1]

    categorical_features = ["Data_Dimensions", "Primary_Variable (Data Type)", "Task (Purpose)", "Target Audience"]
    for feature in categorical_features:
        col_name = f"{feature}_{user_input[feature]}"
        if col_name in one_hot_columns:  # Ensure only valid columns are set
            input_df[col_name] = 1

    input_df = input_df.fillna(0)

    expected_features = len(feature_columns)  # Use total features saved in pipeline
    final_input = input_df[feature_columns].to_numpy()[:, :expected_features]

    print("🔍 Checking Preprocessed Input Before Model Prediction:")
    print("Expected Features in Training:", feature_columns)
    print("Actual Features in Inference:", input_df.columns.tolist())
    print("Processed Input Data:\n", input_df)

    return final_input

def get_prediction(user_input):
    """Predicts the best visualization type and returns a ranked list of recommendations with plots."""
    input_data = preprocess_input(user_input)

    print("🔎 Final input shape being sent to model:", input_data.shape)
    expected_features = model.input_shape[1]

    if input_data.shape[1] != expected_features:
        return "❌ Feature mismatch! Expected {}, but got {}.".format(expected_features, input_data.shape[1]), None

    prediction = model.predict(input_data)[0]  # Get probability scores

    ranked_indices = np.argsort(prediction)[::-1]  # Sort in descending order
    ranked_visualizations = [(chart_type_mapping[i + 1], prediction[i]) for i in ranked_indices]

    print(f"🏆 Ranked Predictions: {ranked_visualizations}")  # Debugging

    visualization_plot = generate_visualization(ranked_indices[0] + 1)

    return ranked_visualizations, visualization_plot  # Return ranked list with plot



def generate_visualization(chart_type):
    """
    Generates a sample or placeholder visualization based on the predicted chart type.
    The chart_type argument is an integer (1–8) that maps to the names defined in chart_type_mapping:
      1 -> "Histogram"
      2 -> "Line Chart"
      3 -> "Linked Graph"
      4 -> "Map"
      5 -> "Parallel Coordinates"
      6 -> "Pie Chart"
      7 -> "Scatter Plot"
      8 -> "Treemap"
    """
    plt.figure(figsize=(6, 4))

    if chart_type == 1:
        # Histogram
        data = np.random.randn(100)
        plt.hist(data, bins=10, color="blue", alpha=0.7)
        plt.title("Histogram (Placeholder)")

    elif chart_type == 2:
        x = np.arange(50)
        y = np.random.randn(50).cumsum()  # cumsum to simulate a trending line
        plt.plot(x, y, marker="o", linestyle="-", color="green")
        plt.title("Line Chart (Placeholder)")

    elif chart_type == 3:
        try:
            
            G = nx.erdos_renyi_graph(n=8, p=0.3, seed=42)
            
            pos = nx.spring_layout(G, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
            nx.draw_networkx_edges(G, pos, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_color='black')
            
            plt.title("Linked Graph (Network Diagram Placeholder)")
            plt.axis('off')
        except ImportError:
            plt.text(0.5, 0.5, "Network Diagram not implemented (requires networkx)", ha="center")
        except Exception as e:
            plt.text(0.5, 0.5, f"Network Diagram rendering failed: {e}", ha="center")

    elif chart_type == 4:
        try:
            
            shapefile_path = os.path.join(DATA_DIR, "ne_110m_admin_0_countries.shp")
            world = gpd.read_file(shapefile_path)
            world.plot(column='pop_est', cmap='OrRd', legend=True, edgecolor='black')
            plt.title("World Map (Colored by Population Estimate)")
            plt.axis('off')
            
        except Exception as e:
            plt.text(0.5, 0.5, f"Map rendering failed: {e}", ha="center")

    elif chart_type == 5:
        try:
            from pandas.plotting import parallel_coordinates
            data = pd.DataFrame({
                "Feature1": np.random.randn(10),
                "Feature2": np.random.randn(10),
                "Feature3": np.random.randn(10),
                "Class": np.random.choice(["A", "B"], size=10)
            })
            parallel_coordinates(data, class_column="Class", color=["blue", "orange"])
            plt.title("Parallel Coordinates (Placeholder)")
        except Exception as e:
            plt.text(0.5, 0.5, "Parallel Coordinates not implemented", ha="center")

    elif chart_type == 6:
        data = np.random.randint(1, 10, 4)
        labels = [f"Cat {i+1}" for i in range(4)]
        plt.pie(data, labels=labels, autopct="%1.1f%%")
        plt.title("Pie Chart (Placeholder)")

    elif chart_type == 7:
        x = np.random.randn(50)
        y = np.random.randn(50)
        plt.scatter(x, y, color="purple", alpha=0.5)
        plt.title("Scatter Plot (Placeholder)")

    elif chart_type == 8:
        try:
            import squarify
            values = np.random.randint(1, 30, 5)
            labels = [f"Segment {i+1}" for i in range(len(values))]
            squarify.plot(sizes=values, label=labels, alpha=0.8)
            plt.axis("off")
            plt.title("Treemap (Placeholder)")
        except ImportError:
            plt.text(0.5, 0.5, "Treemap not implemented (requires squarify)", ha="center")
    else:
        plt.text(0.5, 0.5, "Unknown Chart Type", fontsize=12, ha="center")

    plot_path = os.path.join(PLOT_DIR, "generated_visualization.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def get_explanation(prediction, user_input):
    """
    Generates a simple, human-friendly explanation using GPT-4o mini
    via GitHub AI Inference, ensuring that <strong> tags are used
    for bold text rather than **.
    """
    typical_audience = {
        "Histogram": "non-expert",
        "Line Chart": "non-expert",
        "Pie Chart": "non-expert",
        "Scatter Plot": "non-expert",
        "Linked Graph": "expert",
        "Map": "expert",
        "Parallel Coordinates": "expert",
        "Treemap": "expert"
    }

    raw_audience = user_input.get("Target Audience", "non-expert")
    if isinstance(raw_audience, int):
        user_audience = "Expert" if raw_audience == 1 else "Non-Expert"
    else:
        
        norm = raw_audience.lower().replace("-", "").replace(" ", "")
        if norm == "expert":
            user_audience = "Expert"
        else:
            user_audience = "Non-Expert"

    recommended_aud = typical_audience.get(prediction, "non-expert")
    if recommended_aud.lower() == "expert":
        recommended_audience = "Expert"
    else:
        recommended_audience = "Non-Expert"

    note_html = ""
    if recommended_audience != user_audience:
        note_html = (
            "<strong>Note:</strong> "
            f"Although you requested a visualization for {user_audience} users, "
            f"based on my analysis, {prediction} appears to best suit your dataset's characteristics."
        )

    prompt = f"""
Explain why a <strong>{prediction}</strong> is the best choice based on the dataset details below.

Keep it <strong>short, simple, and natural</strong>. Use clear, easy-to-understand language. 
<strong>List all six inputs first</strong>, then give a <strong>brief, direct reason</strong> 
why this chart is the best choice. Avoid technical terms like "intuitive" or "effectively."
Use <strong> tags for bold text, and do not use ** for bolding.

    <strong>Dataset Details:</strong>
    - <strong>Data Dimension:</strong> {user_input['Data_Dimensions']}
    - <strong>Number of Attributes:</strong> {user_input['No_of_Attributes']}
    - <strong>Number of Records:</strong> {user_input['No_of_Records']}
    - <strong>Primary Variable Type:</strong> {user_input['Primary_Variable (Data Type)']}
    - <strong>Task (Purpose):</strong> {user_input['Task (Purpose)']}
    - <strong>Target Audience:</strong> {user_audience}

    <strong>Example Output Format:</strong>  
    Since your dataset's dimension is <strong>2D</strong>, has <strong>9 attributes</strong>, 
    <strong>4177 records</strong>, main primary variable is <strong>continuous</strong>, and you chose 
    <strong>comparison</strong>, a <strong>Line Chart</strong> fits best. It helps track changes over time 
    in a simple and clear way, making it easy for a <strong>{user_audience}</strong> to understand.
    """

    try:
        explanation = gpt_gateway.handle_chat(prompt)
        print(f"🧠 GPT Explanation: {explanation}")

    except Exception as e:
        explanation = f"⚠️ GPT explanation could not be generated due to an error: {str(e)}"
        print(f"❌ GPT Error: {e}")

    return explanation, note_html

def generate_final_plot(df, x_axis, y_axis, chart_type, feature_columns=None, class_column=None):
    plt.close("all") 
    plt.figure(figsize=(8, 5))
    ctype = chart_type.lower()
    print("DEBUG: ctype =", ctype)
    
    if "histogram" in ctype:
        # Gather all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # If user didn't specify or the specified `x_axis` doesn't exist, we handle it:
        if not x_axis or x_axis not in df.columns:
            # If exactly one numeric column, use it automatically:
            if len(numeric_cols) == 1:
                print(f"UPDATE: Only one numeric column found: {numeric_cols[0]}. Using it for histogram.")
                x_axis = numeric_cols[0]
            else:
                # If multiple numeric columns, fallback to the first
                if numeric_cols:
                    print("UPDATE: x_axis not found. Falling back to first numeric column:", numeric_cols[0])
                    x_axis = numeric_cols[0]
                else:
                    raise ValueError("No numeric columns found for histogram.")

        plt.hist(df[x_axis], bins=10, color="blue", alpha=0.7, edgecolor="black", linewidth=1.0)
        # Label the Y-axis as "Count" for clarity in a histogram
        plt.xlabel(x_axis)
        plt.ylabel("Count")
        plt.title(f"Histogram of {x_axis}")

    elif "pie" in ctype:
        # For pie charts, we assume x_axis is the "grouping" column and y_axis is the numeric value.
        if x_axis not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                print("UPDATE: x_axis not found. Falling back to first numeric column:", numeric_cols[0])
                x_axis = numeric_cols[0]
            else:
                raise ValueError("No numeric columns found for pie chart.")
        if not y_axis or y_axis not in df.columns:
            y_axis = x_axis
        data = df.groupby(x_axis)[y_axis].sum()
        plt.pie(data, labels=data.index, autopct="%1.1f%%")
        plt.xlabel("")
        plt.ylabel("")
        plt.title(f"Pie Chart: Distribution of {x_axis}")

    elif "treemap" in ctype:
        try:
            plt.close("all")
            print("DEBUG: Entered the Treemap block!")
            grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            print("DEBUG Treemap grouping:\n", grouped)
            
            if grouped.shape[0] < 2:
                plt.text(0.5, 0.5, "Not enough categories for a treemap", ha="center")
            else:
                labels = grouped[x_axis].astype(str)
                sizes = grouped[y_axis]

                cmap = plt.cm.viridis
                mini, maxi = sizes.min(), sizes.max()
                if mini == maxi:
                    colors = [cmap(0.5) for _ in sizes]
                else:
                    colors = [cmap((val - mini) / (maxi - mini)) for val in sizes]

                squarify.plot(
                    sizes=sizes,
                    label=labels,
                    color=colors,
                    alpha=0.8
                )
                plt.axis('off')
                plt.title(f"Treemap of {y_axis} by {x_axis}")
        except ImportError:
            plt.text(0.5, 0.5, "Treemap not implemented (requires squarify)", ha="center")
        except Exception as e:
            plt.text(0.5, 0.5, f"Treemap rendering failed: {e}", ha="center")
    elif "map" in ctype:
        try:
            world = gpd.read_file(SHAPEFILE_PATH)
            merged = world.merge(df, left_on="ADMIN", right_on=x_axis, how="left")

            ax = merged.plot(
                column=y_axis,
                cmap='OrRd',
                legend=True,
                edgecolor='black',
                figsize=(10, 6) 
            )

           
            subset = merged[~merged[y_axis].isna()]

            for idx, row in subset.iterrows():
                if row.geometry is not None:
                    centroid = row.geometry.centroid
                    ax.text(
                        x=centroid.x,
                        y=centroid.y,
                        s=str(row[x_axis]), 
                        fontsize=5,
                        ha='center',
                        va='center'
                    )

            plt.title(f"Map of {y_axis} with Country Labels")
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Map rendering failed: {e}", ha="center")


    elif "parallel" in ctype:
        try:
            from pandas.plotting import parallel_coordinates
            # If feature_columns and class_column are provided, use them; else, attempt to infer.
            if feature_columns is not None and class_column is not None:
                plot_df = df[feature_columns + [class_column]]
            else:
                # Fallback: use all numeric columns plus the last non-numeric column (if available) as the class column.
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
                if non_numeric_cols:
                    plot_df = df[numeric_cols + [non_numeric_cols[-1]]]
                    class_column = non_numeric_cols[-1]
                else:
                    plot_df = df.copy()
            parallel_coordinates(plot_df, class_column=class_column)
        except Exception as e:
            plt.text(0.5, 0.5, f"Parallel Coordinates rendering failed: {e}", ha="center")

    elif "scatter" in ctype:
        sns.scatterplot(x=df[x_axis], y=df[y_axis])
    elif "linked" in ctype:
        try:
            import networkx as nx
            G = nx.DiGraph()
            # If both x_axis and y_axis are provided and exist in the dataframe, use them as source and target.
            if x_axis in df.columns and y_axis in df.columns:
                for _, row in df.iterrows():
                    G.add_edge(row[x_axis], row[y_axis])
            else:
                # Fallback: use the x_axis column to generate a chain of nodes.
                nodes = df[x_axis].tolist()
                for i in range(len(nodes) - 1):
                    G.add_edge(nodes[i], nodes[i + 1])
                if len(nodes) == 1:
                    G.add_node(nodes[0])
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
            nx.draw_networkx_edges(G, pos, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_color='black')
            plt.title("Linked Graph Visualization")
            plt.axis("off")
        except Exception as e:
            plt.text(0.5, 0.5, f"Linked Graph rendering failed: {e}", ha="center")

    elif "line" in ctype:
        sns.lineplot(x=df[x_axis], y=df[y_axis])
    else:
        plt.text(0.5, 0.5, f"Unsupported chart type: {chart_type}", ha="center")
    
    
    if "parallel" not in ctype and "histogram" not in ctype and "pie" not in ctype:
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f"{chart_type} of {y_axis} vs {x_axis}")
    elif "parallel" in ctype:
        plt.title(f"{chart_type} Visualization")


    plot_path = os.path.join(PLOT_DIR, "generated_visualization.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

import gpt_gateway

def normalize_target_audience(input_str):
    """Normalize input to either 'Expert' or 'Non-Expert'."""
    lower_input = input_str.lower().replace("-", " ").strip()

    non_expert_terms = ["non expert", "nonexpert", "novice", "beginner", "layman"]
    if any(term in lower_input for term in non_expert_terms):
        return "Non-Expert"

    expert_terms = ["expert", "ceo", "executive", "manager", "director", "analyst"]
    if any(term in lower_input for term in expert_terms):
        return "Expert"

    return classify_audience_with_gpt_fallback(input_str)



def classify_audience_with_gpt_fallback(input_str):
    """Calls GPT to classify the input as 'Expert' or 'Non-Expert'."""
    prompt = (
        "Classify the following input as either 'Expert' or 'Non-Expert' "
        "based on the role or expertise implied. "
        "If the input indicates high-level leadership (e.g. CEO, manager, etc.), classify as 'Expert'. "
        "Return only one word: 'Expert' or 'Non-Expert'.\n"
        f"Input: \"{input_str}\""
    )
    response = gpt_gateway.handle_chat(prompt).strip().lower()
    
    if response not in ["expert", "non-expert"]:
        if "ceo" in input_str.lower() or "manager" in input_str.lower():
            return "Expert"
        return "Non-Expert"

    return response.capitalize()


def normalize_purpose(input_str):
    """Normalize input for visualization purpose, else GPT fallback."""
    distribution_terms = ['distribution', 'spread', 'proportion']
    relationship_terms = ['relationship', 'correlation', 'association']
    comparison_terms = ['comparison', 'compare']
    trends_terms = ['trend', 'trends', 'trend analysis', 'evolution']

    lower_input = input_str.lower()

    if any(term in lower_input for term in distribution_terms):
        return "distribution"
    elif any(term in lower_input for term in relationship_terms):
        return "relationship"
    elif any(term in lower_input for term in comparison_terms):
        return "comparison"
    elif any(term in lower_input for term in trends_terms):
        return "trends"
    else:
        return classify_purpose_with_gpt_fallback(input_str)


def classify_purpose_with_gpt_fallback(input_str):
    """Calls GPT to classify the purpose into distribution, relationship, comparison, or trends."""
    prompt = (
        "Classify the following input into one of the categories: "
        "'distribution', 'relationship', 'comparison', or 'trends'. "
        "Return only the category word in lowercase.\n"
        f"Input: \"{input_str}\""
    )
    response = gpt_gateway.handle_chat(prompt).strip().lower()
    
    valid_purposes = ['distribution', 'relationship', 'comparison', 'trends']
    if response not in valid_purposes:
        return input_str.lower()

    return response

# Define test cases
