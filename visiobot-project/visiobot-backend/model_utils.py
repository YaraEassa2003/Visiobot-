import os
import numpy as np
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

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load preprocessing pipeline
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

    # Standardize numerical values (Ensure both attributes are transformed together)
    standardized_values = scaler.transform(
        pd.DataFrame([[user_input["No_of_Attributes"], user_input["No_of_Records"]]],
                    columns=["No_of_Attributes", "No_of_Records"])
    )
    print("Training Mean & Std Dev:", scaler.mean_, scaler.scale_)

    # Create an empty DataFrame with the correct feature structure
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0  # Initialize all values to zero

    # Insert standardized numerical values
    input_df["No_of_Attributes"] = standardized_values[0, 0]
    input_df["No_of_Records"] = standardized_values[0, 1]

    # Apply One-Hot Encoding using `one_hot_columns` from the pipeline
    categorical_features = ["Data_Dimensions", "Primary_Variable (Data Type)", "Task (Purpose)", "Target Audience"]
    for feature in categorical_features:
        col_name = f"{feature}_{user_input[feature]}"
        if col_name in one_hot_columns:  # Ensure only valid columns are set
            input_df[col_name] = 1

    # Fill missing values with 0 to maintain feature alignment
    input_df = input_df.fillna(0)

    # Ensure final input matches model expectations
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

    # Make prediction
    prediction = model.predict(input_data)[0]  # Get probability scores

    # Rank all 8 visualizations based on probability scores (descending order)
    ranked_indices = np.argsort(prediction)[::-1]  # Sort in descending order
    ranked_visualizations = [(chart_type_mapping[i + 1], prediction[i]) for i in ranked_indices]

    print(f"🏆 Ranked Predictions: {ranked_visualizations}")  # Debugging

    # Generate a plot for the highest-ranked visualization
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
        # Line Chart
        x = np.arange(50)
        y = np.random.randn(50).cumsum()  # cumsum to simulate a trending line
        plt.plot(x, y, marker="o", linestyle="-", color="green")
        plt.title("Line Chart (Placeholder)")

    elif chart_type == 3:
    # Linked Graph as a NETWORK DIAGRAM using NetworkX
        try:
            
            # For demonstration, generate a small random graph
            # Replace with your own data if you have adjacency info
            G = nx.erdos_renyi_graph(n=8, p=0.3, seed=42)
            
            # Spring layout positions nodes in a visually appealing way
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes, edges, and labels
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
            nx.draw_networkx_edges(G, pos, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_color='black')
            
            plt.title("Linked Graph (Network Diagram Placeholder)")
            plt.axis('off')
        except ImportError:
            # Fallback if networkx isn't installed
            plt.text(0.5, 0.5, "Network Diagram not implemented (requires networkx)", ha="center")
        except Exception as e:
            plt.text(0.5, 0.5, f"Network Diagram rendering failed: {e}", ha="center")

    elif chart_type == 4:
        # Actual MAP rendering using GeoPandas
        try:
            
            shapefile_path = os.path.join(DATA_DIR, "ne_110m_admin_0_countries.shp")
            world = gpd.read_file(shapefile_path)
            world.plot(column='pop_est', cmap='OrRd', legend=True, edgecolor='black')
            plt.title("World Map (Colored by Population Estimate)")
            plt.axis('off')
            
        except Exception as e:
            # Fallback if something goes wrong
            plt.text(0.5, 0.5, f"Map rendering failed: {e}", ha="center")

    elif chart_type == 5:
        # Parallel Coordinates
        # We'll do a simplified version if we don't have a real dataset
        # In a real scenario, you'd use `parallel_coordinates(df, class_column=...)`
        try:
            from pandas.plotting import parallel_coordinates
            # Create a small random DataFrame for demonstration
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
        # Pie Chart
        # We'll use random proportions that sum to 1
        data = np.random.randint(1, 10, 4)
        labels = [f"Cat {i+1}" for i in range(4)]
        plt.pie(data, labels=labels, autopct="%1.1f%%")
        plt.title("Pie Chart (Placeholder)")

    elif chart_type == 7:
        # Scatter Plot
        x = np.random.randn(50)
        y = np.random.randn(50)
        plt.scatter(x, y, color="purple", alpha=0.5)
        plt.title("Scatter Plot (Placeholder)")

    elif chart_type == 8:
        # Treemap
        # We'll try squarify if available; otherwise, show text
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

    # Save and return path
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
    # Mapping of each chart type to its typical audience
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
        # If it's 1 => Expert, else => Non-Expert
        user_audience = "Expert" if raw_audience == 1 else "Non-Expert"
    else:
        # Convert user input to a single lowercase string without hyphens/spaces
        # e.g. "non-expert" -> "nonexpert", "expert" -> "expert"
        norm = raw_audience.lower().replace("-", "").replace(" ", "")
        if norm == "expert":
            user_audience = "Expert"
        else:
            # Anything else is treated as "nonexpert"
            user_audience = "Non-Expert"

    # 2) Find the recommended chart's typical audience, e.g. "expert" or "non-expert"
    recommended_aud = typical_audience.get(prediction, "non-expert")
    # We'll convert it to "Expert"/"Non-Expert" for final display
    if recommended_aud.lower() == "expert":
        recommended_audience = "Expert"
    else:
        recommended_audience = "Non-Expert"

    # Build a note if there's a mismatch
    note_html = ""
    if recommended_audience != user_audience:
        note_html = (
            "<strong>Note:</strong> "
            f"Although you requested a visualization for {user_audience} users, "
            f"based on my analysis, {prediction} appears to best suit your dataset's characteristics."
        )

    # Construct prompt for GPT
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
        # Use gpt_gateway instead of a local OpenAI client
        explanation = gpt_gateway.handle_chat(prompt)
        print(f"🧠 GPT Explanation: {explanation}")

    except Exception as e:
        explanation = f"⚠️ GPT explanation could not be generated due to an error: {str(e)}"
        print(f"❌ GPT Error: {e}")

    # Return both the main explanation and the note separately
    return explanation, note_html

def generate_final_plot(df, x_axis, y_axis, chart_type):
    plt.close("all")  # <--- forcibly close any old figures
    plt.figure(figsize=(8, 5))
    ctype = chart_type.lower()
    print("DEBUG: ctype =", ctype)
    
    if "histogram" in ctype:
        plt.hist(df[x_axis], bins=10, color="blue", alpha=0.7)
    elif "pie" in ctype:
        # Assume df[x_axis] contains categories and df[y_axis] the numeric values
        data = df.groupby(x_axis)[y_axis].sum()
        plt.pie(data, labels=data.index, autopct="%1.1f%%")
    

    elif "treemap" in ctype:
        try:
            plt.close("all")
            print("DEBUG: Entered the Treemap block!")
            # Group by category and sum the numeric values
            grouped = df.groupby(x_axis)[y_axis].sum().reset_index()
            print("DEBUG Treemap grouping:\n", grouped)
            
            # Check if we have more than one category:
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

           # merged already has your shapefile and user data combined
            ax = merged.plot(
                column=y_axis,
                cmap='OrRd',
                legend=True,
                edgecolor='black',
                figsize=(10, 6)  # optional for bigger plot
            )

            # Filter out rows that have no match in the user's dataset
            # i.e., no population data or whatever your y_axis is
            subset = merged[~merged[y_axis].isna()]

            # Now label only those countries that have user data
            for idx, row in subset.iterrows():
                if row.geometry is not None:
                    centroid = row.geometry.centroid
                    ax.text(
                        x=centroid.x,
                        y=centroid.y,
                        s=str(row[x_axis]),  # e.g. the user’s country name from your dataset
                        fontsize=5,
                        ha='center',
                        va='center'
                    )

            plt.title(f"Map of {y_axis} with Country Labels")
            plt.axis('off')
        except Exception as e:
            plt.text(0.5, 0.5, f"Map rendering failed: {e}", ha="center")


    elif "parallel" in ctype:
        # For parallel coordinates, you might need a dedicated function (e.g. pandas.plotting.parallel_coordinates)
        try:
            from pandas.plotting import parallel_coordinates
            parallel_coordinates(df, class_column=x_axis)
        except Exception:
            plt.text(0.5, 0.5, "Parallel Coordinates not implemented", ha="center")
    elif "scatter" in ctype:
        sns.scatterplot(x=df[x_axis], y=df[y_axis])
    elif "linked" in ctype:
        try:
            # Create a small random graph
            G = nx.erdos_renyi_graph(n=8, p=0.3, seed=42)

            # Lay it out
            pos = nx.spring_layout(G, seed=42)

            # Draw
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
            nx.draw_networkx_edges(G, pos, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_color='black')

            plt.title("Linked Graph (Network Diagram Placeholder)")
            plt.axis("off")
        except ImportError:
            plt.text(0.5, 0.5, "Network Diagram not implemented (requires networkx)", ha="center")
        except Exception as e:
            plt.text(0.5, 0.5, f"Network Diagram rendering failed: {e}", ha="center")
    elif "line" in ctype:
        sns.lineplot(x=df[x_axis], y=df[y_axis])
    else:
        plt.text(0.5, 0.5, f"Unsupported chart type: {chart_type}", ha="center")
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"{chart_type} of {y_axis} vs {x_axis}")
    
    plot_path = os.path.join(PLOT_DIR, "generated_visualization.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


