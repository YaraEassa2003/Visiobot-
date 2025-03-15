import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import requests
from dotenv import load_dotenv
from openai import OpenAI
import joblib


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
    """Generates a sample visualization based on the predicted chart type."""
    plt.figure(figsize=(6, 4))

    if chart_type == 1:
        plt.hist(np.random.randn(100), bins=10, color="blue", alpha=0.7)
        plt.title("Histogram")
    elif chart_type == 2:
        plt.pie([30, 40, 30], labels=["A", "B", "C"], autopct="%1.1f%%")
        plt.title("Pie Chart")
    elif chart_type == 3:
        sns.heatmap(np.random.rand(10, 10), cmap="coolwarm", annot=True)
        plt.title("Map (Heatmap)")
    elif chart_type == 4:
        plt.barh(["A", "B", "C"], [3, 7, 5], color=["red", "green", "blue"])
        plt.title("Treemap (Bar Chart representation)")
    elif chart_type == 5:
        plt.plot(np.random.randn(50), marker="o", linestyle="-")
        plt.title("Parallel Coordinates (Simplified Line Chart)")
    elif chart_type == 6:
        plt.scatter(np.random.randn(50), np.random.randn(50), color="purple", alpha=0.5)
        plt.title("Scatter Plot")
    elif chart_type == 7:
        plt.plot(np.random.randn(50), np.random.randn(50), linestyle="--")
        plt.title("Linked Graph (Simplified)")
    elif chart_type == 8:
        plt.plot(np.arange(50), np.random.randn(50), marker="o", linestyle="-")
        plt.title("Line Chart")
    else:
        plt.text(0.5, 0.5, "Unknown Chart Type", fontsize=12, ha="center")

    plt.savefig("generated_visualization.png")
    return "generated_visualization.png"

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"
client = OpenAI(
    base_url=endpoint,
    api_key=TOKEN
)

def get_explanation(prediction, user_input):
    """Generates a simple, human-friendly explanation using GPT-4o via GitHub AI Inference."""

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

    <strong>Dataset Details:</strong>
    - <strong>Data Dimension:</strong> {user_input['Data_Dimensions']}
    - <strong>Number of Attributes:</strong> {user_input['No_of_Attributes']}
    - <strong>Number of Records:</strong> {user_input['No_of_Records']}
    - <strong>Primary Variable Type:</strong> {user_input['Primary_Variable (Data Type)']}
    - <strong>Task (Purpose):</strong> {user_input['Task (Purpose)']}
    - <strong>Target Audience:</strong> {'Expert' if user_input['Target Audience'] == 1 else 'Non-Expert'}

    <strong>Example Output Format:</strong>  
    Since your dataset's dimension is <strong>2D</strong>, has <strong>9 attributes</strong>, 
    <strong>4177 records</strong>, main primary variable is <strong>continuous</strong>, and you chose 
    <strong>comparison</strong>, a <strong>Line Chart</strong> fits best. It helps track changes over time 
    in a simple and clear way, making it easy for a <strong>Non-Expert</strong> to understand.
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an AI that provides human-friendly explanations for data visualization recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        explanation = response.choices[0].message.content.strip()
        print(f"🧠 GPT Explanation: {explanation}")

    except Exception as e:
        explanation = f"⚠️ GPT explanation could not be generated due to an error: {str(e)}"
        print(f"❌ GPT Error: {e}")

    # Return both the main explanation and the note separately
    return explanation, note_html

def generate_final_plot(df, x_axis, y_axis, chart_type):
    plt.figure(figsize=(8, 5))
    ctype = chart_type.lower()
    
    if "histogram" in ctype:
        plt.hist(df[x_axis], bins=10, color="blue", alpha=0.7)
    elif "pie" in ctype:
        # Assume df[x_axis] contains categories and df[y_axis] the numeric values
        data = df.groupby(x_axis)[y_axis].sum()
        plt.pie(data, labels=data.index, autopct="%1.1f%%")
    elif "map" in ctype:
        # For map, you might use a simple scatter plot as a placeholder or integrate a mapping library
        plt.scatter(df[x_axis], df[y_axis], c='blue', alpha=0.6)
        plt.title("Map (Placeholder)")
    elif "treemap" in ctype:
        # You can integrate squarify or create a placeholder
        try:
            import squarify
            values = df[y_axis]
            labels = df[x_axis].astype(str)
            squarify.plot(sizes=values, label=labels, alpha=0.7)
        except ImportError:
            plt.text(0.5, 0.5, "Treemap not implemented", ha="center")
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
        plt.plot(df[x_axis], df[y_axis], linestyle="--", marker="o")
    elif "line" in ctype:
        sns.lineplot(x=df[x_axis], y=df[y_axis])
    else:
        plt.text(0.5, 0.5, f"Unsupported chart type: {chart_type}", ha="center")
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f"{chart_type} of {y_axis} vs {x_axis}")
    
    plot_path = "generated_visualization.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


