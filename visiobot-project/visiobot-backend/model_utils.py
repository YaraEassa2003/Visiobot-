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

    prompt = f"""
    Explain why a <strong>{prediction}</strong> is the best choice based on the dataset details below.

    Keep it <strong>short, simple, and natural</strong>. Use clear, easy-to-understand language. <strong>List all six inputs first</strong>, then give a <strong>brief, direct reason</strong> why this chart is the best choice. Avoid technical terms like "intuitive" or "effectively."

    <strong>Dataset Details:</strong>
    - <strong>Data Dimension:</strong> {user_input['Data_Dimensions']}
    - <strong>Number of Attributes:</strong> {user_input['No_of_Attributes']}
    - <strong>Number of Records:</strong> {user_input['No_of_Records']}
    - <strong>Primary Variable Type:</strong> {user_input['Primary_Variable (Data Type)']}
    - <strong>Task Purpose:</strong> {user_input['Task (Purpose)']}
    - <strong>Target Audience:</strong> {'Expert' if user_input['Target Audience'] == 1 else 'Non-Expert'}

    <strong>Example Output Format:</strong>  
    Since your dataset's dimension is <strong>2D</strong>, has <strong>9 attributes</strong>, <strong>4177 records</strong>, main primary variable is <strong>continuous</strong>, and you chose <strong>comparison</strong>, a <strong>Line Chart</strong> fits best. It helps track changes over time in a simple and clear way, making it easy for a Non-Expert to understand.
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
        print(f"🧠 GPT Explanation: {explanation}")  # Debugging: See explanation in terminal
    except Exception as e:
        explanation = f"⚠️ GPT explanation could not be generated due to an error: {str(e)}"
        print(f"❌ GPT Error: {e}")  # Debugging: Log any errors

    return explanation

# Define test cases
test_cases = [
    { #0
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 50,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    { #1
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 20,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 10,
        "No_of_Records": 500,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "trends",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 2,
        "No_of_Records": 150,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "ND",
        "No_of_Attributes": 4,
        "No_of_Records": 600,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "comparison",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 5,
        "No_of_Records": 750,
        "Primary_Variable (Data Type)": "geographical",
        "Task (Purpose)": "relationship",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 7,
        "No_of_Records": 350,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "comparison",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "ND",
        "No_of_Attributes": 9,
        "No_of_Records": 1100,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "relationship",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 3,
        "No_of_Records": 300,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "relationship",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 6,
        "No_of_Records": 850,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "trends",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 8,
        "No_of_Records": 1300,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "trends",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 10,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },  # Expected: Pie Chart
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 10,
        "No_of_Records": 5000,
        "Primary_Variable (Data Type)": "geographical",
        "Task (Purpose)": "comparison",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 20,
        "No_of_Records": 2500,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "comparison",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 9,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 2,
        "No_of_Records": 500,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "relationship",
        "Target Audience": "Expert"
    },  # Expected: Map
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 3,
        "No_of_Records": 300,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "trends",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 3,
        "No_of_Records": 500,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "relationship",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 9,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },

    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 35,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "tredistributionnds",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 60,
        "Primary_Variable (Data Type)": "geographical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Expert"
    }
]



# Test with predefined test cases
predictions = []
for test_case in test_cases:
    prediction, top_3 = get_prediction(test_case)
    predictions.append((test_case, prediction, top_3))

# Display results
df_results = pd.DataFrame(predictions, columns=["Test Case", "Predicted Chart", "Top 3 Predictions"])
print(df_results)



