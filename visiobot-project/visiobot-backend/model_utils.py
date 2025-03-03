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

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "visiobot_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Load dataset to get correct preprocessing parameters
dataset_path = os.path.join(BASE_DIR, "modifiedDataset_WithDIM.csv")
df = pd.read_csv(dataset_path)

# Standardize numerical columns based on training values
numerical_cols = ["No_of_Attributes", "No_of_Records"]
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Store mean and scale values for future standardization
scaler_mean = scaler.mean_
scaler_scale = scaler.scale_

# One-Hot Encode categorical features based on training
df = pd.get_dummies(df, columns=['Data_Dimensions', 'Primary_Variable (Data Type)', 'Task (Purpose)', 'Target Audience'], drop_first=True, dtype=int)

# Define visualization class mapping
chart_type_mapping = {
    1: "Histogram",
    2: "Pie Chart",
    3: "Map",
    4: "Treemap",
    5: "Parallel Coordinates",
    6: "Scatter Plot",
    7: "Linked Graph",
    8: "Line Chart"
}

def preprocess_input(user_input):
    """Prepares user input for model prediction by applying one-hot encoding and standardization."""
    global df  # Use the same structure as the training dataset

    # Standardize numerical columns
    standardized_attrs = (np.array([[user_input["No_of_Attributes"], user_input["No_of_Records"]]]) - scaler_mean) / scaler_scale

    # Create an empty dataframe with the same structure as training data
    input_df = pd.DataFrame(columns=df.columns)
    input_df.loc[0] = 0  # Initialize with zeros

    # Fill numerical values
    input_df["No_of_Attributes"] = standardized_attrs[0][0]
    input_df["No_of_Records"] = standardized_attrs[0][1]

    # One-hot encode categorical values
    cat_features = ["Data_Dimensions", "Primary_Variable (Data Type)", "Task (Purpose)", "Target Audience"]
    for feature in cat_features:
        col_name = f"{feature}_{user_input[feature]}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

    # Debugging: Print before filtering correct features
    print("🛠 Raw input data before filtering:\n", input_df)

    # Ensure only expected features are passed
    expected_features = model.input_shape[1]  # Expected input size (12)
    final_input = input_df[df.columns].to_numpy()[:, :expected_features]  # Trim extra features

    print("✅ Corrected final input shape for model:", final_input.shape)

    return final_input


def get_prediction(user_input):
    """Predicts the best visualization type for the given user input."""

    # Preprocess input for model
    input_data = preprocess_input(user_input)

    print("🔎 Final input shape being sent to model:", input_data.shape)

    # Check input feature count before passing to model
    expected_features = model.input_shape[1]
    if input_data.shape[1] != expected_features:
        return "❌ Feature mismatch! Expected {}, but got {}.".format(expected_features, input_data.shape[1]), None

    # Get model prediction
    prediction = model.predict(input_data)
    predicted_chart_index = np.argmax(prediction) + 1  # Convert to 1-based index

    # Print raw model output for debugging
    print("📊 Model Raw Prediction Probabilities:", prediction)
    print("✅ Model Final Prediction Index:", predicted_chart_index)


    # Show top 3 highest probabilities for debugging
    top_indices = np.argsort(prediction[0])[-3:][::-1]
    print(f"🏆 Top 3 Predictions: {[(chart_type_mapping[i+1], prediction[0][i]) for i in top_indices]}")

    # Get the corresponding chart type name
    chart_type = chart_type_mapping.get(predicted_chart_index, "Unknown Chart Type")

    # Generate the visualization plot
    visualization_plot = generate_visualization(predicted_chart_index)

    return chart_type, visualization_plot


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

# Azure OpenAI API settings
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

# Load environment variables
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

# Azure OpenAI API settings
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

# ✅ Create OpenAI Client for GitHub AI Inference
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