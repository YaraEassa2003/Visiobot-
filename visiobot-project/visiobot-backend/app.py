import os
import model_utils
import dataset_utils
import gpt_gateway
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
import secrets
from flask import send_file
import re
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"], supports_credentials=True)
global_data = {
    "dataset_info": None,
    "dataset_uploaded": False,
    "dataset_path": None,
    "recommendation_queue": None,  
    "current_index": 0
}
UPLOAD_FOLDER = "/workspaces/codespaces-models/visiobot-project/visiobot-backend/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to VisioBot API! Your server is running successfully.",
        "routes": {
            "/process-dataset": "POST - Upload a dataset to process",
            "/get-visualization": "POST - Get a visualization recommendation",
            "/chat": "POST - Interact with the chatbot"
        }
    }), 200


@app.route("/process-dataset", methods=["POST"])
def process_dataset():
    """Handles dataset upload, saves it to a fixed location, and stores path in global_data."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        dataset_info = dataset_utils.extract_dataset_info(file_path)

        global_data["dataset_info"] = dataset_info
        global_data["dataset_uploaded"] = True
        global_data["dataset_path"] = file_path

        print(f"✅ Dataset saved at: {file_path}")
        print("✅ Dataset info stored:", global_data["dataset_info"])

    except Exception as e:
        return jsonify({"error": f"Dataset processing failed: {str(e)}"}), 500

    return jsonify({
        "message": "Dataset processed successfully. Now provide the Task (Purpose) and Target Audience.",
        "dataset_details": dataset_info
    })

@app.route("/get-visualization", methods=["POST"])
def get_visualization():
    """Initial route to get the top recommendation and store the entire ranked queue."""
    data = request.json
    if not global_data["dataset_uploaded"]:
        return jsonify({"error": "No dataset uploaded. Please upload a dataset first."}), 400

    required_fields = ["Task (Purpose)", "Target Audience"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields: Task (Purpose) and Target Audience"}), 400

    dataset_info = global_data["dataset_info"]
    dataset_path = global_data["dataset_path"]
    if not dataset_path:
        return jsonify({"error": "Dataset path not found. Please re-upload."}), 400

    try:
        # Use GPT-based classification to normalize the inputs.
        # Correct calls that match model_utils.py:
        normalized_purpose = model_utils.normalize_purpose(data["Task (Purpose)"])
        normalized_audience = model_utils.normalize_target_audience(data["Target Audience"])


        processed_data = {
            "Data_Dimensions": dataset_info["Data_Dimensions"],
            "No_of_Attributes": dataset_info["No_of_Attributes"],
            "No_of_Records": dataset_info["No_of_Records"],
            "Primary_Variable (Data Type)": dataset_info["Primary_Variable (Data Type)"],
            "Task (Purpose)": normalized_purpose,
            "Target Audience": normalized_audience,
            "Dataset Path": dataset_path
        }
        global_data["full_user_input"] = processed_data

        ranked_predictions, _ = model_utils.get_prediction(processed_data)
        if not ranked_predictions:
            return jsonify({"error": "No valid predictions generated."}), 500

        global_data["recommendation_queue"] = ranked_predictions
        global_data["current_index"] = 0

        return send_current_recommendation(processed_data)

    except ValueError as ve:
        return jsonify({"error": f"Invalid data type: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500


@app.route("/next-visualization", methods=["POST"])
def next_visualization():
    """Route to handle user feedback and possibly move to the next best chart."""
    user_feedback = request.json.get("feedback", "").lower()

    if "recommendation_queue" not in global_data or global_data["recommendation_queue"] is None:
        return jsonify({"error": "No recommendation process in progress. Please start over."}), 400

    if "current_index" not in global_data:
        return jsonify({"error": "Missing current index in global_data."}), 400

    if user_feedback == "yes":
        chart_name, _ = global_data["recommendation_queue"][global_data["current_index"]]
        global_data["final_chart_type"] = chart_name 
        try:
            return get_dataset_columns()  # This returns a JSON with the "final_message" field.
        except Exception as e:
            return jsonify({"error": f"Failed during final step: {str(e)}"}), 500

    elif user_feedback == "no":
        global_data["current_index"] += 1
        if global_data["current_index"] >= len(global_data["recommendation_queue"]):
            return jsonify({
                "message": "No more visualization options are left."
            })
        response_data = send_current_recommendation(global_data["full_user_input"], as_dict=True)
        response_data["pre_message"] = "Generating another visualization now..."

        return jsonify(response_data)
    return jsonify({"error": "Invalid feedback. Please respond with 'yes' or 'no'."}), 400

def send_current_recommendation(user_input, as_dict=False):
    current_index = global_data["current_index"]
    recommendation_queue = global_data["recommendation_queue"]
    if current_index >= len(recommendation_queue):
        result = {"message": "No more visualization options."}
        return result if as_dict else jsonify(result)

    chart_name, _ = recommendation_queue[current_index]
    explanation, note_html = model_utils.get_explanation(chart_name, user_input)

    result = {
        "prediction": chart_name,
        "explanation": explanation,
        "note": note_html,
        "ask_feedback": "Are you satisfied with this recommendation? (Yes/No)"
    }
    return result if as_dict else jsonify(result)

@app.route("/plot.png")
def serve_plot():
    path_to_plot = os.path.join(BASE_DIR, "generated_plots", "generated_visualization.png")
    return send_file(path_to_plot, mimetype="image/png")



@app.route("/get-dataset-columns", methods=["POST"])
def get_dataset_columns():
    """
    Reads the uploaded dataset, extracts its column names,
    and calls GPT with a minimal prompt:
      - Just enumerates columns in <strong> tags
      - Asks the user which columns they want to visualize
    """
    try:
        dataset_path = global_data.get("dataset_path")
        if not dataset_path:
            return jsonify({"error": "Dataset not found."}), 400

        df = pd.read_csv(dataset_path)
        columns = list(df.columns)
        formatted_columns = ", ".join([f"<strong>{col}</strong>" for col in columns])

        message = (
            f"The dataset you uploaded contains columns: {formatted_columns}. "
            "Which columns and what insights would you like to visualize?"
        )
        return jsonify({"final_message": message})

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve dataset columns: {str(e)}"}), 500


@app.route("/final-plot", methods=["POST"])
def final_plot():
    """
    Uses GPT to interpret user input for columns,
    then generates a plot using the final recommended chart type
    from the model. This version simulates image analysis by using
    the dataset summary to generate insights, emphasizing a concise,
    confident tone that begins with "The plot shows..."
    """
    try:
        user_request = request.json.get("user_request", "").strip()
        dataset_path = global_data.get("dataset_path")
        final_chart_type = global_data.get("final_chart_type")

        if not dataset_path or not final_chart_type:
            return jsonify({"error": "No final chart type or dataset found."}), 400

        df = pd.read_csv(dataset_path)
        all_columns = list(df.columns)

        prompt = (
            "Return only valid JSON. No extra text.\n"
            f"Columns: {', '.join(all_columns)}.\n"
            f"User wants: '{user_request}'.\n"
            "Which columns should be x_axis and y_axis?\n"
            "Format: {\"x_axis\": \"<col1>\", \"y_axis\": \"<col2>\"}"
        )

        gpt_response = gpt_gateway.handle_chat(prompt)
        print("🧠 GPT raw response:", repr(gpt_response))

        match = re.search(r"\{.*?\}", gpt_response, re.DOTALL)
        if not match:
            return jsonify({"error": "GPT did not return a JSON object."}), 500

        json_str = match.group(0) 
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse JSON from GPT's response."}), 500

        x_axis = parsed.get("x_axis")
        y_axis = parsed.get("y_axis")
        if not x_axis or not y_axis:
            return jsonify({"error": "GPT JSON missing x_axis or y_axis."}), 500

        plot_path = model_utils.generate_final_plot(df, x_axis, y_axis, final_chart_type)

        summary_stats = df.describe().to_string()

        explanation_prompt = f"""
You are a seasoned business intelligence analyst interpreting a '{final_chart_type}' 
plot of '{y_axis}' vs '{x_axis}'. Although you do not see the actual image, 
here is the dataset summary:
{summary_stats}

In 2–3 sentences, state the most important insights that will impact business decisions. 
Use direct, confident language (e.g., "The plot shows...") and avoid numbering your points.
        """

        plot_description = gpt_gateway.handle_chat(explanation_prompt)

        base_url = "https://cautious-space-train-wrgx7wx5j7g6fv6xr-5000.app.github.dev"
        cache_buster = int(time.time())  # e.g. 1679000000
        plot_url = f"{base_url}/plot.png?cb={cache_buster}"

        return jsonify({
            "message": f"Here is your {final_chart_type} for '{y_axis}' and '{x_axis}'.",
            "plot_description": plot_description.strip(),
            "plot_url": plot_url,
            "ask_restart": "Would you like to start over with a new dataset? (Yes/No)"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate final plot: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    A stricter chat flow with a GPT fallback if the GPT call fails.
    """
    user_message = request.json.get("message", "").strip()
    msg_lower = user_message.lower()

    greetings = ["hello", "hi", "hey"]
    if msg_lower in greetings:
        greeting_prompt = (
            "You are VisioBot, a data visualization assistant. "
            "The user just said hello. Greet them warmly but briefly. "
            "Tell them they need to upload a dataset, specify purpose, and specify audience. "
            "End by inviting them to upload."
        )
        try:
            # If GPT works, great
            response = gpt_gateway.handle_chat(greeting_prompt)
        except Exception as e:
            # Fallback if GPT fails
            print(f"❌ GPT error: {e}")
            response = (
                "Hello! I’m VisioBot. Currently I’m unable to use GPT, "
                "but please upload your dataset, specify the purpose, and specify your audience."
            )
        return jsonify({"response": response})

    # If not a greeting, handle the rest of the flow:
    if "upload" in msg_lower or "dataset" in msg_lower:
        return jsonify({"response": "Please upload your dataset to begin."})
    elif "purpose" in msg_lower:
        return jsonify({"response": "What is the purpose of your visualization? (Distribution, Relationship, Comparison, Trends)"})
    elif "audience" in msg_lower:
        return jsonify({"response": "Who is your target audience? (Expert or Non-Expert)"})
    elif "done" in msg_lower or "generate" in msg_lower:
        return jsonify({"response": "Processing your data... Generating a visualization now."})

    # Off-topic fallback:
    fallback_prompt = (
        "You are VisioBot, a helpful data visualization assistant. "
        "The user said: '{user_message}'. Respond naturally, but do NOT recommend chart types. "
        "If they want a visualization, they must follow the flow: upload → purpose → audience → done."
        "Do NOT converse much your purpose here is to only make the user feel that they are chatting with a human expert."
        "If they say anything else, remind them of the flow and that this is the way they can see best recommendations."
    ).format(user_message=user_message)

    try:
        conversation_reply = gpt_gateway.handle_chat(fallback_prompt)
    except Exception as e:
        print(f"❌ GPT error: {e}")
        conversation_reply = (
            "I can't help further. If you want to proceed, "
            "please upload your dataset, specify purpose, and specify audience."
        )

    return jsonify({"response": conversation_reply})


@app.route("/restart", methods=["POST"])
def restart():
    """
    Resets global_data so the user can start the entire process again.
    """
    global_data["dataset_info"] = None
    global_data["dataset_uploaded"] = False
    global_data["dataset_path"] = None
    global_data["recommendation_queue"] = None
    global_data["current_index"] = 0
    global_data["final_chart_type"] = None


    return jsonify({
        "restart_message": "All set! Let's start fresh. Please upload a  dataset again."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
