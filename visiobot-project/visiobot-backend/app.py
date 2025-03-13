import os
import click
import model_utils
import dataset_utils
import gpt_gateway
from flask import Flask, request, jsonify
from flask_cors import CORS
import secrets
from flask import send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"], supports_credentials=True)
global_data = {
    "dataset_info": None,
    "dataset_uploaded": False,
    "dataset_path": None,
    "recommendation_queue": None,  # will store ranked chart names
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
        processed_data = {
            "Data_Dimensions": dataset_info["Data_Dimensions"],
            "No_of_Attributes": dataset_info["No_of_Attributes"],
            "No_of_Records": dataset_info["No_of_Records"],
            "Primary_Variable (Data Type)": dataset_info["Primary_Variable (Data Type)"],
            "Task (Purpose)": str(data["Task (Purpose)"]).lower(),
            "Target Audience": "Expert" if data["Target Audience"] == 1 else "Non-Expert",
            "Dataset Path": dataset_path
        }
        global_data["full_user_input"] = processed_data
        # Use model_utils to get the ranked list of predictions
        ranked_predictions, _ = model_utils.get_prediction(processed_data)
        # Example: ranked_predictions = [("Scatter Plot", 0.89), ("Line Chart", 0.07), ...]

        if not ranked_predictions:
            return jsonify({"error": "No valid predictions generated."}), 500

        # Store them in global_data
        global_data["recommendation_queue"] = ranked_predictions
        global_data["current_index"] = 0

        # Return the first recommendation
        return send_current_recommendation(processed_data)

    except ValueError as ve:
        return jsonify({"error": f"Invalid data type: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500


@app.route("/next-visualization", methods=["POST"])
def next_visualization():
    """Route to handle user feedback and possibly move to the next best chart."""
    user_feedback = request.json.get("feedback", "").lower()

    # Make sure we have a queue of recommendations
    if "recommendation_queue" not in global_data or global_data["recommendation_queue"] is None:
        return jsonify({"error": "No recommendation process in progress. Please start over."}), 400

    if "current_index" not in global_data:
        return jsonify({"error": "Missing current index in global_data."}), 400

    # If user says "yes", we finalize the current recommendation
    if user_feedback == "yes":
        return jsonify({"message": "Final recommendation accepted."})

    # If user says "no", we move to the next chart
    elif user_feedback == "no":
        global_data["current_index"] += 1
        if global_data["current_index"] >= len(global_data["recommendation_queue"]):
            return jsonify({
                "message": "No more visualization options are left."
            })
        # Return the next recommendation
        response_data = send_current_recommendation(global_data["full_user_input"], as_dict=True)
        response_data["pre_message"] = "Generating another visualization now..."

        return jsonify(response_data)
    # If user typed something else
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
    # Must be an absolute or correct relative path to "generated_visualization.png"
    path_to_plot = os.path.join(BASE_DIR, "generated_visualization.png")
    return send_file(path_to_plot, mimetype="image/png")

@app.route("/chat", methods=["POST"])
def chat():
    """Guides the user through the visualization recommendation process."""
    user_message = request.json.get("message", "").lower()

    if "upload" in user_message or "dataset" in user_message:
        return jsonify({"response": "Please upload your dataset to begin."})

    elif "purpose" in user_message:
        return jsonify({"response": "What is the purpose of your visualization? (e.g., comparison, distribution, trend analysis)"})

    elif "audience" in user_message:
        return jsonify({"response": "Who is your target audience? (Expert or Non-Expert)"})

    elif "done" in user_message or "generate" in user_message:
        return jsonify({"response": "Processing your data... Generating visualization now."})

    else:
        return jsonify({"response": "I can guide you through the process, but I won't generate a recommendation myself. Let's start by uploading your dataset."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
