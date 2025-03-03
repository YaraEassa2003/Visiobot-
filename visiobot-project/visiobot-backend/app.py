import os
import click
import model_utils
import dataset_utils
import gpt_gateway
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import secrets
from flask_session import Session  

app = Flask(__name__)

# ✅ Set a secure session key
app.secret_key = secrets.token_hex(32)  # Generates a new secure key each time

# ✅ Configure session storage
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True  # ✅ Keep session active
app.config["SESSION_USE_SIGNER"] = True  # ✅ Protect against tampering
app.config["SESSION_FILE_DIR"] = "./flask_session_data"  
app.config["SESSION_COOKIE_NAME"] = "visiobot_session"  # ✅ Unique session name
app.config["SESSION_COOKIE_HTTPONLY"] = False  # ✅ Allows JavaScript to access it
app.config["SESSION_COOKIE_SAMESITE"] = "None"  # ✅ Fix CORS session issues
app.config["SESSION_COOKIE_SECURE"] = True  # ✅ Set to True if using HTTPS
Session(app)  

# ✅ Ensure directories exist
UPLOAD_FOLDER = "/workspaces/codespaces-models/visiobot-project/visiobot-backend/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"], supports_credentials=True)

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

# ✅ Route 1: Upload Dataset
@app.route("/process-dataset", methods=["POST"])
def process_dataset():
    """Handles dataset upload, saves it to a fixed location, and stores path in session."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        dataset_info = dataset_utils.extract_dataset_info(file_path)
        
        # ✅ Store dataset info & file path in session
        session["dataset_info"] = dataset_info  
        session["dataset_uploaded"] = True
        session["dataset_path"] = file_path  

        session.modified = True  # ✅ Forces Flask to save session data


        print(f"✅ Dataset saved at: {file_path}")
        print("✅ Dataset info stored in session:", session["dataset_info"])

    except Exception as e:
        return jsonify({"error": f"Dataset processing failed: {str(e)}"}), 500

    return jsonify({
        "message": "Dataset processed successfully. Now provide the Task (Purpose) and Target Audience.",
        "dataset_details": dataset_info
    })

# ✅ Route 2: Get Visualization
@app.route("/get-visualization", methods=["POST"])
def get_visualization():
    """Retrieves dataset from session, combines it with user input, and sends it to the model."""
    data = request.json

    # ✅ Check if session contains dataset info
    print("🔎 Checking session for dataset info...")

    

    dataset_info = session.get("dataset_info", {})
    dataset_path = session.get("dataset_path", None)

    print("✅ Retrieved dataset info:", dataset_info)
    print("✅ Retrieved dataset path:", dataset_path)

    if not session.get("dataset_uploaded"):
        print("❌ Dataset not found in session!")
        return jsonify({"error": "❌ No dataset uploaded. Please upload a dataset first."}), 400
    
    if dataset_path is None:
        return jsonify({"error": "Dataset path not found in session. Please re-upload."}), 400

    required_fields = ["Task (Purpose)", "Target Audience"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields: Task (Purpose) and Target Audience"}), 400

    try:
        # Merge extracted dataset info with user inputs
        processed_data = {
            "Data_Dimensions": dataset_info["Data_Dimensions"],
            "No_of_Attributes": dataset_info["No_of_Attributes"],
            "No_of_Records": dataset_info["No_of_Records"],
            "Primary_Variable (Data Type)": dataset_info["Primary_Variable (Data Type)"],
            "Task (Purpose)": str(data["Task (Purpose)"]).lower(),
            "Target Audience": 1 if data["Target Audience"] == 1 else 0,
            "Dataset Path": dataset_path  # ✅ Pass dataset file path for further processing if needed
        }

        print("📊 Processed data for model:", processed_data)

        prediction, visualization_plot = model_utils.get_prediction(processed_data)

        return jsonify({
            "prediction": prediction,
            "visualization_plot": visualization_plot,
            "explanation": model_utils.get_explanation(prediction, processed_data)
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid data type: {str(ve)}"}), 400

    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

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
