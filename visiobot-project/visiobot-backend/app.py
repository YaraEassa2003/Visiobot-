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
from dateutil.relativedelta import relativedelta 
from flask import send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    expose_headers=["Content-Disposition"],   # let the browser read it
    allow_headers=["Content-Type"],           # allow JSON POSTs
) 

@app.after_request
def enforce_utf8(resp):
    # add the charset only for textual responses
    if resp.mimetype.startswith(("application/json", "text/")):
        resp.headers["Content-Type"] = f"{resp.mimetype}; charset=utf-8"
    return resp

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

# ── NEW route, place it near /plot.png ─────────────────────────────────────────
@app.route("/download-plot")
def download_plot():
    """Send the latest generated plot *as an attachment* so the browser shows
    the normal ‘Save as…’ dialogue."""
    path = os.path.join(BASE_DIR, "generated_plots",
                        "generated_visualization.png")
    return send_file(
        path,
        mimetype="image/png",
        as_attachment=True,              # 👈 triggers download
        download_name="visualization.png"
    )

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
        dataset_info = dataset_utils.extract_dataset_info(file_path, use_gpt_for_hierarchy=True)

        global_data["dataset_info"] = dataset_info
        global_data["dataset_uploaded"] = True
        global_data["dataset_path"] = file_path

        print(f"✅ Dataset saved at: {file_path}")
        print("✅ Dataset info stored:", global_data["dataset_info"])

    except Exception as e:
        return jsonify({"error": f"Dataset processing failed: {str(e)}"}), 500

    return jsonify({
        "message": "Dataset processed successfully. Now provide the Task (Purpose) and Audience.",
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
    try:
        user_feedback = request.json.get("feedback", "").lower()

        if "recommendation_queue" not in global_data or global_data["recommendation_queue"] is None:
            return jsonify({"error": "No recommendation process in progress. Please start over."}), 200  # CHANGED

        if "current_index" not in global_data:
            return jsonify({"error": "Missing current index in global_data."}), 200  # CHANGED

        if user_feedback == "yes":
            chart_name, _ = global_data["recommendation_queue"][global_data["current_index"]]
            global_data["final_chart_type"] = chart_name
            return get_dataset_columns()  # this already returns 200
        elif user_feedback == "no":
            global_data["current_index"] += 1
            if global_data["current_index"] >= len(global_data["recommendation_queue"]):
                return jsonify({"message": "No more visualization options are left.", "ask_restart": "Would you like to start over with a new dataset? (Yes/No)"}), 200  # CHANGED
            response_data = send_current_recommendation(global_data["full_user_input"], as_dict=True)
            return jsonify(response_data), 200  # CHANGED

        return jsonify({"error": "Invalid feedback. Please respond with 'yes' or 'no'."}), 200  # CHANGED

    except Exception as e:
        # CHANGED: always return JSON 200 so fetch().json() never throws
        return jsonify({
            "error": f"Next‑visualization error: {str(e)}",
            "ask_restart": "Would you like to restart? (Yes/No)"
        }), 200

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
        and asks the user for specific inputs based on the chosen chart type.
        """
        try:
            dataset_path = global_data.get("dataset_path")
            if not dataset_path:
                return jsonify({"error": "Dataset not found."}), 400

            df = pd.read_csv(dataset_path)
            cols = list(df.columns)
            formatted_columns = ", ".join([f"<strong>{col}</strong>" for col in cols])
            chart = global_data.get("final_chart_type", "").lower()

            prompts = {
                "histogram":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which single numeric column should I use for the X-axis (to plot its distribution)?",

                "line chart":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which column is your time (or sequential) variable for the X-axis, "
                    "and which numeric column for the Y-axis (to show trends)?",

                "scatter plot":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which two numeric columns should I use for the X-axis and Y-axis to show their relationship?",

                "pie chart":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which categorical column should define the slices and which numeric column should determine slice size?",

                "treemap":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which single categorical column should define the tiles, "
                    "and which numeric column should determine their sizes?",

                "map":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which column contains your geographic identifiers (e.g., country or state names or codes), "
                    "and which numeric column should color the map?",

                "parallel coordinates":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which numeric columns should I plot as features, "
                    "and which categorical column should I use to color/group the lines?",

                "linked graph":
                    f"The dataset you uploaded contains columns: {formatted_columns}. "
                    "Which column should be the source node, "
                    "and which column should be the target node for the network edges?"
            }

            prompt = prompts.get(
                chart,
                f"The dataset you uploaded contains columns: {formatted_columns}. "
                "Which columns would you like to use, and what insight would you like to visualize?"
            )

            return jsonify({"final_message": prompt})

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

        import re
        time_match = re.search(r"last\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)", user_request, re.IGNORECASE)
        if time_match:
            num = int(time_match.group(1))
            unit = time_match.group(2).lower()
            date_columns = [col for col in df.columns if "date" in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                max_date = df[date_col].max()
                if pd.notnull(max_date):
                    if unit in ['day', 'days']:
                        offset = pd.DateOffset(days=num)
                    elif unit in ['week', 'weeks']:
                        offset = pd.DateOffset(weeks=num)
                    elif unit in ['month', 'months']:
                        offset = pd.DateOffset(months=num)
                    elif unit in ['year', 'years']:
                        offset = pd.DateOffset(years=num)
                    else:
                        offset = pd.DateOffset(months=num) 

                    cutoff = max_date - offset
                    df = df[df[date_col] >= cutoff]
                    print(f"UPDATE: Data filtered by last {num} {unit} using date column '{date_col}'.")
        all_columns = list(df.columns)

        prompt =  (
            "Return only valid JSON. No extra text.\n"
            f"Columns: {', '.join(all_columns)}.\n"
            f"User wants: '{user_request}'.\n"
            "If user wants a normal chart, return {\"x_axis\":\"col1\",\"y_axis\":\"col2\"}.\n"
            "If user wants parallel coordinates, return "
            "{\"chart_type\":\"parallel\",\"feature_columns\":[\"colA\",\"colB\"],\"class_column\":\"class\"}.\n"
            "If user wants a linked graph, return {\"x_axis\":\"<source_column>\",\"y_axis\":\"<target_column>\"}.\n"
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

        raw_x = (parsed.get("x_axis") or "").strip()
        raw_y = parsed.get("y_axis") or raw_x
        col_map = {col.lower(): col for col in df.columns}
        x_axis = col_map.get(raw_x.lower(), raw_x)   # CHANGED
        y_axis = col_map.get(raw_y.lower(), raw_y)   # CHANGED
        # ────────────────────────────────────────────────
        chart_type_key = (parsed.get("chart_type") or "").lower()
        feature_cols = parsed.get("feature_columns", [])    # will be [] if not parallel
        class_col    = parsed.get("class_column", None)     # will be None if not parallel
        used_cols_str = ""
        subset_summary = ""

        if chart_type_key == "parallel" or "parallel" in final_chart_type.lower():
            feature_cols = parsed.get("feature_columns", [])
            class_col = parsed.get("class_column")
            if not feature_cols or not class_col:
                return jsonify({"error": "Invalid parallel coordinates JSON: missing feature_columns or class_column."}), 400

            plot_path = model_utils.generate_final_plot(
                df=df,
                chart_type=final_chart_type,  
                x_axis=None,                  
                y_axis=None,                 
                feature_columns=feature_cols, 
                class_column=class_col        
            )

            used_cols_str = f"Feature Columns: {', '.join(feature_cols)}, Class Column: {class_col}"
            numeric_subset = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_subset:
                subset_df = df[numeric_subset].copy()
                subset_summary = subset_df.describe().to_string()
            else:
                subset_summary = "No numeric columns in feature_columns."
        else:
            # get whatever GPT returned (may be None for y_axis)
            x_axis = parsed.get("x_axis") or ""
            y_axis = parsed.get("y_axis")

            # histograms & pies only need one axis → reuse x_axis
            if "histogram" in final_chart_type.lower():
                y_axis = x_axis
            # everything else still requires both
            elif not x_axis or not y_axis:
                return jsonify({"error": "GPT JSON missing x_axis or y_axis."}), 500

            # now safe to plot
            plot_path = model_utils.generate_final_plot(df, x_axis, y_axis, final_chart_type)
            used_cols_str = f"X-axis: {x_axis}, Y-axis: {y_axis}"
            subset_df = df[[x_axis, y_axis]].copy()
            subset_summary = subset_df.describe().to_string()


            # --- TAILORED MESSAGE SECTION ---
        ctl = (final_chart_type or "").lower()
        if "histogram" in ctl:
            msg = f"Here is your Histogram of {x_axis} showing its frequency distribution."
        elif "pie" in ctl:
            msg = f"Here is your Pie Chart showing how the parts of {x_axis} make up the whole."
        elif "line" in ctl:
            msg = f"Here is your Line Chart of {y_axis} over {x_axis} to illustrate the trend."
        elif "scatter" in ctl:
            msg = f"Here is your Scatter Plot comparing {y_axis} against {x_axis}."
        elif "treemap" in ctl:
            msg = f"Here is your Treemap of {y_axis} by {x_axis}."
        elif "map" in ctl:
            msg = f"Here is your Map of {y_axis} across {x_axis}."
        elif "linked" in ctl:
            msg = f"Here is your Linked Graph connecting each {x_axis} to {y_axis}."
        elif "parallel" in ctl:
            feat_list = ", ".join(feature_cols)
            msg = f"Here is your Parallel Coordinates chart comparing {feat_list} grouped by {class_col}."
        else:
            msg = f"Here is your {final_chart_type} using {used_cols_str}."
        # --- END TAILORED MESSAGE SECTION ---
        # --- END TAILORED MESSAGE SECTION ---

        explanation_prompt = f"""
You are a seasoned business intelligence analyst interpreting a '{final_chart_type}' 
chart. The user specifically chose these columns: {used_cols_str}.

Below is the summary of just those columns:
{subset_summary}

In 2–3 sentences, state the most important insights that will impact business decisions. 
Use direct, confident language (e.g., "The plot shows..."), and do not mention columns not shown here.
        """

        plot_description = gpt_gateway.handle_chat(explanation_prompt)

        base_url = "https://cautious-space-train-wrgx7wx5j7g6fv6xr-5000.app.github.dev"
        cache_buster = int(time.time()) 
        plot_url = f"{base_url}/plot.png?cb={cache_buster}"

        return jsonify({
            "message": msg,
            "plot_description": plot_description.strip(),
            "plot_url": plot_url,
            "plot_title": f"{final_chart_type} Visualization",
            "ask_reuse": "Would you like to visualize something else using this same dataset, purpose, and audience preferences? (Yes/No)"
        })
    



    except Exception as e:
        return jsonify({
            "error": f"Final‑plot error: {str(e)}",
            "ask_restart": "Would you like to try again? (Yes/No)"
        }), 200
    
@app.route("/reuse-dataset", methods=["POST"])
def reuse_dataset():
    """
    Provides the user with a list of dataset columns to select new visualization axes,
    using the same dataset, purpose, and audience preferences.
    """
    if not global_data.get("dataset_path"):
        return jsonify({"error": "No dataset found. Please upload a dataset first."}), 400
    try:
        return get_dataset_columns()
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve dataset columns: {str(e)}"}), 500


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
            response = gpt_gateway.handle_chat(greeting_prompt)
        except Exception as e:
            print(f"❌ GPT error: {e}")
            response = (
                "Hello! I’m VisioBot. Currently I’m unable to use GPT, "
                "but please upload your dataset, specify the purpose, and specify your audience."
            )
        return jsonify({"response": response})

    if "upload" in msg_lower or "dataset" in msg_lower:
        return jsonify({"response": "Please upload your dataset to begin."})
    elif "purpose" in msg_lower:
        return jsonify({"response": "What is the purpose of your visualization? (Distribution, Relationship, Comparison, Trends)"})
    elif "audience" in msg_lower:
        return jsonify({"response": "What’s your skill level (or your audience’s): Expert or Non‑Expert?"})
    elif "done" in msg_lower or "generate" in msg_lower:
        return jsonify({"response": "Processing your data... Generating a visualization now."})

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
            "Please follow the proper input process for best results. If you want to proceed, "
            "upload your dataset, specify purpose, and specify audience. Click on the restart button if an error presists."
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