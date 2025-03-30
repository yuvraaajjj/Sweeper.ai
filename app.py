from flask import Flask, request, jsonify, send_file, render_template
import os
from werkzeug.utils import secure_filename
from main import DataState, app as workflow_app  # Import LangGraph workflow app

app = Flask(__name__)

# ✅ Set up file upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    """Render the frontend upload page."""
    return render_template("index.html")  # Ensure 'index.html' exists in the 'templates' folder

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload, processes it, and provides a cleaned CSV."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # ✅ Secure the filename and save
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # ✅ Run the LangGraph workflow correctly using `.run()`
    print(f"📂 Processing file: {file_path}")
    initial_state = DataState(file_path=file_path)
    result = workflow_app.run(initial_state)  # ✅ Use `run()`, not `invoke()`

    # ✅ Get cleaned file path
    cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")

    if not os.path.exists(cleaned_file_path):
        return jsonify({"error": "Processing failed, cleaned file not found."}), 500

    return jsonify({
        "message": "File processed successfully!",
        "cleaned_file_path": cleaned_file_path,
        "analysis_report": result["analysis_report"],
        "domain": result["domain"]
    })

@app.route("/download", methods=["GET"])
def download_file():
    """Allows users to download the cleaned CSV file."""
    file_path = request.args.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
