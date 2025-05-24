from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import json
from models.ocr import extract_text, process_with_llama
from models.grading import grade_answers
from models.chatbot import ask_llama  # Import chatbot function



app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
DATA_FOLDER = os.path.join(os.getcwd(), "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        raw_text = extract_text(image_path)
        structured_qa = process_with_llama(raw_text)

        json_path = os.path.join(DATA_FOLDER, "ocr_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_qa, f, indent=4)

        return render_template("ocr_results.html", results=structured_qa, json_file="ocr_results.json")
    
    except Exception as e:
        return jsonify({"error": f"OCR processing failed: {str(e)}"}), 500

@app.route("/upload_json", methods=["POST"])
def upload_json():
    file = request.files.get("json_file")
    if not file:
        return jsonify({"error": "No JSON file uploaded!"}), 400

    json_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            uploaded_data = json.load(f)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format!"}), 400

    try:
        graded_results = grade_answers(uploaded_data)
    except Exception as e:
        return jsonify({"error": f"Grading failed: {str(e)}"}), 500

    graded_results_path = os.path.join(DATA_FOLDER, "graded_results.json")
    with open(graded_results_path, "w", encoding="utf-8") as f:
        json.dump(graded_results, f, indent=4)

    return redirect("/results")

@app.route("/results")
def results():
    graded_results_path = os.path.join(DATA_FOLDER, "graded_results.json")
    if not os.path.exists(graded_results_path):
        return "Grading results not found. Please upload a JSON file first.", 404

    with open(graded_results_path, "r", encoding="utf-8") as f:
        graded_results = json.load(f)

    return render_template("results.html", results=graded_results)

@app.route("/download_results")
def download_results():
    results_path = os.path.join(DATA_FOLDER, "graded_results.json")
    return send_file(results_path, as_attachment=True, download_name="graded_results.json")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Handles user queries about grading results."""
    data = request.json
    user_query = data.get("message", "")

    if not user_query:
        return jsonify({"response": "Please enter a valid question!"})

    graded_results_path = os.path.join(DATA_FOLDER, "graded_results.json")
    if not os.path.exists(graded_results_path):
        return jsonify({"response": "Grading results not found. Please upload and grade an exam first."})

    with open(graded_results_path, "r", encoding="utf-8") as f:
        graded_results = json.load(f)

    # 🛠️ Make sure Llama always gets graded results
    response = ask_llama(user_query, graded_results)

    return jsonify({"response": response})



if __name__ == "__main__":
    app.run(debug=True)
