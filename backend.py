# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from logic import analyze_resume

app = Flask(__name__)
CORS(app)  # allow frontend to talk to backend

@app.route("/", methods=["GET"])
def home():
    return {"status": "CareerLens backend running"}

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Get job description
        job_desc = request.form.get("job_desc")

        # Get resume file
        file = request.files.get("resume")

        if not file or not job_desc:
            return jsonify({"error": "Resume file or job description missing"}), 400

        file_bytes = file.read()

        # Call logic.py
        result = analyze_resume(file_bytes, job_desc)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
