# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from src.predict import predict_disease
from src.response_generator import generate_response
from main import query_medical_model
import os

# ---------------------------------------------------
# Flask app configuration
# ---------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # ✅ Fixes CORS (browser → Flask connection)

# ---------------------------------------------------
# Route to serve your frontend (index.html)
# ---------------------------------------------------
@app.route('/')
def serve_index():
    # Serves the index.html file in the same folder
    return send_from_directory('templates', 'index.html')


# ---------------------------------------------------
# API endpoint to handle frontend requests
# ---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Parse input JSON from frontend
        data = request.get_json()
        symptoms_input = data.get("symptoms", "")
        symptoms = [s.strip().lower() for s in symptoms_input.split(",") if s.strip()]

        print(f"🩺 Received symptoms: {symptoms}")

        # Step 1 — Predict disease
        predicted_disease = predict_disease(symptoms)

        # Step 2 — Generate response based on prediction
        if "no" in predicted_disease.lower():
            print("⚠️ No disease predicted — generating response anyway.")
            response = generate_response([], symptoms)
        else:
            print(f"✅ Predicted Disease: {predicted_disease}")
            response = generate_response([predicted_disease], symptoms)

        # Step 3 — Get doctor-style reply using OpenRouter model
        print("🧠 Sending prompt to OpenRouter...")
        doctor_reply = query_medical_model(response)
        doctor_reply = (
            doctor_reply.replace("<s>", "")
            .replace("</s>", "")
            .replace("[BOT]", "")
            .strip()
        )

        print("🩻 Doctor Reply (cleaned):", doctor_reply[:150])

        # Step 4 — Return clean JSON to frontend
        return jsonify({
            "predicted_disease": predicted_disease,
            "doctor_reply": doctor_reply
        })

    except Exception as e:
        print(f"❌ Error in /predict route: {e}")
        return jsonify({
            "error": str(e),
            "predicted_disease": None,
            "doctor_reply": "An error occurred while processing your request."
        }), 500


# ---------------------------------------------------
# Run Flask app
# ---------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
