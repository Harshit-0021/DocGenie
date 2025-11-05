# app.py
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import time

# import your existing logic from src / main as needed
from src.predict import predict_disease
from src.response_generator import generate_response
from main import query_medical_model  # assumes query_medical_model is importable

# Twilio config (do NOT commit these publicly)
ACCOUNT_SID = "AC880f411cfee2c483da544992c41449e5"
AUTH_TOKEN  = "2c8569461f48d6764071e673cc9fabea"
FROM_NUMBER = "+14155238886"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

app = Flask(__name__)

# ---- Secret-code configuration ----
SECRET_CODE = "blue-silent"   # set your secret string here (lowercase recommended)
MAX_REQUESTS_PER_MINUTE = 10  # simple rate-limit per phone number
BLOCKLIST = set()  # phone numbers to ignore

# in-memory simple counters (for demo only)
LAST_RESET = time.time()
REQUEST_COUNTS = {}  # phone -> count

def rate_limit_check(from_number):
    """Very simple in-memory rate limiter (demo)."""
    global LAST_RESET, REQUEST_COUNTS
    now = time.time()
    if now - LAST_RESET > 60:
        REQUEST_COUNTS = {}
        LAST_RESET = now
    cnt = REQUEST_COUNTS.get(from_number, 0) + 1
    REQUEST_COUNTS[from_number] = cnt
    return cnt <= MAX_REQUESTS_PER_MINUTE

@app.route("/webhook", methods=["POST"])
def webhook():
    # Twilio sends form-encoded POST by default for incoming messages
    incoming_body = (request.form.get("Body") or "").strip()
    from_number = request.form.get("From")  # e.g. "whatsapp:+91XXXXXXXXXX"

    # Quick validations
    if not incoming_body or not from_number:
        return Response("Missing data", status=200)

    # Blocklist check
    if from_number in BLOCKLIST:
        return Response("OK", status=200)  # silently ignore

    # Rate limiting
    if not rate_limit_check(from_number):
        resp = MessagingResponse()
        resp.message("You are sending too many requests. Try again later.")
        return Response(str(resp), mimetype="application/xml")

    # Normalize and check for secret code (case-insensitive, token must be present)
    # This checks if the exact code appears anywhere in the message body
    if SECRET_CODE.lower() in incoming_body.lower():
        # Extract symptoms: remove the code from message (user can send "letmein123: fever, cough")
        # Remove first occurrence of code (case-insensitive)
        import re
        cleaned = re.sub(re.escape(SECRET_CODE), "", incoming_body, flags=re.IGNORECASE).strip()
        # If user sent only the code, we can ask for symptoms
        if not cleaned:
            # Ask the user to send symptoms after the code
            resp = MessagingResponse()
            resp.message("Send your symptoms after the code, e.g. `letmein123 fever, cough`")
            return Response(str(resp), mimetype="application/xml")

        # Turn cleaned text into symptoms list
        symptoms = [s.strip().lower() for s in cleaned.split(",") if s.strip()]

        # Run your existing pipeline
        predicted_disease = predict_disease(symptoms)
        if "no" in predicted_disease.lower():
            response_struct = generate_response([], symptoms)
        else:
            response_struct = generate_response([predicted_disease], symptoms)

        # Query your LLM / OpenRouter (this calls your existing function)
        try:
            doctor_reply = query_medical_model(response_struct)
        except Exception as e:
            # Fallback message if LLM/API fails
            doctor_reply = "Sorry, the AI doctor is temporarily unavailable. Try again later."

        # Reply via Twilio (use TwiML so Twilio returns message in webhook)
        resp = MessagingResponse()
        resp.message(doctor_reply)
        return Response(str(resp), mimetype="application/xml")

    else:
        # Secret code not present — politely refuse
        resp = MessagingResponse()
        resp.message("To access the AI Doctor, include the secret code with your symptoms.")
        return Response(str(resp), mimetype="application/xml")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
