
# main.py
import requests
from src.predict import predict_disease
from src.response_generator import generate_response

# -----------------------------
# 1️⃣ Hugging Face Inference Config
# -----------------------------
# API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"
# API_KEY = "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"  # your valid HF token



#
# HEADERS = {
#     "Authorization": f"Bearer {API_TOKEN}",
#     "Content-Type": "application/json"
# }


import requests

def query_medical_model(response):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = "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"  # get from openrouter.ai
    prompt = response["llm_prompt"]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",  # free open model
        "messages": [
            {"role": "system", "content": "You are a caring, professional doctor providing concise medical advice."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 250,
        "temperature": 0.7
    }

    try:
        res = requests.post(API_URL, headers=headers, json=payload)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
        # Clean model artifacts (like <s>, [BOT], etc.)
        reply = reply.replace("<s>", "").replace("</s>", "").replace("[BOT]", "").strip()
        return reply


    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return "Doctor service unavailable. Please try again later."


def main():
    print("🩺 Disease Prediction Bot")
    print("------------------------")

    # Step 1: Take user input symptoms
    symptoms_input = input("Enter symptoms separated by commas: ").strip()
    symptoms = [s.strip().lower() for s in symptoms_input.split(",") if s.strip()]

    if not symptoms:
        print("⚠️ No symptoms entered. Please try again.")
        return

    # Step 2: Predict disease
    predicted_disease = predict_disease(symptoms)

    # Step 3: Generate structured response
    if "no" in predicted_disease.lower():
        print("\n⚠️ No disease predicted. Providing advice based on symptoms...")
        response = generate_response([], symptoms)
    else:
        print(f"\n✅ Predicted Disease: {predicted_disease}")
        response = generate_response([predicted_disease], symptoms)

        # Step 4: Show disease info (if available)
        print("\n📋 Disease Information:")
        for idx, pred in enumerate(response["predictions"], start=1):
            print(f"🦠 Disease: {pred['disease'].title()}")
            print(f"ℹ️ Description: {pred['description'].capitalize()}")
            print("💊 Precautions:")
            for p_idx, p in enumerate(pred["precautions"], start=1):
                print(f"   {p_idx}. {p.capitalize()}")

    # Step 5: Get Doctor’s advice from Hugging Face LLM
    print("\n🤖 Contacting AI Doctor...\n")
    doctor_reply = query_medical_model(response)

    print("🩻 Doctor's Advice:")
    print(doctor_reply)


if __name__ == "__main__":
    main()
