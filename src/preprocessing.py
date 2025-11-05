# preprocessing_and_prediction.py

import speech_recognition as sr
import spacy
import numpy as np

# Import your utils + predict functions
from src.utils import load_resources
from sklearn.exceptions import NotFittedError

# ----------------------------
# 1️⃣ Load spaCy model
# ----------------------------
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# 2️⃣ Load ML model + encoders once
# ----------------------------
disease_model, label_encoder, mlb_encoder = load_resources()

# ----------------------------
# 3️⃣ Voice-to-Text Function
# ----------------------------
def voice_to_text(audio_file_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

# ----------------------------
# 4️⃣ Text Cleaning & Symptom Extraction
# ----------------------------
def extract_symptoms(text):
    """
    Using spaCy for:
    - Lowercasing
    - Stopword removal
    - Lemmatization
    - NER for medical terms
    """
    doc = nlp(text.lower())

    # Collect nouns, adjectives, and verbs (common in symptoms)
    symptoms = [token.lemma_ for token in doc
                if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'ADJ', 'VERB']]

    # spaCy NER (detects entities like DISEASE, CONDITION)
    for ent in doc.ents:
        if ent.label_ in ['DISEASE', 'CONDITION']:
            symptoms.append(ent.text.lower())

    return list(set(symptoms))  # unique

# ----------------------------
# 5️⃣ Top-N Disease Prediction (from model)
# ----------------------------
def top_diseases_from_input(user_text, top_n=3):
    # Extract symptoms from text
    user_symptoms = extract_symptoms(user_text)

    if not user_symptoms:
        return []

    try:
        # Encode symptoms to multi-hot vector
        symptoms_encoded = mlb_encoder.transform([user_symptoms])

        # If model supports probabilities → get top_n
        if hasattr(disease_model, "predict_proba"):
            probs = disease_model.predict_proba(symptoms_encoded)[0]
            top_indices = np.argsort(probs)[::-1][:top_n]
            top_diseases = [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top_indices]
        else:
            # Fallback: just return single prediction
            prediction = disease_model.predict(symptoms_encoded)
            disease = label_encoder.inverse_transform(prediction)[0]
            top_diseases = [(disease, 1.0)]

        return top_diseases

    except NotFittedError:
        print("⚠️ Model is not fitted. Please retrain your model.")
        return []

