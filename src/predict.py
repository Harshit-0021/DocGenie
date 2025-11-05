from src.utils import load_resources

# Load model + encoders once
disease_model, label_encoder, mlb_encoder = load_resources()

def predict_disease(symptoms: list) -> str:
    """
    Predict disease based on a list of symptoms.
    Example: predict_disease(["fever", "cough", "headache"])
    """

    if not symptoms:
        return "No symptoms provided."

    # Ensure only known symptoms are passed
    known_symptoms = set(mlb_encoder.classes_)
    valid_symptoms = [s for s in symptoms if s in known_symptoms]

    if not valid_symptoms:
        return "No valid symptoms provided."

    # Encode symptoms to multi-hot vector
    symptoms_encoded = mlb_encoder.transform([valid_symptoms])

    # Predict
    prediction = disease_model.predict(symptoms_encoded)

    # Decode back to disease name
    disease = label_encoder.inverse_transform([int(prediction[0])])[0]

    return disease
