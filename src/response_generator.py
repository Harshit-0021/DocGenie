# src/response_generator.py

import pandas as pd
import os

# Load CSVs only once at module import
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
data_path = os.path.join(BASE_DIR, "data")

# Read CSVs and normalize everything to lowercase
description_df = pd.read_csv(os.path.join(data_path, "symptom_Description.csv"))
precaution_df = pd.read_csv(os.path.join(data_path, "symptom_precaution.csv"))

# Ensure Disease column is lowercase
description_df["Disease"] = description_df["Disease"].str.lower()
precaution_df["Disease"] = precaution_df["Disease"].str.lower()

# Ensure text columns are lowercase
if "Description" in description_df.columns:
    description_df["Description"] = description_df["Description"].str.lower()

for col in precaution_df.columns:
    if col != "Disease":
        precaution_df[col] = precaution_df[col].astype(str).str.lower()


def generate_response(diseases: list, user_symptoms: list):
    """
    Build response with description + precautions for predicted diseases.
    Also returns a natural prompt string that can be passed to an LLM.
    """
    # Normalize inputs
    diseases = [d.lower() for d in diseases]
    user_symptoms = [s.lower() for s in user_symptoms]

    response = {
        "user_symptoms": user_symptoms,
        "predictions": [],
        "llm_prompt": ""  # final string to pass to LLM
    }

    prompt_parts = [
        f"Consider yourself a professional doctor consultant.\n",
        f"The patient reported the following symptoms: {', '.join(user_symptoms)}.\n\n",

        " Important Instructions:\n"
        "- The symptoms provided are TRUE and should always be trusted.\n"
        "- The list of possible diseases and precautions below are machine-generated and CAN BE WRONG.\n"
        "- As a doctor, you must carefully analyze and decide what could be correct.\n"
        "- If a disease is too severe, clearly advise the patient to seek immediate physical medical support.\n"
        "- If precautions are missing, you must generate appropriate ones from your medical knowledge.\n"
        "- You should also feel free to ask follow-up questions to the patient for better diagnosis, just like a real doctor.\n\n",

        "Example Case:\n"
        "For example, if the patient has mild viral symptoms like fever or headache, the model might incorrectly suggest something serious like dengue. "
        "In that case, you must analyze thoroughly, explain possibilities, and guide the patient responsibly.\n\n",

        "Here are the possible diseases with details (remember: they may be wrong):\n"
    ]

    for idx, disease in enumerate(diseases, start=1):
        # Fetch description
        desc_row = description_df[description_df["Disease"].str.lower() == disease]
        description = desc_row["Description"].values[0] if not desc_row.empty else "no description available."

        # Fetch precautions
        prec_row = precaution_df[precaution_df["Disease"].str.lower() == disease]
        if not prec_row.empty:
            precautions = prec_row.drop("Disease", axis=1).values.flatten().tolist()
            precautions = [p for p in precautions if pd.notna(p)]
        else:
            precautions = ["no precautions available."]

        # Store in structured response
        response["predictions"].append({
            "disease": disease,
            "description": description,
            "precautions": precautions
        })

        # Add to LLM prompt
        prompt_parts.append(
            f"\n{idx}. disease: {disease}\n"
            f"   description: {description}\n"
            f"   precautions: {', '.join(precautions)}\n"
        )

    # Final instructions
    prompt_parts.append(
        "\nGive the final response point wise.\n"
        "\nNow, as a doctor consultant, write a natural and caring reply for the patient.\n"
        "- Explain the situation based on symptoms.\n"
        "- Mention possible conditions and their descriptions.\n"
        "- Give suitable precautions (use your medical knowledge if missing).\n"
        "- If condition seems critical, advise urgent physical consultation.\n\n"
        "🚫 Final Rule: You are replying directly to the patient. "
        "Do not mention this model, the data source, or that you are following a prompt. "
        "Just give a clear, professional doctor’s response.\n"
        "Give the reply as a chatbot under 150 words — concise, clear, and caring."
        "If you gets nan in any value simply ignore it, it's just a null value."

    )

    response["llm_prompt"] = "".join(prompt_parts)
    return response

