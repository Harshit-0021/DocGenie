import os
import pickle

# Base path (project root)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load pickle files safely
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# Load all models/encoders
def load_resources():
    disease_model = load_pickle("disease_model.pkl")
    label_encoder = load_pickle("label_encoder.pkl")
    mlb_encoder = load_pickle("mlb_encoder.pkl")
    return disease_model, label_encoder, mlb_encoder

# import pickle
# import pickle
#
# pkl_path = r"F:\Projects\Disease-prediction\models\disease_model.pkl"
#
# try:
#     with open(pkl_path, "rb") as f:
#         obj = pickle.load(f)
#     print("Loaded successfully!")
#     print("Type of model object:", type(obj))
# except Exception as e:
#     print("Could not load normally.")
#     print("Error:", e)

