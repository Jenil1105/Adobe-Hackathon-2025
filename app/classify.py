import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models once
pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
clf_head = joblib.load(os.path.join(MODEL_DIR, "classifier_head.pkl"))
clf_level = joblib.load(os.path.join(MODEL_DIR, "classifier_level.pkl"))

# Force local-only loading
minilm_model = SentenceTransformer(
    os.path.join(MODEL_DIR, "minilm"),
    cache_folder=os.path.join(MODEL_DIR, "minilm"),
    local_files_only=True  # ✅ Key to block HuggingFaceHub access
)

def extract_features(line):
    # You can improve this later
    return [
        line["font_size"],
        int(line["is_bold"]),
        line["x0"],
        line["y0"],
        line["x1"],
        line["y1"],
        len(line["text"])
    ]

def predict_headings(lines):
    headings = []

    # Get embeddings for all lines at once
    texts = [line["text"] for line in lines]
    embeddings = minilm_model.encode(texts)

    # Apply PCA
    reduced = pca.transform(embeddings)

    # Combine with manual features
    manual_features = np.array([extract_features(line) for line in lines])
    combined_features = np.hstack([manual_features, reduced])

    # Predict heading yes/no
    is_heading_preds = clf_head.predict(combined_features)

    for idx, is_heading in enumerate(is_heading_preds):
        if is_heading:
            line = lines[idx]

            # Predict level
            level_pred = clf_level.predict([combined_features[idx]])[0]

            headings.append({
                "level": f"H{level_pred}",
                "text": line["text"],
                "page": line["page"]
            })

    return headings
