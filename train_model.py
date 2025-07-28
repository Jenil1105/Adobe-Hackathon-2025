import os
import joblib
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from generate_data import generate_dataset
import shutil

def train():
    # Multilingual MiniLM
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    X, y_head, y_level = generate_dataset(2000)
    print(f"Total samples: {len(X)}, headings: {sum(y_head)}")


    # Build feature matrix
    feats = []
    for sample in tqdm(X, desc="Extracting..."):
        # Build structural features from sample keys:
        struct = [
            sample['size_ratio'],
            sample['position_x'],
            sample['position_y'],
            sample['centered'],
            sample['length'],
            sample['case_ratio'],
            sample['digit_ratio'],
            sample['punct_present'],
            sample['prev_line_distance'],
            sample['next_line_size_ratio'],
        ]
        # Get embedding
        emb = embedder.encode([sample['text']])[0]
        feats.append(struct + emb.tolist())
    X_mat = np.array(feats)
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_mat)

    # Head vs Body
    rf_head = RandomForestClassifier(n_estimators=100, max_depth=15,
                                     class_weight='balanced', random_state=42)
    rf_head.fit(X_pca, y_head)

    # Level classifier
    idx = np.where(y_head==1)[0]
    rf_lvl = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_lvl.fit(X_pca[idx], np.array(y_level)[idx])

    # Save
    os.makedirs('app/models', exist_ok=True)
    embedder.save('app/models/minilm')
    joblib.dump(pca, 'app/models/pca.pkl')
    joblib.dump(rf_head, 'app/models/classifier_head.pkl')
    joblib.dump(rf_lvl, 'app/models/classifier_level.pkl')
    print("Training complete!")

if __name__ == '__main__':
    train()

