# PDF Outline Extractor

This project extracts hierarchical outlines (headings/subheadings) from PDF files using a machine learning approach with `sentence-transformers` and `scikit-learn`. It's designed to run inside a lightweight, CPU-only Docker environment.

## 🧠 Approach

### 1. Problem Statement
Given a PDF file, extract the document’s outline — identifying sections and sub-sections based on semantic similarity and layout features.

### 2. Solution Overview
- **Text Extraction**: Using `pdfplumber` to extract text line-by-line.
- **Embeddings**: Each line is embedded using `sentence-transformers` (MiniLM model).
- **Dimensionality Reduction**: `PCA` is used to reduce embedding dimensionality.
- **Clustering & Classification**: Headings are classified using a trained `scikit-learn` model.
- **Output**: A structured outline is generated and saved in JSON format.

## 🗂 Project Structure

pdf-outline-extractor/
├── app/
│ ├── main.py
│ ├── classify.py
│ ├── utils.py
├── models/
│ ├── heading_classifier.pkl
│ └── all-MiniLM-L6-v2/
├── requirements.txt
└── Dockerfile

shell
Copy
Edit

## 🐳 Dockerized Setup (CPU Only)

### 1. Build the Docker Image

```bash
docker build -t pdf-outline-extractor .
2. Run the Container
bash
Copy
Edit
docker run --rm ^
  -v ${PWD}\app\input:/app/input ^
  -v ${PWD}\app\output:/app/output ^
  --network none ^
  pdf-outline-extractor
Replace ^ with \ or \ with / depending on your OS (PowerShell vs Bash).

📦 Dependencies
ini
Copy
Edit
pdfplumber==0.10.3
scikit-learn==1.2.2
sentence-transformers==2.2.2
numpy==1.26.2
joblib==1.2.0
reportlab==4.0.4
faker==18.11.2
pandas==1.5.3
tqdm==4.65.0
All dependencies are CPU-only. No CUDA or GPU acceleration is used.

🚫 Known Issues & Fixes
CUDA downloading?

Fixed by pinning sentence-transformers==2.2.2 and using compatible transformers and torch versions.

PCA input mismatch?

Ensure the same MiniLM model is used during training and inference.

Docker cache size?

Run cleanup commands below.

🔁 Clean-Up Commands
Remove All Docker Containers
bash
Copy
Edit
docker rm -f $(docker ps -aq)
Remove All Docker Images
bash
Copy
Edit
docker rmi -f $(docker images -aq)
Remove All Docker Volumes
bash
Copy
Edit
docker volume rm $(docker volume ls -q)
Clear Docker Build Cache
bash
Copy
Edit
docker builder prune --all --force
📤 Output Format
Output JSON in /app/output/:

json
Copy
Edit
[
  {
    "title": "1. Introduction",
    "children": [
      { "title": "1.1 Background" },
      { "title": "1.2 Objective" }
    ]
  },
  {
    "title": "2. Methodology",
    "children": []
  }
]
🤖 Model Details
Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Classifier: Trained using scikit-learn PCA + Logistic Regression

Runs fully on CPU

💻 Requirements
Docker

At least 4GB RAM

OS: Linux / Windows / macOS (with Docker Desktop)

👥 Credits
Built for Adobe Hackathon 2025

Developed by [Your Team Name]

vbnet
Copy
Edit

Let me know if you'd like this saved to a `.md` file or bundled with your Docker project!
