# Use a slim Python 3.11 base for sentence-transformers >=5.0.0 compatibility
FROM python:3.11-slim

# Prevent Python writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system build deps for PyMuPDF, scikit-learn, etc.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
#   - app/ contains main.py, utils.py, classify.py, input/ & output/ folders
#   - models/ contains your .pkl files and the local MiniLM folder
COPY app/    ./        # app/main.py → /app/main.py, app/utils.py → /app/utils.py, etc.

# Default entrypoint: process everything under /app/input → /app/output
CMD ["python", "main.py"]
