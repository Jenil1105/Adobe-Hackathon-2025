# Use full Python image with build tools and wheel compatibility
FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["python", "app/main.py"]
