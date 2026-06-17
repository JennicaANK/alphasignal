# AlphaSignal Dockerfile
# Containerizes the FastAPI backend and Streamlit frontend

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed reports outputs

# Expose both ports
EXPOSE 8000 8501

# Default command runs the FastAPI backend
# Streamlit is started separately
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]