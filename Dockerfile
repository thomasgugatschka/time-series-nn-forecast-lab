# Base Image
FROM python:3.12-slim

# Create and set the working directory
WORKDIR /app

# Install system dependencies (e.g., gcc, make, etc., for Python packages requiring compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service account key
COPY service-account-key.json /app/service-account-key.json

# Copy project files
COPY . .

# Set environment variables for Python and your project
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
ENV PROJECT_BASE_DIR="/app"

# Set environment variables for GCS
ENV USE_GCS=True
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json

# Default command to run the hyperparameter tuning script
#CMD ["python", "-m", "src.scripts.hyperparameter_tuning", "--config_path", "config_rnn_sinusoidal_0.8-0.1.yaml"]
CMD ["python", "-m", "src.scripts.hyperparameter_tuning"]