# Base Image
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
ENV PROJECT_BASE_DIR="/app"

# Create and set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies and Python dependencies
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Default command to run the hyperparameter tuning script
CMD ["python", "-m", "-scr.scripts.hyperparameter_tuning", "--config_path", "config.yaml"]
