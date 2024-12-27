# Base Image
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/project_root"
ENV PROJECT_BASE_DIR="/workspace/project_root"

# Create and set the working directory
WORKDIR /workspace/project_root

# Copy all files from the project into the container
COPY . /workspace/project_root

# Define a volume for experiments
VOLUME /workspace/project_root/experiments

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt


# Expose necessary ports (if applicable)
EXPOSE 8000

# Default command to run the hyperparameter tuning script
CMD ["python", "scripts/hyperparameter_tuning.py", "--config_path", "config.yaml"]
