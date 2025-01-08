# Base Image
FROM python:3.12-slim

# Create and set the working directory
WORKDIR /app

# Install system dependencies (e.g., gcc, make, etc., for Python packages requiring compilation)
# Install system dependencies (e.g., gcc, curl) and ensure Python3 is linked correctly
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libpq-dev \
    && ln -sf /usr/local/bin/python3 /usr/bin/python3 \
    && ln -sf /usr/local/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*



# Install Google Cloud SDK (for gsutil)
RUN curl -sSL https://sdk.cloud.google.com | bash > /dev/null && \
    /root/google-cloud-sdk/install.sh --quiet && \
    /root/google-cloud-sdk/bin/gcloud components install gsutil --quiet
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

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

# Environment variables for syncing and local mode
ENV SYNC_CONFIG_FROM_GCS=True
ENV LOCAL_MODE=False

#ENV PLOTS_SHOW=False
#ENV PLOTS_SAVE=False

# Define the ENTRYPOINT to include the sync logic
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

# CMD will be executed after ENTRYPOINT
CMD ["python", "-m", "src.scripts.hyperparameter_tuning"]