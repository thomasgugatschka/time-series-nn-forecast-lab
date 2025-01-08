#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print each command before execution (useful for debugging)

echo "Starting container..."

# Check if SYNC_CONFIG_FROM_GCS is True
if [ "$SYNC_CONFIG_FROM_GCS" = "True" ]; then
    echo "SYNC_CONFIG_FROM_GCS is set to True. Syncing configurations from GCS..."

    # Authenticate with the service account
    if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "Authenticating service account with $GOOGLE_APPLICATION_CREDENTIALS..."
        gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
    else
        echo "Error: Service account key file $GOOGLE_APPLICATION_CREDENTIALS not found!"
        exit 1
    fi

    # Sync configurations from GCS
    BUCKET_NAME="forecast-lab-experiments"
    CONFIGS_PATH="/app/configs"
    GCS_CONFIGS_PATH="gs://$BUCKET_NAME/configs"

    echo "Checking connectivity to GCS bucket..."
    if gsutil ls "$GCS_CONFIGS_PATH"; then
        echo "Syncing from $GCS_CONFIGS_PATH to $CONFIGS_PATH..."
        gsutil -m rsync -r "$GCS_CONFIGS_PATH" "$CONFIGS_PATH" || {
            echo "Error: GCS sync failed!"
            exit 1
        }
        echo "Configurations synced successfully."
    else
        echo "Error: Unable to access GCS bucket at $GCS_CONFIGS_PATH"
        exit 1
    fi
else
    echo "SYNC_CONFIG_FROM_GCS is not set or False. Skipping configuration sync."
fi

# Debugging: List the files in the configs directory
echo "Listing contents of $CONFIGS_PATH..."
ls -R "$CONFIGS_PATH" || {
    echo "Error: Unable to list contents of $CONFIGS_PATH"
    exit 1
}

# Run the main task
echo "Executing main task:"
for arg in "$@"; do
    echo -n "$arg "
done
echo

exec "$@"
