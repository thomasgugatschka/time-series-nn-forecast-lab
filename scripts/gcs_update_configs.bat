@echo off
:: gcs_update_configs.bat - Update configs from local to GCS

:: Import common functions and variables
call "%~dp0gcp\gcp_common.bat"

:: Sync local configs to GCS bucket
echo Uploading local configurations from %LOCAL_CONFIG_PATH% to gs://%GCS_BUCKET_NAME%/configs...
gsutil -m rsync -r "%LOCAL_CONFIG_PATH%" "gs://%GCS_BUCKET_NAME%/%GCS_CONFIG_PREFIX%"

:: Completion message
if %errorlevel% neq 0 (
    echo Upload failed. Please check the logs and try again.
    exit /b 1
) else (
    echo Upload completed successfully.
)
