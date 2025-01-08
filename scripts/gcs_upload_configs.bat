@echo off
:: Set GCS bucket name
set BUCKET_NAME=forecast-lab-experiments


:: Get the directory of the batch file
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\


:: Set the local config folder path
set LOCAL_CONFIG_PATH=%PROJECT_ROOT%\configs



:: Ensure gsutil is installed and available in PATH
where gsutil >nul 2>nul
if %errorlevel% neq 0 (
    echo gsutil not found. Please ensure the Google Cloud SDK is installed and gsutil is in your PATH.
    exit /b 1
)

:: Sync local configs to GCS bucket
echo Uploading local configurations from %LOCAL_CONFIG_PATH% to gs://%BUCKET_NAME%/configs...
gsutil -m rsync -r %LOCAL_CONFIG_PATH% gs://%BUCKET_NAME%/configs

:: Completion message
if %errorlevel% neq 0 (
    echo Upload failed. Please check the logs and try again.
    exit /b 1
) else (
    echo Upload completed successfully.
)
