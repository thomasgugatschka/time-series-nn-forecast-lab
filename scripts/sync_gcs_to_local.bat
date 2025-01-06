@echo off
:: Set GCS bucket and local path
set BUCKET_NAME=forecast-lab-experiments
set GCS_PREFIX=experiments/
set LOCAL_SYNC_PATH=C:\Users\Thomas\localgit\time-series-nn-forecast-lab\experiments

:: Ensure gsutil is available
where gsutil >nul 2>nul
if %errorlevel% neq 0 (
    echo gsutil not found. Please ensure the Google Cloud SDK is installed and gsutil is in your PATH.
    exit /b 1
)

:: Perform the sync using gsutil
echo Syncing gs://%BUCKET_NAME%/%GCS_PREFIX% to %LOCAL_SYNC_PATH%
gsutil -m rsync -r "gs://%BUCKET_NAME%/%GCS_PREFIX%" "%LOCAL_SYNC_PATH%"

:: Completion message
if %errorlevel% neq 0 (
    echo Sync failed.
    exit /b 1
) else (
    echo Sync completed successfully.
)
