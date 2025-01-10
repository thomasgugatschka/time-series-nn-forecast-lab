@echo off
:: gcs_sync_experiments.bat - Sync experiments from GCS to local

:: Import common functions and variables
call "%~dp0gcp\gcp_common.bat"

:: Perform the sync using gsutil
echo Syncing gs://%GCS_BUCKET_NAME%/%GCS_EXPERIMENT_PREFIX% to %LOCAL_EXPERIMENT_PATH%
gsutil -m rsync -r "gs://%GCS_BUCKET_NAME%/%GCS_EXPERIMENT_PREFIX%" "%LOCAL_EXPERIMENT_PATH%"

:: Completion message
if %errorlevel% neq 0 (
    echo Sync failed.
    exit /b 1
) else (
    echo Sync completed successfully.
)
