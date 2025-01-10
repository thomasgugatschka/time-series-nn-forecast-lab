@echo off
:: gcp\gcp_common.bat - Common functions and variables for GCP-related batch scripts

:: Shared Variables
set GCS_BUCKET_NAME=forecast-lab-experiments
set GCS_EXPERIMENT_PREFIX=experiments/
set GCS_CONFIG_PREFIX=configs/

:: Resolve local paths to the project root
pushd "%~dp0..\.."
set LOCAL_EXPERIMENT_PATH=%cd%\experiments
set LOCAL_CONFIG_PATH=%cd%\configs
popd

:: Ensure paths exist
if not exist "%LOCAL_EXPERIMENT_PATH%" (
    echo [Error] LOCAL_EXPERIMENT_PATH does not exist: %LOCAL_EXPERIMENT_PATH%
    exit /b 1
)

if not exist "%LOCAL_CONFIG_PATH%" (
    echo [Error] LOCAL_CONFIG_PATH does not exist: %LOCAL_CONFIG_PATH%
    exit /b 1
)

:: Function: Check if gsutil is available
where gsutil >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] gsutil not found. Please ensure the Google Cloud SDK is installed and gsutil is in your PATH.
    exit /b 1
)

:: echo LOCAL_EXPERIMENT_PATH=%LOCAL_EXPERIMENT_PATH%
:: echo LOCAL_CONFIG_PATH=%LOCAL_CONFIG_PATH%