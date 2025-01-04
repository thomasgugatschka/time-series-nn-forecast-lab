@echo off

:: Activate the virtual environment
call C:\Users\Thomas\localgit\time-series-nn-forecast-lab\.venv\Scripts\activate

:: Get the directory of the batch file
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\

:: Set PYTHONPATH to the project root directory
set PYTHONPATH=%PROJECT_ROOT%



:: Config for RNN with ETTh dataset
set config_path=experiment_configs\config_gru_ETTh_0.7-0.15.yaml
:: set config_path=experiment_configs\config_gru_ETTm_0.8-0.1.yaml

:: Run the hyperparameter tuning script
python -m src.scripts.hyperparameter_tuning --config_path "%config_path%"
pause

:: Deactivate the virtual environment after completion
deactivate
