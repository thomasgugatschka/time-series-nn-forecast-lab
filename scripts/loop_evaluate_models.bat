@echo off

:: Activate the virtual environment
call C:\Users\Thomas\localgit\time-series-nn-forecast-lab\.venv\Scripts\activate

:: Get the directory of the batch file
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\

:: Set PYTHONPATH to the project root directory
set PYTHONPATH=%PROJECT_ROOT%

:: Use the directory of the batch file to locate config_list.txt
for /f "tokens=*" %%c in ('type "%~dp0config_list.txt"') do (
    echo Running training with config: %%c
    python -m src.scripts.evaluate_models --config_path "%%c"
)

:: Deactivate the virtual environment after completion
deactivate
