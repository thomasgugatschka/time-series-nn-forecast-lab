import os
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime

TRACKING_FILE_TEMPLATE = "batch_tracking_{operation}_{timestamp}.json"

def read_config_file(config_list_file):
    """
    Read configuration file paths from a text file.

    :param config_list_file: Path to the file containing configuration file paths.
    :return: List of configuration file paths.
    """
    if not os.path.exists(config_list_file):
        print(f"Configuration list file not found: {config_list_file}")
        sys.exit(1)

    with open(config_list_file, "r") as file:
        return [line.strip() for line in file if line.strip()]

def update_tracking_file(config, script, status, tracking_file):
    """
    Update the tracking file with the execution status of a script and configuration.

    :param config: Configuration file path.
    :param script: Script name.
    :param status: Execution status (e.g., "success", "failure").
    :param tracking_file: The path to the tracking file.
    """
    if not os.path.exists(tracking_file):
        tracking_data = {}
    else:
        with open(tracking_file, "r") as file:
            tracking_data = json.load(file)

    if config not in tracking_data:
        tracking_data[config] = {}

    tracking_data[config][script] = status

    with open(tracking_file, "w") as file:
        json.dump(tracking_data, file, indent=4)

def run_script(script, config, optional_params=None, tracking_file=None):
    """
    Run a Python script with a single configuration file and optional parameters.

    :param script: The Python module to execute.
    :param config: Configuration file path.
    :param optional_params: Additional command-line parameters as a list.
    :param tracking_file: Path to the tracking file for logging the execution status.
    """
    project_root = Path(__file__).resolve().parent.parent
    venv_activate = project_root / ".venv" / "Scripts" / "activate"

    # Construct the command to activate the virtual environment and run the script
    command = [
        f'call {venv_activate} && python -m {script} --config_path "{config}"'
    ]
    if optional_params:
        command.extend(optional_params)

    try:
        print(f"Running: {' '.join(command)}")
        subprocess.run(" ".join(command), shell=True, check=True)
        if tracking_file:
            update_tracking_file(config, script, "success", tracking_file)
    except subprocess.CalledProcessError as e:
        print(f"Error while running script {script} with config {config}: {e}")
        if tracking_file:
            update_tracking_file(config, script, "failure", tracking_file)

def main():
    # Define the operation manually (tuning, training, or evaluation)

    # --- SETTINGS ---
    operation = "evaluation"  # tuning, training, evaluation, visualizeDS
    location = "file" # file, script
    prefix_file = "experiment_configs/illness/"
    # ----------------

    # Generate timestamped tracking file name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tracking_file = TRACKING_FILE_TEMPLATE.format(operation=operation, timestamp=timestamp)

    if location == "file":
        config_list = read_config_file("config_file_list.txt")
    elif location == "script":
        # Define configurations directly in the script (optional)
        config_list = [
#            "experiment_configs/ETTh/config_rnn_ETTh_0.4-0.1-0.1.yaml",
            "config.yaml"
        ]
    else:
        print(f"Invalid location: {location}. Choose from file or script.")
        sys.exit(1)

    # Map operations to scripts
    operation_scripts = {
        "tuning": "src.scripts.hyperparameter_tuning",
        "training": "src.scripts.train_models",
        "evaluation": "src.scripts.evaluate_models",
        "visualizeDS": "src.scripts.visualize_dataset"
    }

    if operation not in operation_scripts:
        print(f"Invalid operation: {operation}. Choose from tuning, training, or evaluation.")
        sys.exit(1)

    script_to_run = operation_scripts[operation]

    # Add optional parameter for training
    optional_params = []
    if operation == "training":
        optional_params.append("--num_models 2")

    # Run the selected script for each configuration
    for config in config_list:
        if location == "file":
            config = prefix_file + config
        run_script(script_to_run, config, optional_params, tracking_file=tracking_file)

if __name__ == "__main__":
    main()
