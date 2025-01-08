import os
import subprocess
import sys
from pathlib import Path

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

def run_script(script, config, optional_params=None):
    """
    Run a Python script with a single configuration file and optional parameters.

    :param script: The Python module to execute.
    :param config: Configuration file path.
    :param optional_params: Additional command-line parameters as a list.
    """
    project_root = Path(__file__).resolve().parent.parent
    venv_activate = project_root / ".venv" / "Scripts" / "activate"

    # Construct the command to activate the virtual environment and run the script
    command = [
        f'call {venv_activate} && python -m {script} --config_path "{config}"'
    ]
    if optional_params:
        command.extend(optional_params)

    print(f"Running: {' '.join(command)}")
    subprocess.run(" ".join(command), shell=True, check=True)

def main():
    # Define the operation manually (tuning, training, or evaluation)
    operation = "tuning"  # tuning, training, evaluation
    location = "file" # file, script



    if location == "file":
        config_list = read_config_file("config_file_list.txt")
    elif location == "script":
        # Define configurations directly in the script (optional)
        config_list = [
            "experiment_configs/ETTh/config_rnn_ETTh_0.4-0.1-0.1.yaml",
        ]
    else:
        print(f"Invalid location: {location}. Choose from file or script.")
        sys.exit(1)


    # Map operations to scripts
    operation_scripts = {
        "tuning": "src.scripts.hyperparameter_tuning",
        "training": "src.scripts.train_models",
        "evaluation": "src.scripts.evaluate_models",
    }

    if operation not in operation_scripts:
        print(f"Invalid operation: {operation}. Choose from tuning, training, or evaluation.")
        sys.exit(1)

    script_to_run = operation_scripts[operation]



    # Add optional parameter for training
    optional_params = []
    if operation == "training":
        optional_params.append("--num_models 5")

    # Run the selected script for each configuration
    for config in config_list:
        run_script(script_to_run, config, optional_params)

if __name__ == "__main__":
    main()
