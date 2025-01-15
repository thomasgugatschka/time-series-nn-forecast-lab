import yaml
import os
from itertools import product


def generate_configurations(base_template, dataset_name, target_root):
    # Define model types and data split variations
    model_types = ["RNN", "LSTM", "GRU", "Transformer", "TCN", "CNN"]
    data_splits = [round(i * 0.1, 1) for i in range(8, 0, -1)]  # 0.8 to 0.1 in steps of 0.1

    # Model-specific configurations
    model_configs = {
        "RNN": {"cpu_per_trial": 8, "max_concurrent_trials": 2},
        "LSTM": {"cpu_per_trial": 8, "max_concurrent_trials": 2},
        "GRU": {"cpu_per_trial": 8, "max_concurrent_trials": 2},
        "Transformer": {"cpu_per_trial": 16, "max_concurrent_trials": 1, "time_series_window": ["fixed-window", "full-sequence"]},
        "TCN": {"cpu_per_trial": 16, "max_concurrent_trials": 1},
        "CNN": {"cpu_per_trial": 8, "max_concurrent_trials": 2},
    }

    # Ensure target directory exists
    dataset_dir = os.path.join(target_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Create a list to store generated filenames
    generated_files = []

    for model_type, frac_train in product(model_types, data_splits):
        # Create a deep copy of the base template
        config = base_template.copy()

        # Update model type and dataset
        config["model_type"] = model_type
        config["dataset"] = dataset_name

        # Apply model-specific overrides
        model_specific_config = model_configs.get(model_type, {})
        for key, value in model_specific_config.items():
            if key == "time_series_window":
                # Skip setting time_series_window here; it will be handled below
                continue
            config[key] = value

        # Update data split
        config["data_params"]["data_split"] = {
            "frac_train": frac_train,
            "frac_val": 0.1,
            "frac_test": 0.1,
            "unused_split": "beginning"
        }

        # Handle time_series_window specifically for Transformer
        time_series_windows = model_specific_config.get("time_series_window", [config["data_params"]["time_series_window"]])

        # Generate filenames for each window type if applicable
        for window_type in time_series_windows:
            config["data_params"]["time_series_window"] = window_type

            # Generate filename
            split_label = f"{frac_train}-0.1-0.1"
            filename = f"config_{model_type.lower()}_{dataset_name}_{split_label}_{window_type}.yaml" \
                if len(time_series_windows) > 1 else f"config_{model_type.lower()}_{dataset_name}_{split_label}.yaml"

            filepath = os.path.join(dataset_dir, filename)

            # Save YAML file
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Generated: {filepath}")

            # Append to the list of generated files
            generated_files.append(filename)

    # Save the list of generated files in the dataset directory
    index_file = os.path.join(dataset_dir, "generated_files_index.txt")
    with open(index_file, "w") as f:
        f.write("\n".join(generated_files))
    print(f"Generated file index: {index_file}")


def main():
    # Base YAML template
    base_template = {
        "model_type": "",  # Placeholder, will be set dynamically
        "dataset": "",  # Placeholder, will be set dynamically
        "num_samples": 50,
        "cpu_per_trial": 8,  # Default value
        "gpu_per_trial": 0,
        "max_concurrent_trials": 2,  # Default value
        "trial_params": {
            "max_epochs": 100,
            "grace_period": 5,
            "reduction_factor": 2,
        },
        "data_params": {
            "data_split": {
                "frac_train": "",
                "frac_val": "",
                "frac_test": "",
                "unused_split": "beginning",
            },
            "time_series_window": "fixed-window",  # Placeholder for Transformer
        },
        "search_algorithm": "optuna",
        "scheduler": "asha",
    }

    # Datasets to process
    datasets = ["weatherstation"]  # Add your datasets here
    target_root = "../configs/experiment_configs"

    for dataset_name in datasets:
        print(f"Generating configurations for dataset: {dataset_name}")
        generate_configurations(base_template, dataset_name, target_root)


if __name__ == "__main__":
    main()
