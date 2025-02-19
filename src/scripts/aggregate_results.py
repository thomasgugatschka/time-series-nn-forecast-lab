import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.manager.path_manager import path_manager
from src.manager.file_manager import file_manager

def load_metrics(file_path):
    """Load metrics from a given JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def load_main_config(config_path):
    """Load the main configuration YAML file."""
    return file_manager.load_yaml(config_path)

def parse_experiment_folder_name(folder_name):
    """Parse experiment folder name to extract metadata."""
    parts = folder_name.split("_")
    model_type = parts[0]
    dataset_name = parts[1]
    ts_past = int(parts[2])
    ts_future = int(parts[3])
    train_split = float(parts[4])
    val_split = float(parts[5])
    test_split = float(parts[6])
    start_date = parts[7]
    return {
        "model_type": model_type,
        "dataset_name": dataset_name,
        "ts_past": ts_past,
        "ts_future": ts_future,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "start_date": start_date,
    }

def plot_all_metrics(condensed_df, output_path, dataset_name):
    """Generate and save plots for all metrics by train split."""
    metrics = [col for col in condensed_df.columns if col.startswith("normalized_") or col.startswith("denormalized_")]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for model in condensed_df["model_identifier"].unique():
            model_data = condensed_df[condensed_df["model_identifier"] == model]
            plt.plot(
                model_data["train_split"],
                model_data[metric],
                marker="o",
                label=model
            )

        plt.title(f"{metric.replace('_', ' ').title()} by Train Split ({dataset_name})")
        plt.xlabel("Train Split")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        metric_plot_path = output_path / f"{metric}_by_train_split.png"
        plt.savefig(metric_plot_path)
        plt.close()

def aggregate_metrics(dataset_name, ts_past, ts_future, output_dir, experiment_dir):
    """
    Aggregate evaluation metrics for a given dataset and save results.

    Args:
        dataset_name (str): Name of the dataset.
        ts_past (int): Number of past time steps.
        ts_future (int): Number of future time steps.
        output_dir (str): Directory to save aggregated results.
        experiment_dir (str): Directory containing experiments.
    """
    expanded_results = []
    condensed_results = []

    # Iterate through all experiment folders
    dataset_path = Path(experiment_dir) / dataset_name
    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir():
            continue

        try:
            metadata = parse_experiment_folder_name(model_dir.name)
        except (ValueError, IndexError):
            print(f"Skipping folder with invalid name format: {model_dir.name}")
            continue

        eval_results_path = model_dir / "evaluation_results/overall_metrics_summary.json"
        main_config_path = model_dir / "main_config.yaml"

        if not eval_results_path.exists() or not main_config_path.exists():
            print(f"Skipping {model_dir.name}, required files not found.")
            continue

        metrics_data = load_metrics(eval_results_path)
        main_config = load_main_config(main_config_path)

        # Identify Transformer-specific setting
        time_series_window = main_config.get("data_params", {}).get("time_series_window", "unknown")
        metadata["time_series_window"] = time_series_window

        # Create a unique identifier for the model
        model_identifier = f"{metadata['model_type']}-{time_series_window}"
        metadata["model_identifier"] = model_identifier

        # Expanded results: Include all metrics
        expanded_results.append({
            **metadata,
            "normalized": metrics_data.get("normalized", {}),
            "denormalized": metrics_data.get("denormalized", {}),
        })

        # Condensed results: Main metrics (MSE, MAE, etc.)
        condensed_results.append({
            "model_identifier": model_identifier,
            "train_split": metadata["train_split"],
            "val_split": metadata["val_split"],
            "test_split": metadata["test_split"],
            **{
                f"normalized_avg_{metric}": metrics_data["normalized"]["average"].get(metric, None)
                for metric in metrics_data["normalized"]["average"].keys()
            },
            **{
                f"denormalized_avg_{metric}": metrics_data["denormalized"]["average"].get(metric, None)
                for metric in metrics_data["denormalized"]["average"].keys()
            },
        })

    # Save results to files
    output_path = Path(output_dir) / f"{dataset_name}_{ts_past}_{ts_future}"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path_plots = output_path / "metrics_visualization"
    output_path_plots.mkdir(parents=True, exist_ok=True)

    # Save expanded results
    expanded_json = output_path / "expanded_results.json"
    with open(expanded_json, "w") as f:
        json.dump(expanded_results, f, indent=4)

    expanded_csv = output_path / "expanded_results.csv"
    pd.DataFrame(expanded_results).to_csv(expanded_csv, index=False)

    # Save condensed results
    condensed_json = output_path / "condensed_results.json"
    with open(condensed_json, "w") as f:
        json.dump(condensed_results, f, indent=4)

    condensed_csv = output_path / "condensed_results.csv"
    pd.DataFrame(condensed_results).to_csv(condensed_csv, index=False)

    # Convert condensed results to Markdown and LaTeX
    condensed_df = pd.DataFrame(condensed_results)

    condensed_md = output_path / "condensed_results.md"
    with open(condensed_md, "w") as f:
        f.write(f"# Aggregated Results for {dataset_name}\n\n")
        f.write(f"- Dataset Name: {dataset_name}\n")
        f.write(f"- ts_past: {ts_past}\n")
        f.write(f"- ts_future: {ts_future}\n\n")
        f.write(condensed_df.to_markdown(index=False))

    condensed_latex = output_path / "condensed_results.tex"
    with open(condensed_latex, "w") as f:
        f.write(f"\\section*{{Aggregated Results for {dataset_name}}}\n")
        f.write(f"\\begin{{itemize}}\n")
        f.write(f"  \\item Dataset Name: {dataset_name}\n")
        f.write(f"  \\item ts\\_past: {ts_past}\n")
        f.write(f"  \\item ts\\_future: {ts_future}\n")
        f.write(f"\\end{{itemize}}\n\n")
        f.write(condensed_df.to_latex(index=False, escape=False))

    # Generate plots for all metrics
    plot_all_metrics(condensed_df, output_path_plots, dataset_name)

    print(f"Aggregated results saved to {output_path}")

def main():

    # TODO -> IMPORTANT -> from LITERATURE derive the metrics for the "tex" and "md" files
    #       -> MAE, MSE, RMSE, MAPE ?


    # TODO -> REFACTOR -> Change so this works with argparse()
    #       -> As all other scripts use a config file as input
    #       -> Parse from the config the dataset and then the current ts_past and ts_future
    #       -> Have other inputs as well to input the dataset_name and ts_past and ts_future

    # TODO -> REFACTOR -> Modify to use PlotManager and FileManager

    # TODO -> REFACTOR -> Build Path to Plots in a more generic way -> use ExperimentManager or create a new ResultManager?

    dataset_config = {
        "dataset_name": "weatherstation",
        "ts_past": 168,
        "ts_future": 24,
    }

    output_dir = path_manager.get_path("result_dir")
    experiment_dir = path_manager.get_path("experiment_dir")

    aggregate_metrics(
        dataset_name=dataset_config["dataset_name"],
        ts_past=dataset_config["ts_past"],
        ts_future=dataset_config["ts_future"],
        output_dir=output_dir,
        experiment_dir=experiment_dir,
    )

if __name__ == "__main__":
    main()
