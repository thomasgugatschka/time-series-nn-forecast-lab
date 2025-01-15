import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.manager.path_manager import path_manager

def load_metrics(file_path):
    """Load metrics from a given JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


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


def plot_metrics_by_train_split(condensed_df, output_path, dataset_name):
    """Generate and save plots for metrics by train split."""
    metrics = ["normalized_avg_mse", "normalized_avg_mae", "denormalized_avg_mse", "denormalized_avg_mae"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for model_type in condensed_df["model_type"].unique():
            model_data = condensed_df[condensed_df["model_type"] == model_type]
            plt.plot(
                model_data["train_split"],
                model_data[metric],
                marker="o",
                label=model_type
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

    # Combined plot with color-coded models
    plt.figure(figsize=(12, 8))

    for model_type in condensed_df["model_type"].unique():
        model_data = condensed_df[condensed_df["model_type"] == model_type]
        plt.plot(
            model_data["train_split"],
            model_data["normalized_avg_mse"],
            marker="o",
            label=f"{model_type} (MSE)"
        )
        plt.plot(
            model_data["train_split"],
            model_data["normalized_avg_mae"],
            marker="x",
            linestyle="--",
            label=f"{model_type} (MAE)"
        )

    plt.title(f"Metrics Comparison by Train Split ({dataset_name})")
    plt.xlabel("Train Split")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    combined_plot_path = output_path / "metrics_comparison_by_train_split.png"
    plt.savefig(combined_plot_path)
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

        if not eval_results_path.exists():
            print(f"Skipping {model_dir.name}, evaluation file not found.")
            continue

        metrics_data = load_metrics(eval_results_path)

        # Expanded results: Include all metrics
        expanded_results.append({
            **metadata,
            "normalized": metrics_data.get("normalized", {}),
            "denormalized": metrics_data.get("denormalized", {}),
        })

        # Condensed results: Main metrics (MSE, MAE, etc.)
        condensed_results.append({
            "model_type": metadata["model_type"],
            "train_split": metadata["train_split"],
            "val_split": metadata["val_split"],
            "test_split": metadata["test_split"],
            "normalized_avg_mse": metrics_data["normalized"]["average"]["mse"],
            "normalized_avg_mae": metrics_data["normalized"]["average"]["mae"],
            "denormalized_avg_mse": metrics_data["denormalized"]["average"]["mse"],
            "denormalized_avg_mae": metrics_data["denormalized"]["average"]["mae"],
        })

    # Save results to files
    output_path = Path(output_dir) / f"{dataset_name}_{ts_past}_{ts_future}"
    output_path.mkdir(parents=True, exist_ok=True)

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


    # Generate plots
    plot_metrics_by_train_split(condensed_df, output_path, dataset_name)


    print(f"Aggregated results saved to {output_path}")

def main():

    # TODO -> REFACTOR -> Change so this works with argparse()
    #       -> As all other scripts use a config file as input
    #       -> Parse from the config the dataset and then the current ts_past and ts_future
    #       -> Have other inputs as well to input the dataset_name and ts_past and ts_future


    # TODO -> REFACTOR -> Add special parsing for Transformer (full-sequence / fixed-window)

    dataset_config = {
        "dataset_name": "exchangerate",
        "ts_past": 30,
        "ts_future": 7,
    }

    output_dir = path_manager.get_path("base_dir") / "results"
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
