import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.manager.path_manager import path_manager
from src.manager.file_manager import file_manager

def load_metrics(file_path):
    """Load metrics from a given JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def extract_metric_values(raw_metrics, metric_name):
    """Extract values for a specific metric from raw metrics (normalized and denormalized)."""
    values = []
    for entry in raw_metrics:
        for metric_set in entry:
            if metric_name in metric_set and not np.isnan(metric_set[metric_name]):
                values.append(metric_set[metric_name])
    return values

def aggregate_metric(metrics_list):
    """Compute average, stddev, best, and worst for a given metric."""
    if not metrics_list:
        return None, None, None, None
    avg = np.mean(metrics_list)
    stddev = np.std(metrics_list)
    best = np.min(metrics_list)
    worst = np.max(metrics_list)
    return avg, stddev, best, worst

def process_experiments(dataset_name, output_dir, experiment_dir):
    """
    Aggregate evaluation metrics and generate confidence band plots per model type.

    Args:
        dataset_name (str): Name of the dataset.
        output_dir (str): Directory to save aggregated results.
        experiment_dir (str): Directory containing experiments.
    """
    metrics_of_interest = ["mae", "mse", "rmse", "mape", "smape", "wape", "corr", "nrmse", "rrse"]
    dataset_path = Path(experiment_dir) / dataset_name
    aggregated_results = {metric: [] for metric in metrics_of_interest}

    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir():
            continue

        eval_results_path = model_dir / "evaluation_results/overall_metrics_summary.json"
        if not eval_results_path.exists():
            continue

        metrics_data = load_metrics(eval_results_path)
        model_name = model_dir.name.split("_")[0]  # Extract only model type

        for metric_name in metrics_of_interest:
            values_norm = extract_metric_values(metrics_data.get("raw_metrics_norm", []), metric_name)

            avg_norm, stddev_norm, best_norm, worst_norm = aggregate_metric(values_norm)

            aggregated_results[metric_name].append({
                "Model Type": model_name,
                "Train Split": float(model_dir.name.split("_")[4]) * 100,  # Convert to percentage
                f"{metric_name.upper()} Avg": avg_norm,
                f"{metric_name.upper()} StdDev": stddev_norm,
                f"{metric_name.upper()} Best": best_norm,
                f"{metric_name.upper()} Worst": worst_norm,
            })

    # Convert results to DataFrame and generate plots
    for metric, results in aggregated_results.items():
        results_df = pd.DataFrame(results)
        save_results(results_df, output_dir, metric)
        plot_confidence_band_per_model_type(results_df, output_dir, metric)

def save_results(results_df, output_dir, metric_name):
    """Save results to CSV and LaTeX format."""
    csv_output_path = Path(output_dir) / f"{metric_name}_results.csv"
    results_df.to_csv(csv_output_path, index=False)

    latex_output_path = Path(output_dir) / f"{metric_name}_results.tex"
    with open(latex_output_path, "w") as f:
        f.write(results_df.to_latex(index=False, float_format="%.4f"))

    print(f"Results for {metric_name} saved as CSV and LaTeX.")

def plot_confidence_band_per_model_type(results_df, output_dir, metric_name):
    """Generate confidence band plots per model type."""
    model_types = results_df["Model Type"].unique()

    for model in model_types:
        model_data = results_df[results_df["Model Type"] == model]

        plt.figure(figsize=(14, 8))
        #plt.ylim(0.0, 0.5)
        plt.plot(model_data["Train Split"], model_data[f"{metric_name.upper()} Avg"], marker="o", label=f"{model} Avg")
        plt.fill_between(
            model_data["Train Split"],
            model_data[f"{metric_name.upper()} Avg"] - model_data[f"{metric_name.upper()} StdDev"],
            model_data[f"{metric_name.upper()} Avg"] + model_data[f"{metric_name.upper()} StdDev"],
            alpha=0.2,
            label=f"{model} StdDev"
        )

        plt.title(f"{metric_name.upper()} Confidence Interval for {model}")
        plt.xlabel("Train Split (%)")
        plt.ylabel(metric_name.upper())
        plt.legend()
        plt.grid(True)

        plot_output_path = Path(output_dir) / f"{metric_name}_confidence_band_{model}.png"
        plt.savefig(plot_output_path, bbox_inches="tight")
        plt.close()
        print(f"Confidence Band plot saved: {plot_output_path}")


def main():
    dataset_config = {
        "dataset_name": "weatherstation",
        "ts_past": 168,
        "ts_future": 24,
    }

    output_dir = path_manager.get_path("result_dir") / f"{dataset_config["dataset_name"]}_{dataset_config["ts_past"]}_{dataset_config["ts_future"]}" / "aggregate_from_raw"
    experiment_dir = path_manager.get_path("experiment_dir")

    process_experiments(
        dataset_name=dataset_config["dataset_name"],
 #       ts_past=dataset_config["ts_past"],
 #       ts_future=dataset_config["ts_future"],
        output_dir=output_dir,
        experiment_dir=experiment_dir,
    )

if __name__ == "__main__":
    main()
