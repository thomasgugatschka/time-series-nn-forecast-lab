import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import plotly.express as px

from src.manager.config_manager import ConfigManager
from src.manager.data_manager import DataManager
from src.manager.experiment_manager import ExperimentManager
from src.manager.file_manager import file_manager
from src.manager.plot_manager import PlotManager
from src.utilities.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# TODO -> REFACTOR -> PlotManager should be used. Script could also be removed completely TensorBoard analysis is more convenient.
def parse_args():
    """Parse arguments for the script."""
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top configurations to include in the parallel plot.")
    return parser.parse_args()

def load_data(file_path):
    """Load JSON data and return as a Pandas DataFrame."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.json_normalize(data)
    return df

def filter_and_prepare(df):
    """Prepare the DataFrame by filtering relevant columns dynamically."""
    # Select metrics and hyperparameters dynamically
    metrics = ["val_loss", "train_loss"]
    hyperparameters = [col for col in df.columns if col.startswith("config/")]

    # Combine metrics and hyperparameters
    selected_columns = metrics + hyperparameters
    filtered_df = df[selected_columns]

    # Rename columns for clarity
    filtered_df.columns = [col.replace("config/", "").replace("/", "_") for col in filtered_df.columns]
    return filtered_df

def plot_loss_trends(df):
    """Plot validation and training loss trends over iterations."""
    if "training_iteration" in df.columns:
        trend_df = df.groupby("training_iteration")[["val_loss", "train_loss"]].mean()
        trend_df.plot(figsize=(10, 6))
        plt.title("Loss Trends Across Training Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(["Validation Loss", "Training Loss"])
        plt.grid(True)
        plt.show()

def cluster_and_visualize(df):
    """Perform clustering and visualize results for key combinations."""
    # Identify numerical columns for clustering
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    clustering_features = df[numerical_cols].dropna()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(clustering_features)

    # Pairplot for visualizing clusters
    sns.pairplot(df, vars=["val_loss", "train_loss"], hue="cluster", palette="viridis", height=2.5)
    plt.suptitle("Clustering of Validation and Training Loss", y=1.02)
    plt.show()

def plot_hyperparameter_vs_loss(df):
    """Create scatter plots for hyperparameters against validation loss."""
    hyperparameters = [col for col in df.columns if col not in ["val_loss", "train_loss", "cluster"]]

    for param in hyperparameters:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=param, y="val_loss", hue="cluster", palette="viridis")
        plt.title(f"{param} vs Validation Loss")
        plt.xlabel(param)
        plt.ylabel("Validation Loss")
        plt.grid(True)
        plt.show()

def parallel_coordinates_plot(df, top_n=10):
    """Create an interactive parallel coordinates plot for hyperparameters and metrics."""
    # Sort by validation loss and select top N rows
    df_sorted = df.sort_values("val_loss").head(top_n)

    # Ensure numerical encoding for all categorical data
    for col in df_sorted.columns:
        if df_sorted[col].dtype == 'object':
            df_sorted[col] = df_sorted[col].astype('category').cat.codes

    # Dynamically detect relevant columns for the plot
    dimensions = [
        col for col in df_sorted.columns if col not in ["val_loss", "train_loss"]
    ]

    fig = px.parallel_coordinates(
        df_sorted,
        dimensions=dimensions,
        color="val_loss",
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"Top {top_n} Hyperparameter Configurations"
    )
    fig.show()

def plot_3d_scatter(df):
    """Create a 3D scatter plot for hyperparameters and validation loss."""
    # Dynamically identify the required columns
    hidden_size_col = [col for col in df.columns if "hidden_size" in col][0]
    dropout_col = [col for col in df.columns if "dropout" in col][0]
    batch_size_col = [col for col in df.columns if "batch_size" in col][0]

    fig = px.scatter_3d(
        df,
        x=hidden_size_col,
        y=dropout_col,
        z="val_loss",
        color="val_loss",
        size=batch_size_col,
        title="3D Scatter Plot of Hyperparameters vs Validation Loss",
        labels={
            hidden_size_col: "Hidden Size",
            dropout_col: "Dropout",
            "val_loss": "Validation Loss",
            batch_size_col: "Batch Size"
        },
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.show()

def plot_heatmap(df, x_param, y_param):
    """Generate a heatmap showing the interaction between two hyperparameters and their effect on validation loss."""
    pivot_table = df.pivot_table(
        values="val_loss", index=y_param, columns=x_param, aggfunc="mean"
    )

    fig = px.imshow(
        pivot_table,
        title=f"Heatmap: {x_param} vs {y_param} (Validation Loss)",
        labels=dict(color="Validation Loss"),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_title=x_param,
        yaxis_title=y_param,
    )
    fig.show()

def plot_parallel_categories(df):
    """Create a parallel categories plot for hyperparameters and metrics."""
    # Dynamically select relevant columns for the plot
    dimensions = [
        "val_loss",  # Always include validation loss
        "train_loss",  # Always include training loss
    ] + [col for col in df.columns if col.startswith("model_params_") or col.startswith("training_params_")]

    # Ensure all categorical data is encoded numerically for Plotly
    for col in dimensions:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # Create the plot
    fig = px.parallel_categories(
        df,
        dimensions=dimensions,
        color="val_loss",
        color_continuous_scale="Viridis",
        title="Parallel Categories Plot: Hyperparameters and Validation Loss",
    )
    fig.show()

def plot_correlation_matrix(df):
    """Create an interactive correlation matrix including categorical encoding."""
    # Encode categorical columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes

    # Compute correlation matrix
    correlation_matrix = df.corr()

    # Create an interactive heatmap
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Correlation Matrix of Hyperparameters and Metrics",
        labels=dict(color="Correlation"),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(width=800, height=800)
    fig.show()

def plot_loss_trends_interactive(df):
    """Interactive line chart showing validation loss trends over iterations."""
    if "training_iteration" in df.columns:
        fig = px.line(
            df,
            x="training_iteration",
            y="val_loss",
            color="trial_id",
            title="Validation Loss Trends Across Iterations",
            labels={"val_loss": "Validation Loss", "training_iteration": "Iterations"}
        )
        fig.show()

def plot_hyperparameter_importance(df):
    """Estimate hyperparameter importance using Random Forest regression."""
    features = df.drop(columns=["val_loss", "train_loss"])
    target = df["val_loss"]

    # Fit a simple model
    model = RandomForestRegressor(random_state=42)
    model.fit(features, target)

    # Extract feature importance
    importance = pd.DataFrame({
        "Feature": features.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Plot importance
    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Hyperparameter Importance",
        labels={"Importance": "Feature Importance"}
    )
    fig.show()

def save_results(df, output_dir):
    """Save processed DataFrame and clusters for future analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "processed_results.csv", index=False)
    print(f"Processed results saved to {output_dir / 'processed_results.csv'}")


def main():
    args = parse_args()

    config_manager = ConfigManager(args.config_path)
#    data_manager = DataManager(config_manager)
    experiment_manager = ExperimentManager(config_manager)
    experiment_name, experiment_path, _ = experiment_manager.manage_experiment()
#    plot_manager = PlotManager(config_manager, data_manager, phase="plots_hyperparameters")


    # Load and Prepare data
    df = load_data(experiment_path / "all_hyperparameters.json")
    print("Data loaded successfully.")
    filtered_df = filter_and_prepare(df)
    print("Data prepared for analysis.")

    # Plot loss trends
    plot_loss_trends(filtered_df)

    # Perform clustering and visualize
    cluster_and_visualize(filtered_df)

    # Plot hyperparameters against validation loss
    plot_hyperparameter_vs_loss(filtered_df)

    # Interactive parallel coordinates plot
    parallel_coordinates_plot(filtered_df, top_n=args.top_n)

    # Call additional visualizations
    plot_3d_scatter(filtered_df)
    plot_heatmap(filtered_df, x_param="model_params_hidden_size", y_param="model_params_dropout")
    plot_parallel_categories(filtered_df)
    plot_correlation_matrix(filtered_df)
    plot_loss_trends_interactive(filtered_df)
    plot_hyperparameter_importance(filtered_df)

    #Save results
    save_results(filtered_df, experiment_path / "plots_hyperparameters")


if __name__ == "__main__":
    main()

