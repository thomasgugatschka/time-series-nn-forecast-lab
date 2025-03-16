import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#ds = "weatherstation_168_24"
#ds = "exchangerate_30_7"
#ds = "ETTh_96_24"
ds = "illness_96_24"

# Load JSON file
file_path =  "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\" + ds + "\\expanded_results.json"

csv_output_path = "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\" + ds + "\\formatted\\experiment_results.csv"
latex_output_path = "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\" + ds + "\\formatted\\experiment_results.tex"
plot_path =  "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\" + ds + "\\formatted\\"

with open(file_path, "r") as f:
    data = json.load(f)

# Select relevant metrics
metrics = ["mae", "mse", "rmse", "mape"]
table_data = []

# Process data
for entry in data:
    model = entry["model_identifier"]
    train_split = f"{int(entry['train_split'] * 100)}%"  # Convert train_split to percentage format

    row = {
        "Model": model.replace("-full-sequence", " FS").replace("-fixed-window", " FW")
        if "Transformer" in model
        else model.replace("-full-sequence", "").replace("-fixed-window", ""),
        "Train Split": train_split
    }

    for metric in metrics:
        if metric in entry["normalized"]["average"]:
            row[f"{metric.upper()} Avg"] = entry["normalized"]["average"].get(metric, np.nan)
            row[f"{metric.upper()} StdDev"] = entry["normalized"]["stddev"].get(metric, np.nan)
            row[f"{metric.upper()} Best"] = entry["normalized"]["best"].get(metric, np.nan)
            row[f"{metric.upper()} Worst"] = entry["normalized"]["worst"].get(metric, np.nan)

    table_data.append(row)

# Convert to DataFrame
df = pd.DataFrame(table_data)

# Save to CSV

df.to_csv(csv_output_path, index=False)
print(f"CSV saved to: {csv_output_path}")

# Generate LaTeX Table
latex_table = df.to_latex(index=False, float_format="%.4f", column_format="|l|c|c|c|c|c|c|c|c|",
                          caption="Experiment Results: Model Performance", label="tab:experiment_results")

with open(latex_output_path, "w") as f:
    f.write(latex_table)
print(f"LaTeX table saved to: {latex_output_path}")

# Plot metrics across models and train splits
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        plt.plot(model_data["Train Split"], model_data[f"{metric.upper()} Avg"], marker="o", label=model)

    plt.title(f"{metric.upper()} by Train Split")
    plt.xlabel("Train Split")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plot_output_path = f"{plot_path}/{metric}_by_train_split.png"
    plt.savefig(plot_output_path)
    print(f"Plot saved: {plot_output_path}")

# NEW PLOTS: Standard Deviation Across Train Splits
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        plt.plot(model_data["Train Split"], model_data[f"{metric.upper()} StdDev"], marker="o", linestyle="dashed",
                 label=model)

    plt.title(f"Standard Deviation of {metric.upper()} by Train Split")
    plt.xlabel("Train Split")
    plt.ylabel(f"{metric.upper()} StdDev")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plot_output_path = f"{plot_path}/{metric}_stddev_by_train_split.png"
    plt.savefig(plot_output_path)
    print(f"Plot saved: {plot_output_path}")

# NEW PLOTS: Best vs. Worst Performance
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        plt.plot(model_data["Train Split"], model_data[f"{metric.upper()} Best"], marker="o", linestyle="-",
                 label=f"{model} Best")
        plt.plot(model_data["Train Split"], model_data[f"{metric.upper()} Worst"], marker="x", linestyle="--",
                 label=f"{model} Worst")

    plt.title(f"Best vs. Worst {metric.upper()} by Train Split")
    plt.xlabel("Train Split")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plot_output_path = f"{plot_path}/{metric}_best_worst_by_train_split.png"
    plt.savefig(plot_output_path)
    print(f"Plot saved: {plot_output_path}")

# Plot 1: Line Plot with Error Bars (Std Dev)
for metric in metrics:
    plt.figure(figsize=(12, 6))
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        plt.errorbar(
            model_data["Train Split"], model_data[f"{metric.upper()} Avg"],
            yerr=model_data[f"{metric.upper()} StdDev"], fmt="o-", capsize=5, label=model
        )

    plt.title(f"Mean {metric.upper()} with Standard Deviation")
    plt.xlabel("Train Split")
    plt.ylabel(metric.upper())
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)

    plot_output_path = f"{plot_path}/{metric}_errorbar_plot.png"
    plt.savefig(plot_output_path, bbox_inches="tight")
    print(f"Error Bar plot saved: {plot_output_path}")

# Plot 2: Line Plot with Confidence Bands (Fill Between)
for metric in metrics:
    plt.figure(figsize=(12, 6))
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        plt.plot(model_data["Train Split"], model_data[f"{metric.upper()} Avg"], marker="o", label=model)
        plt.fill_between(
            model_data["Train Split"],
            model_data[f"{metric.upper()} Avg"] - model_data[f"{metric.upper()} StdDev"],
            model_data[f"{metric.upper()} Avg"] + model_data[f"{metric.upper()} StdDev"],
            alpha=0.2
        )

    plt.title(f"{metric.upper()} with Confidence Interval")
    plt.xlabel("Train Split")
    plt.ylabel(metric.upper())
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)

    plot_output_path = f"{plot_path}/{metric}_confidence_band.png"
    plt.savefig(plot_output_path, bbox_inches="tight")
    print(f"Confidence Band plot saved: {plot_output_path}")

# Plot 2: Grouped Bar Chart for Best vs. Worst Performance
#for metric in metrics:
#    plt.figure(figsize=(14, 7))
#    df_melted = df.melt(id_vars=["Model", "Train Split"],
#                        value_vars=[f"{metric.upper()} Best", f"{metric.upper()} Worst"],
#                        var_name="Performance Type", value_name=metric.upper())

#    sns.barplot(data=df_melted, x="Train Split", y=metric.upper(), hue="Performance Type", errorbar=None)
#    plt.title(f"Best vs. Worst {metric.upper()} by Train Split")
#    plt.xlabel("Train Split")
#    plt.ylabel(metric.upper())
#    plt.xticks(rotation=45)
#    plt.legend(title="Performance", bbox_to_anchor=(1.05, 1), loc="upper left")
#    plt.grid(True)

#    plot_output_path = f"{plot_path}/{metric}_best_worst_barchart.png"
#    plt.savefig(plot_output_path, bbox_inches="tight")
#    print(f"Bar chart saved: {plot_output_path}")

# Plot 3: Violin Plot for Best vs. Worst Performance
#for metric in metrics:
#    plt.figure(figsize=(14, 7))
#    df_melted = df.melt(id_vars=["Model", "Train Split"],
#                        value_vars=[f"{metric.upper()} Best", f"{metric.upper()} Worst"],
#                        var_name="Performance Type", value_name=metric.upper())

#    sns.violinplot(data=df_melted, x="Train Split", y=metric.upper(), hue="Performance Type", split=True,
#                   inner="quartile")
#    plt.title(f"Best vs. Worst {metric.upper()} - Violin Plot")
#    plt.xlabel("Train Split")
#    plt.ylabel(metric.upper())
#    plt.xticks(rotation=45)
#    plt.legend(title="Performance", bbox_to_anchor=(1.05, 1), loc="upper left")
#    plt.grid(True)

#    plot_output_path = f"{plot_path}/{metric}_violin_plot.png"
#    plt.savefig(plot_output_path, bbox_inches="tight")
#    print(f"Violin plot saved: {plot_output_path}")

# Plot 3: Heatmap of Model Performance
for metric in metrics:
    plt.figure(figsize=(12, 6))
    pivot_table = df.pivot(index="Model", columns="Train Split", values=f"{metric.upper()} Avg")

    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5)
    plt.title(f"Heatmap of {metric.upper()} Across Models and Train Splits")
    plt.xlabel("Train Split")
    plt.ylabel("Model")

    plot_output_path = f"{plot_path}/{metric}_heatmap.png"
    plt.savefig(plot_output_path, bbox_inches="tight")
    print(f"Heatmap saved: {plot_output_path}")

# Generate Confidence Interval Plots for Each Model Separately
for metric in metrics:
    for model in df["Model"].unique():
        plt.figure(figsize=(12, 6))
        model_data = df[df["Model"] == model]

        # Compute upper and lower bounds for Confidence Interval
        ci_upper = model_data[f"{metric.upper()} Avg"] + model_data[f"{metric.upper()} StdDev"]
        ci_lower = model_data[f"{metric.upper()} Avg"] - model_data[f"{metric.upper()} StdDev"]

        # Plot Mean Line
        plt.plot(
            model_data["Train Split"],
            model_data[f"{metric.upper()} Avg"],
            marker="o",
            color="blue",
            label=f"Mean {metric.upper()}",
            linestyle="-"
        )

        # Fill area for Confidence Interval
        plt.fill_between(
            model_data["Train Split"],
            ci_lower,
            ci_upper,
            alpha=0.2,
            color="blue",
            label="Confidence Interval (±1 StdDev)"
        )

        # Plot Best and Worst values with independent points
    #    plt.scatter(
    #        model_data["Train Split"],
    #        model_data[f"{metric.upper()} Best"],
    #        marker="^",
    #        color="green",
    #        s=100,
    #        label=f"Best {metric.upper()}"
    #    )
    #    plt.scatter(
    #        model_data["Train Split"],
    #        model_data[f"{metric.upper()} Worst"],
    #        marker="v",
    #        color="red",
    #        s=100,
    #        label=f"Worst {metric.upper()}"
    #    )

        plt.title(f"{model} - {metric.upper()} with Confidence Interval")
        plt.xlabel("Train Split")
        plt.ylabel(metric.upper())
        plt.legend(title="Performance Indicators", loc="upper left")
        plt.grid(True)

        # Save the plot
        plot_output_path = f"{plot_path}/{model}_{metric}_confidence_plot.png"
        plt.savefig(plot_output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_output_path}")

plt.show()
