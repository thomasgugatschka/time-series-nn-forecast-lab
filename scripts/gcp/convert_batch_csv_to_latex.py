import pandas as pd

# Load the original CSV file again to ensure raw data
file_path = "/mnt/data/batch_jobs_report.csv"
df_original = pd.read_csv(file_path)


# Function to extract model and split correctly, handling Transformer-specific cases
def extract_model_split_v5(command):
    try:
        parts = command.split()
        model = "N/A"
        split = "N/A"

        # Look for the model type in the command arguments
        for i, part in enumerate(parts):
            if "--config_path" in part and i + 1 < len(parts):
                config_path = parts[i + 1]
                segments = config_path.split("/")
                if len(segments) > 1:
                    model_info = segments[-1].split("_")
                    model = model_info[1]  # Extract model type

                    # Special handling for Transformer models
                    if "transformer" in model.lower():
                        split_values = model_info[-1].replace(".yaml", "")  # Last segment
                        # Extract split ratios if present
                        if "-" in model_info[-2]:
                            split_values = f"{model_info[-2]}_{split_values}"
                        split = f"full-sequence ({split_values})"
                    else:
                        split = model_info[-1].replace(".yaml", "")

                break

        return model, split
    except Exception:
        return "N/A", "N/A"


# Apply the updated function to extract Model and Split
df_original[["Model", "Split"]] = df_original["Command"].apply(lambda x: pd.Series(extract_model_split_v5(str(x))))

# Select relevant columns
latex_df = df_original[["Run Duration (s)", "CPU (milli)", "Memory (MiB)", "Model", "Split"]]

# Rename columns for LaTeX output
latex_df.columns = ["Run Duration (s)", "vCPUs", "Memory (GB)", "Model", "Split"]

# Sort by Model and then by Split
latex_df = latex_df.sort_values(by=["Model", "Split"])

# Generate LaTeX table with separate Model and Split columns
latex_code = latex_df.to_latex(index=False, escape=True, column_format="|c|c|c|c|c|")

# Save LaTeX table to a file
latex_file_path = "/mnt/data/batch_jobs_report_final.tex"
with open(latex_file_path, "w") as latex_file:
    latex_file.write(latex_code)


# Return the file path for download
latex_file_path
