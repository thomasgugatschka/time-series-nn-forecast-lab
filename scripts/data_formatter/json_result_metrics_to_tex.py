import json

# Load the JSON data
input_file = "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\weatherstation_168_24\\condensed_results.json"
output_tex_file = "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\weatherstation_168_24\\formatted\\condensed_results_formatted.tex"

with open(input_file, "r") as f:
    data = json.load(f)

# Prepare LaTeX table header
latex_content = """
\\begin{longtable}{l c c c c c}
    \\toprule
    \\textbf{Model} & \\textbf{Train Split} & \\textbf{MAE} & \\textbf{MSE} & \\textbf{RMSE} & \\textbf{MAPE} \\\\
    \\midrule
    \\endfirsthead

    \\multicolumn{6}{c}{{\\textit{Continued from previous page}}} \\\\
    \\toprule
    \\textbf{Model} & \\textbf{Train Split} & \\textbf{MAE} & \\textbf{MSE} & \\textbf{RMSE} & \\textbf{MAPE} \\\\
    \\midrule
    \\endhead

    \\midrule \\multicolumn{6}{r}{{\\textit{Continued on next page}}} \\\\
    \\endfoot

    \\bottomrule
    \\caption{Experimental Results: Model Performance Metrics}
    \\label{tab:results_longtable}
    \\endlastfoot
"""

# Process data and format rows
for entry in data:
    model = entry["model_identifier"]

    if "transformer" in model.lower():
        model = model.replace("-full-sequence", "-FS").replace("-fixed-window", "-FW")
    else:
        model = model.replace("-full-sequence", "").replace("-fixed-window", "")


    train_split = f"{int(entry['train_split'] * 100)}\\%"  # Convert to percentage format
    mae = entry["normalized_avg_mae"]
    mse = entry["normalized_avg_mse"]
    rmse = entry["normalized_avg_rmse"]
    mape = entry["normalized_avg_mape"]

    latex_content += f"        {model} & {train_split} & {mae:.4f} & {mse:.4f} & {rmse:.4f} & {mape:.2f} \\\\\n"

# Close LaTeX longtable and place caption & label at the bottom
latex_content += """
\\end{longtable}
"""

# Save to a .tex file
with open(output_tex_file, "w") as f:
    f.write(latex_content)

print(f"LaTeX table successfully generated: {output_tex_file}")
