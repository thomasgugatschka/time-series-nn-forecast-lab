import json
from collections import defaultdict


def format_value(val):
    """
    Format a numeric value to have at most 5 digits (excluding the comma)
    with a fixed-point notation and replace the decimal point with a comma.

    - If |val| >= 1: let n = number of digits in the integer part, then use (5 - n) decimals.
    - If |val| < 1: use 4 decimals.
    """
    try:
        if not isinstance(val, (int, float)):
            return str(val)
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= 1:
            int_digits = len(str(int(abs_val)))
            decimals = max(0, 5 - int_digits)
            s = f"{val:.{decimals}f}"
        else:
            # For numbers less than 1, always use 4 decimals.
            s = f"{val:.4f}"
        # Replace the decimal point with a comma.
        #s = s.replace(".", ",")
        return s
    except Exception:
        return str(val)


def generate_table(model_label, entries, metric_source, dataset_name):
    """
    Generates a LaTeX longtable for the specified model group.
    Excludes the metrics 'msle' and 'rmsle' and formats numeric values using format_value.
    """
    # Sort entries by train_split.
    entries.sort(key=lambda x: x['train_split'])
    splits = [f"{int(x['train_split'] * 100)}\\%" for x in entries]

    # Use the keys from the chosen metric_source -> "average" as base metrics, excluding msle and rmsle.
    base_metrics = list(entries[0][metric_source]['average'].keys())
    base_metrics = [m for m in base_metrics if m not in ('msle', 'rmsle')]
    categories = ['average', 'stddev', 'best', 'worst']

    n_cols = 2 + len(splits)
    header = (
            r"{\small" "\n"
            r"\begin{longtable}{" + "l l " + " ".join(["c"] * len(splits)) + r"}" "\n"
                                                                             r"    \toprule" "\n"
                                                                             r"    \textbf{Modell} & \textbf{Metrik} & " + " & ".join(
        [f"\\textbf{{{s}}}" for s in splits]) + r" \\" "\n"
                                                r"    \midrule" "\n"
                                                r"    \endfirsthead" "\n"
                                                r"    \multicolumn{" + str(
        n_cols) + r"}{c}{\textit{Fortsetzung von vorheriger Seite}} \\" "\n"
                  r"    \toprule" "\n"
                  r"    \textbf{Modell} & \textbf{Metrik} & " + " & ".join(
        [f"\\textbf{{{s}}}" for s in splits]) + r" \\" "\n"
                                                r"    \midrule" "\n"
                                                r"    \endhead" "\n"
                                                r"    \midrule" "\n"
                                                r"    \multicolumn{" + str(
        n_cols) + r"}{r}{\textit{Fortsetzung auf der nächsten Seite}} \\" "\n"
                  r"    \midrule" "\n"
                  r"    \endfoot" "\n"
                  r"    \bottomrule" "\n"
                  r"    \caption{" + dataset_name + r" -- " + metric_source.capitalize() + r" Metriken für " + model_label + r"}" "\n"
                                                                                                                             r"    \label{tab:" + dataset_name.lower() + "_" + metric_source.lower() + "_" + model_label.lower().replace(
        ' ', '_') + r"}" "\n"
                    r"    \endlastfoot" "\n"
    )

    rows = ""
    for metric in base_metrics:
        for idx, cat in enumerate(categories):
            vals = [format_value(x[metric_source][cat][metric]) for x in entries]
            metric_with_suffix = f"{metric.upper()}\\textsubscript{{{cat}}}"
            if idx == 0:
                row = r"\multirow{" + str(
                    len(categories)) + r"}{*}{" + model_label + r"}" + " & " + metric_with_suffix + " & " + " & ".join(
                    vals) + r" \\" + "\n"
            else:
                row = " & " + metric_with_suffix + " & " + " & ".join(vals) + r" \\" + "\n"
            rows += row
        rows += r"\cmidrule(lr){2-" + str(n_cols) + r"}" + "\n"

    footer = r"\end{longtable}" + "\n}"
    return header + rows + footer


def main():
    # Define the dataset name and file path.
    resultname = "weatherstation_168_24"
    path = "C:\\Users\\Thomas\\localgit\\time-series-nn-forecast-lab\\results\\" + resultname + "\\expanded_results.json"

    # Load JSON data with UTF-8 encoding.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process both metric sources.
    metric_sources = ["normalized", "denormalized"]

    # Group entries by model type.
    models = defaultdict(list)
    for item in data:
        models[item['model_type'].lower()].append(item)

    for metric_source in metric_sources:
        for mtype, entries in models.items():
            if mtype == "transformer":
                # For Transformer, group by time_series_window.
                transformer_groups = defaultdict(list)
                for item in entries:
                    window = item.get("time_series_window", "").lower()
                    transformer_groups[window].append(item)
                for window, group in transformer_groups.items():
                    model_label = "Transformer-FS" if window == "full-sequence" else "Transformer-FW"
                    table = generate_table(model_label, group, metric_source, resultname)
                    outfile = f"combined_table_{resultname}_{metric_source}_{model_label.replace('-', '_')}.tex"
                    with open(outfile, "w", encoding="utf-8") as f_out:
                        f_out.write(table)
                    print(f"Wrote {outfile}")
            else:
                model_label = mtype.upper()
                table = generate_table(model_label, entries, metric_source, resultname)
                outfile = f"combined_table_{resultname}_{metric_source}_{model_label.lower()}.tex"
                with open(outfile, "w", encoding="utf-8") as f_out:
                    f_out.write(table)
                print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
