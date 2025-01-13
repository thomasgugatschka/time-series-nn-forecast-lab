

# TODO -> Write Script to aggregate results from after model evaluation.
# Input: Dataset Configuration file -> dataset name & ts_past, ts_future
# Output: Aggregated results for each model type and splits variations
# Output Type: JSON, CSV, MD, LaTeX
# Output Location: experiments/{dataset_name}_aggregated_results/*.*
# Output Content: Two Versions of Aggregated Results one Expanded and one Condensed
    # Expanded: All Metrics for all Models for all Splits and Normalized and Denormalized Metrics
    # Condensed: Main Metrics (MSE, ...) Stddev, Best, Worst of Metrics for all Models for all Splits