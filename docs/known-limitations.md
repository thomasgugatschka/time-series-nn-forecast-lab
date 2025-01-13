# Known Limitations 


## Choosing the Best Epoch in Hyperparameter Tuning

When tuning with early stopping, you might see better validation performance at an earlier epoch than the stopping point. Should you use:

1. **The Early Stopping Epoch**:
   - Reflects operational settings where early stopping balances training and overfitting.
   - Aligns with the tuning process but may not capture the best validation performance.

2. **The Epoch with Lowest Validation Error**:
   - Maximizes generalization by using the best validation performance.
   - Requires tracking and saving the model at the best epoch during training.

### Recommendation
- **For Best Generalization**: Use the epoch with the lowest validation error.
- **For Real-World Usage**: Use the early stopping epoch.
- Save checkpoints for both and evaluate them to decide what works best for your goals.

### Current Implementation
- Training till the early stopping epoch.

## Logging Transformer Models with TensorBoard

### Overview
TensorBoard struggles with logging Transformer models due to their dynamic nature. This often causes errors like:

- **TracingCheckError**: PyTorch fails when tracing complex operations in Transformers, especially self-attention.
- **RuntimeError**: Issues arise from dynamic control flow, like variable sequence lengths.

### Workaround
To avoid these errors, we skip logging the graph for Transformer models:

```python
# tensorboard_callback.py
model_name = wrapper.model.__class__.__name__
if 'Transformer' not in model_name:
    self.writer.add_graph(self.model, self.example_input)
else:
    logger.warning(f"Skipping graph logging for {model_name} due to known tracing issues.")
```

---

## Logging Final Metrics to TensorBoard

### Issue
Hyperparameters logged using add_hparams() in TensorBoard may not appear in the HPARAMS tab, though they show up in Scalars and Time Series. This affects metrics like final validation and training losses.

### Cause
This issue likely results from a bug or compatibility problem in the current versions of PyTorch or TensorFlow.

### Workaround
Log hyperparameters manually as scalars to ensure visibility in Scalars and Time Series.

---

## GCP Batch Limitations
**Limited Parallelization**:
- GCP Batch currently starts only 1–3 jobs in parallel, regardless of the number of submitted jobs.
- No clear pattern or reason for this limitation has been identified. According to the documentation, this behavior depends on **resource availability** in the region.

**Non-Resumable Experiments**:
- Experiments are **not resumable** when using GCP Batch or similar Docker-based deployments.
- This is because experiment files (e.g., logs, metadata) are not pulled from GCS to the container during execution.

**TensorBoard Logging**:
- **Live monitoring via TensorBoard** is not supported, as logs are not streamed to GCS during training.

**Costs**:
- Using GCP Batch incurs costs for **Compute Engine**, **Cloud Storage**, and the **Artifact Repository**.

---




