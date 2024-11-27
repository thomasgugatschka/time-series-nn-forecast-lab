# Known Limitations 

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

## Logging Final Metrics to TensorBoard

### Issue
Hyperparameters logged using add_hparams() in TensorBoard may not appear in the HPARAMS tab, though they show up in Scalars and Time Series. This affects metrics like final validation and training losses.

### Cause
This issue likely results from a bug or compatibility problem in the current versions of PyTorch or TensorFlow.

### Workaround
Log hyperparameters manually as scalars to ensure visibility in Scalars and Time Series.

