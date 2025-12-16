# fastapi_feature_attribution_local_time_series




Methods

1. gradinet, other : captum
2. LIMESegment
3. Instance-wise Feature Importance in Time (FIT)
4. Explaining Time Series Classifiers trhough Meaningful Perturbation and Optimisation
5. WindowSHAP
6. shats -- done
7. Timeshap (maybe)
8. TSR : temporal salinevy rescaling 
9. Explainable Time Series Anomaly Detection using Masked Latent Generative Modeling




-----

# ShaTS (Shapelet-based Time Series) Explainer

**ShaTS** is a feature attribution method designed specifically for time series data. It estimates the importance of specific time steps or channels by evaluating how the model's prediction changes when parts of the input are replaced with a background reference distribution (using Shapley values).

This implementation provides a standardized API wrapper (`ShatsExplainer`) compatible with the project's explainer interface.

## 1\. Quick Start

```python
from app.services.explainers.shats_impl import ShatsExplainer

# 1. Initialize Explainer
explainer = ShatsExplainer()

# 2. Configure Request
request = {
    "model": model_wrapper_func,       # Callable returning probabilities
    "data": [input_sample],            # List of samples to explain (Time, Channels)
    "background_data": background_set, # List of reference samples (Time, Channels)
    "params": {
        "grouping_strategy": "time",   # "time" or "feature"
        "implementation": "fast",      # "fast", "approx", "kernel", "cached"
        "m": 5                         # Granularity parameter
    }
}

# 3. Compute Explanations
response = explainer.explain(request)
shats_values = response["result"] 
# Result Shape: (N_samples, N_groups, N_classes)
```

## 2\. API Reference

### `ShatsExplainer.explain(request)`

Computes feature importance scores.

#### **Input: `request` (Dictionary)**

| Key | Type | Description |
| :--- | :--- | :--- |
| **`model`** | `Callable` | A wrapper function that takes a list/tensor of inputs `(Batch, Time, Channels)` and returns model probabilities `(Batch, Classes)`. |
| **`data`** | `List[np.array]` | A list containing the input samples to explain. <br>**Shape:** `(Time_Steps, Channels)` <br>*Note: If your model expects `(Channels, Time)`, transpose before passing here.* |
| **`background_data`** | `List[np.array]` | A list of samples representing the "background" distribution (e.g., random samples from the training/test set). Used to simulate "missing" features. <br>**Shape:** `(Time_Steps, Channels)` |
| **`params`** | `Dictionary` | Configuration options (see below). |

#### **Configuration: `params` (Dictionary)**

| Parameter | Options | Default | Description |
| :--- | :--- | :--- | :--- |
| **`grouping_strategy`** | `'time'` | `'time'` | **`'time'`**: Calculates importance for each **time step** (row).<br>**`'feature'`**: Calculates importance for each **channel/variable** (column).<br>**`'multifeature'`**: Custom grouping (requires `custom_groups`). |
| **`implementation`** | `'fast'`, `'approx'`, `'kernel'`, `'cached'` | `'fast'` | **`'fast'`**: Optimized implementation (Recommended).<br>**`'approx'`**: Standard approximation logic.<br>**`'kernel'`**: KernelSHAP-based estimation.<br>**`'cached'`**: Caches results for similar inputs (faster for sequential data). |
| **`m`** | `int` | `5` | **Granularity**: Controls the number of subsets generated for Shapley estimation. Higher values = more precise but slower. |
| **`batch_size`** | `int` | `32` | Number of perturbed samples sent to the model at once. |
| **`custom_groups`** | `List[List[int]]` | `None` | Required only if `grouping_strategy='multifeature'`. Defines which feature indices belong to which group. |

#### **Output: `response` (Dictionary)**

Returns a dictionary containing the results.

```python
{
    "result": [
        # List of explanations (one per input sample)
        [
            # Matrix of Importance Scores (Shape: Groups x Classes)
            [class_0_score, class_1_score, ...], # Group 1
            [class_0_score, class_1_score, ...], # Group 2
            ...
        ]
    ]
}
```

  * **Dimension 0**: Sample Index (corresponding to `data`).
  * **Dimension 1**: Group Index.
      * If `grouping_strategy='time'`, size = `Time_Steps`.
      * If `grouping_strategy='feature'`, size = `Channels`.
  * **Dimension 2**: Class Index.

## 3\. Data Shapes & Transposition

ShaTS assumes inputs are shaped **`(Time, Channels)`**.
Most PyTorch CNNs expect **`(Channels, Time)`**.

**You must handle transposition in two places:**

1.  **Before ShaTS**: Transpose your input `(C, T) -> (T, C)` before adding it to `request['data']`.
2.  **Inside Wrapper**: Transpose back `(T, C) -> (C, T)` inside your `model` wrapper function before feeding it to the CNN.

## 4\. Visualization Types

The API does not return plots directly; it returns raw scores. You can visualize them based on the strategy used:

| Strategy | Output Shape | Recommended Plot | Meaning |
| :--- | :--- | :--- | :--- |
| **`time`** | `(Time, Classes)` | **Heatmap** (1D) over time | Shows **when** the important event occurred. |
| **`feature`** | `(Channels, Classes)` | **Bar Chart** per channel | Shows **which signal** (sensor) contributed most. |

## 5\. Example Usage

```python
# Assuming 'model' expects (Batch, Channels, Time)
# And 'sample' is (Channels, Time)

# 1. Define Model Wrapper (Handles Transpose back to C, T)
def model_wrapper(x_numpy):
    tensor_in = torch.tensor(x_numpy).float().to(device)
    # Permute: (Batch, Time, Channels) -> (Batch, Channels, Time)
    tensor_in = tensor_in.permute(0, 2, 1) 
    return model(tensor_in).cpu().numpy()

# 2. Prepare Data (Transpose to T, C)
input_data = sample.T 
bg_data = [x.T for x in background_samples]

# 3. Call ShaTS
explainer = ShatsExplainer()
request = {
    "model": model_wrapper,
    "data": [input_data],
    "background_data": bg_data,
    "params": {"grouping_strategy": "time", "m": 5}
}
result = explainer.explain(request)["result"]

# 4. Extract Scores for Class 0
scores = np.array(result)[0, :, 0] # Shape: (Time_Steps,)
```