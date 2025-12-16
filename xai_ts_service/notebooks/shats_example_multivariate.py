"""
Arabic Digits ShaTS Explanation Example (Enhanced Documentation)

This script demonstrates how to generate feature attribution explanations for the 
SpokenArabicDigits dataset using the ShaTS (Shapelet-based Time Series) method.

It covers:
1.  **Model Training**: Training a SimpleCNN on multivariate time series (13 channels, 65 timesteps).
2.  **ShaTS Integration**: Wrapping the model and preparing data for the ShaTS explainer.
3.  **Temporal Explanation**: Calculating and visualizing which *time steps* are most important.
4.  **Channel Explanation**: Calculating and visualizing which *features (channels)* are most important.

--------------------------------------------------------------------------------
KEY CONCEPTS & DATA SHAPES
--------------------------------------------------------------------------------
* **Original Data (PyTorch Format)**: `(Batch, Channels, Time)`
    - Example: `(1, 13, 65)`
    - This is what the CNN model expects.

* **ShaTS Data Format**: `(Batch, Time, Channels)`
    - Example: `(1, 65, 13)`
    - ShaTS groups features by the first dimension (Time) by default.
    - We must TRANSPOSE inputs before sending to ShaTS, and transpose back in the wrapper.

* **Grouping Strategies**:
    - `'time'`: Calculates importance for each time step (Result: 65 scores).
    - `'feature'`: Calculates importance for each channel (Result: 13 scores).
    - Note: A combined 2D (Time x Feature) map is not natively produced by ShaTS.
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import local modules
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f'{script_path}/../')

import base.model as bm
import base.data as bd
# Import the ShaTS implementation (Ensure this path matches your project structure)
import app.services.explainers.shats_impl as shats

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# PART 1: DATA LOADING & MODEL TRAINING
# ==============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading SpokenArabicDigits dataset...')
dataloader_train, dataset_train = bd.get_UCR_UEA_dataloader('SpokenArabicDigits', split='train')
dataloader_test, dataset_test = bd.get_UCR_UEA_dataloader('SpokenArabicDigits', split='test')

output_classes = dataset_train.y_shape[1]
print(f'Dataset info: {len(dataset_train)} train samples, {len(dataset_test)} test samples, {output_classes} classes')

# Check data dimensions
# Expected: (Channels, Time) -> (13, 65) for SpokenArabicDigits
sample_shape = dataset_train[0][0].shape
print(f'Sample shape: {sample_shape}')
is_multivariate = len(sample_shape) > 1 and sample_shape[0] > 1

# Initialize Model
if is_multivariate:
    print(f'Multivariate time series with {sample_shape[0]} channels and {sample_shape[1]} time steps')
    model = bm.SimpleCNNMulti(
        input_channels=sample_shape[0], 
        output_channels=output_classes,
        sequence_length=sample_shape[1]
    ).to(device)
else:
    print(f'Univariate time series with {sample_shape[-1]} time steps')
    model = bm.SimpleCNN(output_channels=output_classes).to(device)

# Model Persistence (Load or Train)
models_dir = os.path.abspath(os.path.join(script_path, '..', 'models'))
os.makedirs(models_dir, exist_ok=True)
model_type = "multi" if is_multivariate else "uni"
model_file = os.path.join(models_dir, f'cnn_{model_type}_arabicdigits_{output_classes}ch.pth')

model_loaded = False
if os.path.exists(model_file):
    print(f'Loading saved model from {model_file}')
    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model_loaded = True
else:
    print(f'No saved model at {model_file}; training will run.')

# Training Helpers
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def trainer(model_, dataloader, criterion_):
    running_loss = 0
    model_.train()
    for _, (inputs, labels) in enumerate(dataloader):
        if is_multivariate:
            inputs = inputs.float().to(device)
        else:
            inputs = inputs.reshape(inputs.shape[0], 1, -1).float().to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        preds = model_(inputs)
        loss_val = criterion_(preds, labels.argmax(dim=-1))
        loss_val.backward()
        optimizer.step()
        running_loss += loss_val.item()
    return running_loss / len(dataloader)

def validator(model_, dataloader, criterion_):
    running_loss = 0
    model_.eval()
    for _, (inputs, labels) in enumerate(dataloader):
        if is_multivariate:
            inputs = inputs.float().to(device)
        else:
            inputs = inputs.reshape(inputs.shape[0], 1, -1).float().to(device)
        labels = labels.float().to(device)
        preds = model_(inputs)
        loss_val = criterion_(preds, labels.argmax(dim=-1))
        running_loss += loss_val.item()
    return running_loss / len(dataloader)

# Train Loop
if not model_loaded:
    print('Training model...')
    epochs = 100
    for epoch in range(epochs):
        train_loss = trainer(model, dataloader_train, criterion)
        if epoch % 10 == 0:
            val_loss = validator(model, dataloader_test, criterion)
            print(f'Epoch {epoch:3d}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')
    torch.save(model.state_dict(), model_file)
    print(f'Model saved to {model_file}')

# ==============================================================================
# PART 2: SAMPLE SELECTION
# ==============================================================================

print('Selecting a sample to explain...')
model.eval()
sample, label = None, None
original_pred_np, original_class = None, None
max_attempts = 100

for attempts in range(max_attempts):
    random_idx = np.random.randint(0, len(dataset_test))
    candidate_sample, candidate_label = dataset_test[random_idx]
    
    with torch.no_grad():
        sample_tensor = torch.tensor(candidate_sample, dtype=torch.float32, device=device)
        # Reshape to (1, Channels, Time) for the model
        if is_multivariate:
            sample_tensor = sample_tensor.unsqueeze(0) 
        else:
            sample_tensor = sample_tensor.unsqueeze(0).unsqueeze(0)
        
        pred_output = model(sample_tensor)
        pred_class = torch.argmax(pred_output, dim=-1).item()
        true_class = np.argmax(candidate_label)
        
        if pred_class == true_class:
            sample = candidate_sample
            label = candidate_label
            original_class = pred_class
            original_pred_np = torch.softmax(pred_output, dim=-1).squeeze().cpu().numpy()
            print(f'Found correctly classified sample {random_idx} (Class {original_class})')
            break

if sample is None:
    print('Failed to find a correctly classified sample.')
    exit(1)

# ==============================================================================
# PART 3: SHATS INTEGRATION & CONFIGURATION
# ==============================================================================
print("\n" + "="*80)
print(" SHATS EXPLANATION SETUP ")
print("="*80)

# --- A. Data Transformation ---
# ShaTS requires input shape: (Time, Channels)
# Our data is currently:      (Channels, Time) -> (13, 65)
# We must TRANSPOSE the sample and background data.

if is_multivariate:
    shats_input_sample = sample.T  # Shape becomes (65, 13)
else:
    shats_input_sample = sample.reshape(-1, 1)

# --- B. Background Data Generation ---
# ShaTS needs a "background" dataset to sample perturbations from.
# We take 50 random samples from the test set and apply the same transpose.
bg_indices = np.random.choice(len(dataset_test), 50, replace=False)
background_data_list = []
for idx in bg_indices:
    bg_sample, _ = dataset_test[idx]
    if is_multivariate:
        background_data_list.append(bg_sample.T) 
    else:
        background_data_list.append(bg_sample.reshape(-1, 1))

print(f"\n[DATA DIMENSION CHECK]")
print(f"Original PyTorch Sample: {sample.shape} (Channels, Time)")
print(f"ShaTS Input Sample:      {shats_input_sample.shape} (Time, Channels)")
print(f"Background Dataset Size: {len(background_data_list)} samples")

# --- C. Model Wrapper ---
# Since we feed ShaTS transposed data, ShaTS will pass transposed data 
# (Batch, Time, Channels) to the model.
# This wrapper catches that input, swaps the dimensions back to 
# (Batch, Channels, Time), and returns probabilities.

class VerboseModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.has_printed_shape = False

    def __call__(self, x_numpy):
        # 1. Convert numpy array from ShaTS to Tensor
        tensor_in = torch.tensor(x_numpy, dtype=torch.float32).to(self.device)
        
        # 2. Debug Log (Prints only once)
        if not self.has_printed_shape:
            print(f"\n[MODEL WRAPPER LOG]")
            print(f"   Input from ShaTS: {tensor_in.shape} (Batch, Time, Channels)")
            self.has_printed_shape = True

        # 3. Transpose for PyTorch Model
        #    (Batch, Time, Channels) -> (Batch, Channels, Time)
        if len(tensor_in.shape) == 3:
            tensor_in = tensor_in.permute(0, 2, 1)
        
        # 4. Get Predictions
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu().numpy()

shats_wrapper = VerboseModelWrapper(model, device)
explainer = shats.ShatsExplainer()

# ==============================================================================
# PART 4: TEMPORAL IMPORTANCE (STRATEGY: 'time')
# ==============================================================================
# Objective: Identify which TIME STEPS (e.g., t=20 to t=30) were most important.
# Grouping:  ShaTS will group all 13 channels together for each time step.

print("\n" + "-"*40)
print(" Calculating TEMPORAL Importance ('time')")
print("-" * 40)

time_request = {
    "model": shats_wrapper,
    "data": [shats_input_sample], 
    "background_data": background_data_list, 
    "params": {
        "grouping_strategy": "time",     # <--- KEY PARAMETER
        "implementation": "fast",        
        "m": 5,                          
        "batch_size": 32
    }
}

start_t = time.time()
response_time = explainer.explain(time_request)
print(f"Done in {time.time() - start_t:.2f}s")

# Result Shape: (N_Samples, N_TimeSteps, N_Classes)
result_time = np.array(response_time["result"])
time_scores = result_time[0, :, original_class] # Extract scores for the predicted class

# --- Visualization: Temporal Heatmap ---
plt.figure(figsize=(14, 8))
gs = plt.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

# Subplot 1: Original Signal
ax1 = plt.subplot(gs[0])
time_axis = np.arange(len(time_scores))
if is_multivariate:
    for i in range(sample.shape[0]):
        ax1.plot(time_axis, sample[i], alpha=0.3, color='gray') # Faint channels
    ax1.plot(time_axis, np.mean(sample, axis=0), color='black', label='Mean Intensity')
else:
    ax1.plot(time_axis, sample, color='black')
ax1.set_title(f"Original Signal (Class {original_class})")
ax1.set_ylabel("Amplitude")
ax1.legend()

# Subplot 2: Importance Heatmap
ax2 = plt.subplot(gs[1])
# Normalize scores (-1 to 1) for consistent colormap
max_abs = np.max(np.abs(time_scores)) + 1e-9
norm_scores = time_scores / max_abs

im = ax2.imshow(
    norm_scores.reshape(1, -1),
    cmap='seismic',     # Red=Positive, Blue=Negative
    aspect='auto',
    vmin=-1, vmax=1,
    extent=[0, len(time_axis), 0, 1]
)

# Overlay Line Plot
ax2_line = ax2.twinx()
ax2_line.plot(time_axis, norm_scores, color='purple', linestyle='--', label='Importance')
ax2_line.set_ylabel("Importance Magnitude", color='purple')
ax2.set_title("ShaTS Temporal Importance (Which TIME matters?)")
ax2.set_xlabel("Time Step")
ax2.set_yticks([])

plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2, label='Contribution (Red+, Blue-)')
plt.savefig(f'shats_temporal_class_{original_class}.png', dpi=300)
print(f"Saved temporal plot to shats_temporal_class_{original_class}.png")
plt.show()


# ==============================================================================
# PART 5: CHANNEL IMPORTANCE (STRATEGY: 'feature')
# ==============================================================================
# Objective: Identify which CHANNELS (e.g., MFCC_1 vs MFCC_12) were most important.
# Grouping:  ShaTS will group all 65 time steps together for each channel.

print("\n" + "-"*40)
print(" Calculating CHANNEL Importance ('feature')")
print("-" * 40)

feature_request = {
    "model": shats_wrapper,
    "data": [shats_input_sample], 
    "background_data": background_data_list, 
    "params": {
        "grouping_strategy": "feature",  # <--- KEY PARAMETER
        "implementation": "fast",        
        "m": 5,
        "batch_size": 32
    }
}

start_f = time.time()
response_feat = explainer.explain(feature_request)
print(f"Done in {time.time() - start_f:.2f}s")

# Result Shape: (N_Samples, N_Channels, N_Classes)
result_feat = np.array(response_feat["result"])
channel_scores = result_feat[0, :, original_class]

print(f"Channel Scores Shape: {channel_scores.shape} (13 channels)")
print(f"Top Channel Index:    {np.argmax(channel_scores)}")

# --- Visualization: Feature Bar Chart ---
plt.figure(figsize=(12, 6))

max_abs_feat = np.max(np.abs(channel_scores)) + 1e-9
norm_feat = channel_scores / max_abs_feat
colors = plt.cm.seismic(0.5 + 0.5 * norm_feat)

channels = np.arange(len(channel_scores))
plt.bar(channels, channel_scores, color=colors, edgecolor='black', alpha=0.8)
plt.axhline(0, color='black', linewidth=1)

plt.title(f"ShaTS Channel Importance (Which FEATURE matters?)")
plt.xlabel("Channel Index")
plt.ylabel("Importance Score")
plt.xticks(channels)

# Colorbar for Bar Chart
sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(-max_abs_feat, max_abs_feat))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca()) # Explicitly attach to current axes
cbar.set_label('Contribution')

plt.savefig(f'shats_channel_class_{original_class}.png', dpi=300)
print(f"Saved channel plot to shats_channel_class_{original_class}.png")
plt.show()

print("\nDone.")