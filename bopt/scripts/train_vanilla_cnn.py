import os
from IPython import embed
from IPython import embed
import bopt
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')  # Non-GUI backend to avoid Tkinter issues
import torch
import torch.nn as nn
import numpy as np
import datetime
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Clear GPU memory
torch.cuda.empty_cache()
signal = 'cnn'
direction = 'shifted'

# Select Mouse
# h5_filepath = '/mnt/data/cortical_data/20ms_weedle_reconstructed.h5'
h5_filepath = '/mnt/data/ctx/charm_50_rec.h5'
# h5_filepath = '/mnt/data/ctx/goldroger_rec_50b_reconstructed.h5'
experiment='charmander'

run_name = f'{experiment}_{direction}'
save_dir = f'/home/zalaoui/charmander_bestcells/{run_name}'
os.makedirs(save_dir, exist_ok=True)
project_name = f'{experiment}_all_clusters'

device = bopt.cuda_init()


samplerate = 50
num_before = 25
num_after = 5
# Model parameters random seed
seed = 2222
torch.random.manual_seed(seed)

# Model training parameters
batch_size = 256
learning_rate = 1e-5
l1_lambda = 1e-5
weight_decay = 1e-5
num_epochs = 100

# Default model architecture
num_layers = 6
num_channels = 24
kernel_size = 7
padding = 'valid'
mlp_size = 256

# Select training and validation data
train_series = ['series_005/epoch_001', 'series_006/epoch_001']
train_idxs = [0, -samplerate * 10]
test_series = ['series_005/epoch_001', 'series_006/epoch_001']
test_idxs = [-samplerate * 10, -1]


run = wandb.init(project=project_name, name=run_name)
wandb.config.update({
    "num_layers": num_layers,
    "num_channels": num_channels,
    "kernel_size": kernel_size,
    "padding": padding,
    "mlp_size": mlp_size,
    "learning_rate": learning_rate,
    "l1_lambda": l1_lambda,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "seed": seed
})

# which_clusters = [
#     4, 15, 41, 42, 43, 50, 62, 107, 121, 168, 225, 226, 245, 251, 259, 261,
#     263, 271, 282, 294, 302, 327, 334, 340, 342, 347, 363, 364, 367, 375, 400,
#     555
# ]

# Cell Selection from All Cell Model
path='/home/zalaoui/models/charmander_all_clusters/all_cell_corrs_epoch_99.txt'
threshold=0.15
cells, cells_above_thresh= bopt.get_cells_above_threshold(path, threshold)
cell_ids = [cell_id for cell_id, _ in cells_above_thresh]
which_clusters = None

train_loader, train_dataset = bopt.create_dataloader(
    h5_filepath=h5_filepath,
    series=train_series,
    start_idx=train_idxs[0],
    end_idx=train_idxs[1],
    direction=direction,
    shuffle=True,  # Shuffle for training
    batch_size=batch_size,
    which_clusters=which_clusters,
    zero_blinks=False,
)

# Create test loader
test_loader, test_dataset = bopt.create_dataloader(
    h5_filepath=h5_filepath,
    series=test_series,
    start_idx=test_idxs[0],
    end_idx=test_idxs[1],
    direction=direction,
    shuffle=False,  # No shuffle for testing
    batch_size=batch_size,
    which_clusters=which_clusters,
    zero_blinks=False,
)

print(f'Unit Ids Loaded: {train_dataset.cluster_ids}')

# # Initialize model
# model = bopt.CNNComponent(dataset=train_dataset,
#                     num_layers=num_layers,
#                     num_channels=num_channels,
#                     kernel_sizes=kernel_size,
#                     padding=padding,
#                     input_layernorm=None).to(device)

model = bopt.CNNComponent(dataset=train_dataset, num_layers=num_layers, num_channels=num_channels, kernel_sizes=kernel_size, padding=padding, input_layernorm=False).to(device)
# Initialize loss function and optimizer
loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=5,
                                                       verbose=True)

# Train model
bopt.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir,
        signal=signal,
    )
