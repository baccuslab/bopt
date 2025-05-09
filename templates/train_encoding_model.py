import os
from IPython import embed
import bopt
import scipy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime

# Clear GPU memory torch.cuda.empty_cache()

run_name = 'goldroger_cnn'
device = bopt.cuda_init()

# Timing information for single forward pass
samplerate = 50
num_before = 25
num_after = 5

# Model parameters random seed
seed = 2222
torch.random.manual_seed(seed)

h5_filepath = '/home/jbmelander/goldroger_50hz_prxs.h5'

# Set some meta-training parameters

epoch_save_interval = 3

# Model training parameters
batch_size = 128
learning_rate = 1e-4
l1_lambda = 1e-5
weight_decay = 1e-5
num_epochs = 100

# Select training and validation data
train_series = ['series_005/epoch_001', 'series_006/epoch_001']
train_idxs = [0, -samplerate * 10]
test_series = ['series_005/epoch_001', 'series_006/epoch_001']
test_idxs = [-samplerate * 10, -1]

# very best clusetrs
# which_clusters = [223, 224, 239, 246, 254, 260, 263, 265]  # kakuna
# which_clusters = [
#     107, 225, 227, 245, 259, 263, 271, 294, 314, 327, 334, 340, 342, 364, 367,
#     375, 379, 380, 385, 386, 390, 555
# ]  # charmander

which_clusters = None
train_dataset = bopt.CorticalDataset(h5_filepath,
                                     train_series,
                                     num_before=num_before,
                                     num_after=num_after,
                                     start_idx=train_idxs[0],
                                     end_idx=train_idxs[1],
                                     which_clusters=which_clusters)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=5)
test_dataset = bopt.CorticalDataset(h5_filepath,
                                    test_series,
                                    num_before=num_before,
                                    num_after=num_after,
                                    start_idx=test_idxs[0],
                                    end_idx=test_idxs[1],
                                    which_clusters=which_clusters)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

which_clusters = train_dataset.cluster_ids
# Initialize run and make directories

model = bopt.LN(dataset=train_dataset).to(device)
# model = bopt.CNN2(dataset=train_dataset,
#                   num_layers=4,
#                   num_channels=20,
#                   kernel_sizes=12,
#                   padding='valid',
#                   mlp_size=256).to(device)
#
# Initialize loss function and optimizer
loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.5,
                                                       patience=5,
                                                       verbose=True)

if os.path.exists('/home/jbmelander/Models/{}'.format(run_name)):
    yn = input('Model directory already exists. Overwrite? (y/n)')
    if yn == 'y':
        os.makedirs('/home/jbmelander/Models/{}'.format(run_name),
                    exist_ok=True)
    else:
        raise ValueError('Model directory already exists')

else:
    os.makedirs('/home/jbmelander/Models/{}'.format(run_name), exist_ok=True)


# Save initial model
def validate(model, dataloader):
    model.eval()

    resps = []
    true = []
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        stimulus, response = data[0].to(device).float(), data[1].to(device)
        true.append(response.detach().cpu().numpy())
        output = model(stimulus).detach().cpu().numpy()
        resps.append(output)

    resps = np.concatenate(resps, axis=0).T
    true = np.concatenate(true, axis=0).T

    corrs = []
    cell = 0
    for CELL, R, T in zip(which_clusters, resps, true):
        plt.plot(T, 'k', alpha=0.5)
        plt.plot(R, 'r')
        corr = np.corrcoef(R, T)[0, 1]
        print(corr)
        corrs.append(corr)
        plt.title('Correlation: {}'.format(corr))
        plt.savefig('/home/jbmelander/Models/{}/val_{}.png'.format(
            run_name, CELL))
        plt.close()

        cell += 1
    print('Mean correlation: {}'.format(np.nanmean(corrs)))
    print('Median correlation: {}'.format(np.nanmedian(corrs)))
    print('Max correlation: {}'.format(np.nanmax(corrs)))
    print('Min correlation: {}'.format(np.nanmin(corrs)))

    return np.nanmean(corrs), np.nanmedian(corrs), np.nanmax(corrs), np.nanmin(
        corrs)


best_max_corr = -np.inf
best_median_corr = -np.inf
best_mean_corr = -np.inf

for epoch in range(num_epochs):
    for i, data in tqdm.tqdm(enumerate(train_loader)):
        model.train()
        optimizer.zero_grad()

        stimulus, response = data[0].to(device).float(), data[1].to(device)
        output = model(stimulus)
        poisson_loss = loss_fn(output, response)

        for name, param in model.named_parameters():
            if 'weight' in name:
                poisson_loss += l1_lambda * torch.norm(param, p=1)

        poisson_loss.backward()
        print(poisson_loss.item())

        optimizer.step()

        if i % 300 == 0:
            model.eval()
            mean, median, max, min = validate(model, test_loader)
            model.train()

    if epoch % epoch_save_interval == 0:
        torch.save(
            model,
            '/home/jbmelander/Models/{}/epoch_{}.pt'.format(run_name, epoch))
    scheduler.step(poisson_loss)
    torch.save(model, '/home/jbmelander/Models/{}/current.pt'.format(run_name))

    model.eval()
    mean, median, max, min = validate(model, test_loader)

    if mean > best_mean_corr:
        best_mean_corr = mean
        torch.save(model,
                   '/home/jbmelander/Models/{}/best_mean.pt'.format(run_name))
    if median > best_median_corr:
        best_median_corr = median
        torch.save(
            model,
            '/home/jbmelander/Models/{}/best_median.pt'.format(run_name))
    if max > best_max_corr:
        best_max_corr = max
        torch.save(model,
                   '/home/jbmelander/Models/{}/best_max.pt'.format(run_name))
