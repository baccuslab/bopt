import torch
from IPython import embed
import h5py as h5
import torch
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import scipy.stats
import bopt
import tqdm
import numpy as np
import matplotlib as mpl
import imageio
import wandb


def dataset_from_atomic_model(model,
                              experiment,
                              series,
                              idxs=(0, -1),
                              alt_file=None,
                              data_streams=None):
    if alt_file is not None:
        h5_filepath = alt_file
    else:
        h5_filepath = model.datasets[experiment].h5_filepath
    direction = model.datasets[experiment].stimulus_key
    clusters = model.datasets[experiment].clusters
    num_before = model.datasets[experiment].num_before
    num_after = model.datasets[experiment].num_after

    start_idx = idxs[0]
    end_idx = idxs[1]

    if not isinstance(series, list):
        series = [series]

    dataset = bopt.CorticalDataset(h5_filepath,
                                   series,
                                   clusters=clusters,
                                   num_before=num_before,
                                   num_after=num_after,
                                   stimulus_key=direction,
                                   start_idx=start_idx,
                                   end_idx=end_idx,
                                   data_streams=data_streams)

    return dataset


def dataset_from_model(model,
                       series,
                       idxs=(0, -1),
                       alt_file=None,
                       signals=None,
                       which_clusters=None):
    # Determine the h5 file path
    if alt_file is not None:
        h5_filepath = alt_file
    else:
        h5_filepath = model.dataset.h5_filepath

    # If which_clusters is not provided, try to retrieve clusters from the model's dataset.
    if which_clusters is None and hasattr(model.dataset, 'clusters'):
        which_clusters = model.dataset.clusters

    # Retrieve num_before and num_after values (or use defaults if not defined in model.dataset)
    num_before = getattr(model.dataset, 'num_before', 30)
    num_after = getattr(model.dataset, 'num_after', 10)

    # Unpack start and end indices
    start_idx, end_idx = idxs

    # Ensure series is a list
    if not isinstance(series, list):
        series = [series]

    # Create an instance of your new CorticalDataset
    dataset = bopt.CorticalDataset(h5_filepath,
                                   series,
                                   num_before=num_before,
                                   num_after=num_after,
                                   start_idx=start_idx,
                                   end_idx=end_idx,
                                   signals=signals,
                                   which_clusters=which_clusters)
    return dataset


def load_saved_model(model_path, device='cuda'):
    model = torch.load(model_path, map_location=device)
    return model


def present_stim(model, stim, device='cuda', batch_size=128):
    # Stim must be a torch.Tensor of shape (n_cells, height, width)
    # Returns the model responses to a stimulus
    # stim: torch.Tensor of shape (n_cells, timepoints-window)

    dataset = bopt.ArtificialStimulus(stim, model.window)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    model_outputs = []

    for i, data in enumerate(dataloader):
        X = data.to(device).float()
        out = model(X)
        model_outputs.append(out.cpu().detach().numpy())

    model_outputs = np.concatenate(model_outputs).T
    return model_outputs


def gather_unrolled_responses(model, dataset, device=None):
    if device is None:
        device = bopt.cuda_init()

    if not dataset.spike_history:
        raise ValueError('Dataset must have spike history')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False)
    # Recurrent predictions
    rec_predictions = []
    # Using ground truth history
    true_predictions = []
    responses = []

    cluster_ids = dataset.clusters
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            if i == 0:
                current_h = data[2].to(device).float()

            stim = data[0].to(device).float()
            resp = data[1].to(device).float()
            spike_hist = data[2].to(device).float()

            rec_pred = torch.squeeze(model(stim, current_h))
            rec_predictions.append(rec_pred.detach().cpu().numpy())

            true_pred = torch.squeeze(model(stim, spike_hist))
            true_predictions.append(true_pred.detach().cpu().numpy())

            responses.append(resp.detach().cpu().numpy())

            # Shift the history
            current_h[:, :, :-1] = current_h[:, :, 1:]
            current_h[:, :, -1] = rec_pred

    rec_predictions = np.array(rec_predictions).T
    true_predictions = np.array(true_predictions).T
    responses = np.array(responses)[:, 0, :].T

    rp_dict = {}
    tp_dict = {}
    resp_dict = {}

    for rp, tp, resp, clust in zip(rec_predictions, true_predictions,
                                   responses, cluster_ids):
        rp_dict[str(clust)] = np.array(rp)
        tp_dict[str(clust)] = np.array(tp)
        resp_dict[str(clust)] = np.array(resp)

    return resp_dict, rp_dict, tp_dict


def gather_responses(model,
                     dataset,
                     device='cuda',
                     batch_size=128,
                     use_static_nl=False,
                     use_spike_history=False):

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    responses = []
    predictions = []

    if use_static_nl:
        if 'static_nls' in model.__dict__:
            print('Using static nonlinearity...')
            use_nl = True

    else:
        use_nl = False

    for i, data in enumerate(tqdm.tqdm(data_loader)):
        stimulus = data[0].to(device).float().to(device)
        response = data[1].to(device)

        if use_spike_history:
            print('SpikeHistory')
            history = data[2].to(device).float()
            prediction = model(stimulus, history)

        else:
            prediction = model(stimulus)

        responses.append(response.cpu().detach().numpy())

        prediction = prediction.cpu().detach().numpy()
        if use_nl:
            nl_prediction = model.static_nls[i].predict(prediction)
            predictions.append(nl_prediction)
        else:
            predictions.append(prediction)

    predictions = np.concatenate(predictions).T
    responses = np.concatenate(responses).T

    cluster_ids = dataset.clusters
    corrs = []

    for clust, resp, pred in zip(cluster_ids, responses, predictions):
        pearsonr = scipy.stats.pearsonr(resp, pred)[0]
        corrs.append(pearsonr)

    responses_dict = {}
    predictions_dict = {}
    corrs_dict = {}

    for clust, resp, pred, corr in zip(cluster_ids, responses, predictions,
                                       corrs):
        responses_dict[str(clust)] = resp
        predictions_dict[str(clust)] = pred
        corrs_dict[str(clust)] = corr

    return responses_dict, predictions_dict, corrs_dict


def model_split_half(repeats, predictions, N=30):
    """
    Over N random permutations, splits the repeats into two halves and calculate the correlation between them.
    Returns a list of the correlation for all permutations

    repeats: np.array of shape (n_repeats, n_timepoints)   
    """
    neural_corrs = []
    model_corrs = []
    model_neural_corrs = []

    for i in range(N):
        # Get random indices
        indices = np.arange(len(repeats))

        # Set new seed
        np.random.seed(i)
        np.random.shuffle(indices)

        resps1 = repeats[indices[:int(len(repeats) / 2)]]
        resps2 = repeats[indices[int(len(repeats) / 2):]]

        preds1 = predictions[indices[:int(len(repeats) / 2)]]
        preds2 = predictions[indices[int(len(repeats) / 2):]]

        rm1 = resps1.mean(axis=0)
        rm2 = resps2.mean(axis=0)
        pm1 = preds1.mean(axis=0)
        pm2 = preds2.mean(axis=0)

        neural_corrs.append(scipy.stats.pearsonr(rm1, rm2)[0])
        model_corrs.append(scipy.stats.pearsonr(pm1, pm2)[0])
        pm1_rm1 = scipy.stats.pearsonr(pm1, rm1)[0]
        pm2_rm2 = scipy.stats.pearsonr(pm2, rm2)[0]
        model_neural_corrs.append([pm1_rm1, pm2_rm2])

    return neural_corrs, model_corrs, model_neural_corrs


def split_half(repeats, N=30):
    """
    Over N random permutations, splits the repeats into two halves and calculate the correlation between them.
    Returns a list of the correlation for all permutations

    repeats: np.array of shape (n_repeats, n_timepoints)   
    """
    corrs = []
    for i in range(N):
        np.random.shuffle(repeats)
        half = int(len(repeats) / 2)
        half1 = repeats[:half]
        half2 = repeats[half:]

        corr = np.corrcoef(half1.mean(axis=0), half2.mean(axis=0))[0, 1]
        corrs.append(corr)

    return corrs


def get_repeats(filepath, series_name, firing_rate_tax=None):
    """
    Grabs the repeats from a series in a h5 file
    """
    file = h5.File(filepath, 'r')

    series = file[series_name]
    epoch_names = [key for key in series.keys() if key.startswith('epoch_')]

    cluster_ids = file['cluster_ids'][:]

    resps = []
    phis = []
    thetas = []
    locos = []

    for epoch_name in epoch_names:
        unit_ids = list(series[epoch_name]['resp'].keys())

        resp_agg = []
        phis.append(series[epoch_name]['phis'][:])
        thetas.append(series[epoch_name]['thetas'][:])
        locos.append(series[epoch_name]['locomotion'][:])

        cluster_ids = []

        for unit in unit_ids:
            resp = series[epoch_name]['resp'][unit][:]

            resp_agg.append(resp)
            cluster_ids.append(unit)

        resps.append(resp_agg)

    resps = np.array(resps)
    phis = np.array(phis)
    thetas = np.array(thetas)
    locos = np.array(locos)

    resps = np.swapaxes(resps, 0, 1)
    return cluster_ids, resps, phis, thetas, locos


def get_repeats_new(filepath, series_name):
    """
    Grabs the repeats from a series in an H5 file.

    Parameters:
    - filepath (str): Path to the H5 file.
    - series_name (str): Name of the series in the H5 file.

    Returns:
    - np.array: Array of firing rate responses with dimensions (units, trials).
    """
    with h5.File(filepath, 'r') as file:
        series = file['data'][series_name]
        epoch_names = [
            key for key in series.keys() if key.startswith('epoch_')
        ]

        resps = []

        for epoch_name in epoch_names:
            resp_agg = []
            firing_rates = series[epoch_name]['firing_rates']

            for unit in range(len(firing_rates)):  # Fixed the loop
                resp = firing_rates[unit][:]
                resp_agg.append(resp)

            resps.append(resp_agg)

        resps = np.array(resps)
        resps = np.swapaxes(resps, 0, 1)

    return resps


def identify_high_reliability_neurons(filepath, series_list, threshold):
    """
    Identifies neurons with split-half reliability above a given threshold.

    Parameters:
    - filepath (str): Path to the H5 file.
    - series_list (list): List of series names to analyze.
    - threshold (float): The minimum split-half reliability required.

    Returns:
    - dict: Dictionary where keys are series names and values are lists of neuron indices exceeding the threshold.
    """
    high_reliability_neurons = {}

    for series_name in series_list:
        # Get repeat data from H5 file
        resp = get_repeats_new(filepath, series_name)
        num_units = resp.shape[0]  # Number of neurons

        split_half_results = {}

        for i in range(num_units):
            trials = resp[i, :]

            if trials.size == 0 or np.isnan(trials).all():
                split_half_results[i] = np.nan
                continue

            # Compute split-half reliability
            split_half_i = split_half(trials[:, None], N=100)
            split_half_value = np.nanmean(split_half_i) if len(
                split_half_i) > 0 else np.nan
            split_half_results[i] = split_half_value

        high_neurons = [(i, split_half_results[i])
                        for i, value in split_half_results.items()
                        if not np.isnan(value) and value > threshold]
        high_reliability_neurons[series_name] = high_neurons
        # Identify neurons exceeding the threshold
        print(
            f"Series {series_name}: {len(high_neurons)} neurons exceed the threshold of {threshold}"
        )
        for neuron_idx, split_half_value in high_neurons:
            print(
                f"  Neuron {neuron_idx}: Split-Half Reliability = {split_half_value:.3f}"
            )
        print("\n")

    return split_half_results, high_reliability_neurons


def plot_split_half(filepath, series_name):
    """
    Computes and plots the split-half reliability for each cluster in the dataset.

    Parameters:
    - filepath (str): Path to the H5 file containing the series.
    - series_name (str): The name of the series within the H5 file to analyze.

    Returns:
    - None: Displays a histogram of split-half reliability values.
    """

    # Get repeat data from H5 file
    resp = get_repeats_new(
        filepath,
        series_name)  # Now only retrieving resp since clusters don't exist
    num_units = resp.shape[0]  # Number of units

    split_half_results = {}

    for i in range(num_units):  # Iterate over units instead of clusters
        trials = resp[i, :]  # Fix indexing

        if trials.size == 0 or np.isnan(trials).all():
            split_half_results[
                i] = np.nan  # Using i as the key since clusters are undefined
            continue

        # Compute split-half reliability
        split_half_i = split_half(trials[:, None], N=100)
        split_half_results[i] = np.nanmean(split_half_i) if len(
            split_half_i) > 0 else np.nan

    # Filter out NaN values
    split_half_values = np.array(list(split_half_results.values()))
    split_half_values = split_half_values[~np.isnan(split_half_values)]

    # Plot Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(split_half_values,
             bins=20,
             color='skyblue',
             edgecolor='black',
             alpha=0.7)
    plt.title(f"Histogram of Split-Half Reliability for {series_name}")
    plt.xlabel("Split-Half Reliability")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()
# Create a validation function
def gather_paramresponses(model, dataloader, signal, device, save_dir=None):
    model.eval()
    
    resps = []
    true = []
    for i, data in enumerate(tqdm.tqdm(dataloader, desc="Validating")):
        stimulus, response, locomotion, velocity = data[0].to(device).float(), data[1].to(device), data[2].to(device).float(), data[3].to(device).float()
        
        # Handle different signal types
        if signal == 'cnn':
            with torch.no_grad():
                output = model(stimulus).detach().cpu().numpy()
        
        elif signal == 'gaze':
            with torch.no_grad():
                output = model(stimulus,velocity).detach().cpu().numpy()
        
        elif signal == 'locomotion':
            with torch.no_grad():
                output = model(stimulus,locomotion).detach().cpu().numpy()
        elif signal == 'both':
            combined_signals = torch.cat([velocity, locomotion], dim=1)
            with torch.no_grad():
                output = model(stimulus, combined_signals).detach().cpu().numpy()
        elif signal == 'shifter':
            # For SimpleShifter, the model will handle the shifting internally
            with torch.no_grad():
                # If your SimpleShifter expects a tuple:
                batch = (stimulus, response, locomotion, velocity )
                output = model(stimulus, locomotion, velocity).detach().cpu().numpy()

        resps.append(output)
        true.append(response.detach().cpu().numpy())
    
    # The rest of your validation code stays the same
    resps = np.concatenate(resps, axis=0).T
    true = np.concatenate(true, axis=0).T
    
    corrs = {}
    cell = 0
    for CELL, (R, T) in enumerate(zip(resps, true)):
        plt.figure(figsize=(10, 5))
        plt.plot(T, 'k', alpha=0.5, label='True')
        plt.plot(R, 'r', label='Predicted')
        plt.legend()
        corr = np.corrcoef(R, T)[0, 1]
        unit_id = dataloader.dataset.cluster_ids[CELL]
        corrs[unit_id] = corr
        plt.title(f'Cell: {unit_id}, Correlation: {corr:.4f}')
        save_path = f'{save_dir}/val_{unit_id}.png'
        plt.savefig(save_path)
        plt.close()
        
        cell += 1
    
    mean_corr = np.nanmean(np.array(list(corrs.values())))
    median_corr = np.nanmedian(np.array(list(corrs.values())))
    max_corr = np.nanmax(np.array(list(corrs.values())))
    min_corr = np.nanmin(np.array(list(corrs.values())))
    max_corr_cell = max(corrs, key=corrs.get)
    
    print(f'Mean correlation: {mean_corr:.4f}')
    print(f'Median correlation: {median_corr:.4f}')
    print(f'Max correlation: {max_corr:.4f}, Cell: {max_corr_cell}')
    print(f'Min correlation: {min_corr:.4f}')
    
    return corrs



def save_cell_correlations(corrs, save_path):
    """
    Save cell correlations to a text file.
    
    Parameters:
    -----------
    corrs : dict
        A dictionary where keys are cell IDs and values are correlation values
    save_path : str
        Full file path where the correlations will be saved
    """
    with open(save_path, 'w') as f:
        for cell_id, corr_value in corrs.items():
            f.write('{} {:.4f}\n'.format(int(cell_id), corr_value))


def train_model(
    model, 
    train_loader, 
    test_loader, 
    optimizer, 
    loss_fn,
    scheduler,
    num_epochs, 
    device, 
    save_dir, 
    signal,
    save_intervals=[300,3],
    l1_lambda=0.001,

):
    """
    Train a neural network model with Poisson loss, L1 regularization, and periodic validation.

    Args:
        model (torch.nn.Module): The neural network model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        test_loader (torch.utils.data.DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimization algorithm
        loss_fn (callable): Loss function (e.g., Poisson loss)
        num_epochs (int): Number of training epochs
        device (torch.device): Device to run training on (cuda/cpu)
        save_dir (str): Directory to save model checkpoints and results
        signal (str): Signal type for validation
        l1_lambda (float, optional): L1 regularization strength. Defaults to 0.001.
        validation_interval (int, optional): Number of batches between validation. Defaults to 300.

    Returns:
        dict: Training statistics including validation correlations
    """
    best_mean_corr = -np.inf
    best_median_corr = -np.inf
    best_max_corr = -np.inf

    # Ensure model is on correct device
    model.to(device)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        # Progress bar for the epoch

        for i, data in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            optimizer.zero_grad()

            # Unpack data
            stimulus, response, locomotion, velocity = data[0].to(
                device).float(), data[1].to(device), data[2].to(
                    device).float(), data[3].to(device).float()
            # Forward pass
            if signal == 'cnn':
                output = model(stimulus)
            elif signal == 'gaze':
                output = model(stimulus, velocity)
            elif signal == 'locomotion':
                output = model(stimulus, locomotion)
            elif signal == 'both':
                combined_signals = torch.cat([velocity, locomotion], dim=1)  # Concatenate along feature dimension
                output = model(stimulus, combined_signals)
            elif signal == 'shifter':
                output = model(stimulus, locomotion, velocity)
            else:
                raise ValueError(f"Unknown signal type: {signal}")
            poisson_loss = loss_fn(output, response)

            # L1 regularization
            l1_reg = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg += l1_lambda * torch.norm(param, p=1)

            # Total loss
            total_loss = poisson_loss + l1_reg
            total_loss.backward()

            # Optimization step
            optimizer.step()

            # Logging
            wandb.log({
                'batch_loss': total_loss.item(),
                'poisson_loss': poisson_loss.item(),
                'l1_reg':l1_reg.item() if isinstance(l1_reg, torch.Tensor) else l1_reg,
                'batch': i + epoch * len(train_loader),
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Update epoch statistics
            epoch_loss += total_loss.item()
            batch_count += 1

            # Periodic validation
            if i % save_intervals[0] == 0:
                model.eval()
                corrs = gather_paramresponses(
                    model, 
                    test_loader, 
                    signal, 
                    device, 
                    save_dir, 
                )

                mean_corr = np.nanmean(np.array(list(corrs.values())))
                median_corr = np.nanmedian(np.array(list(corrs.values())))
                max_corr = np.nanmax(np.array(list(corrs.values())))
                min_corr = np.nanmin(np.array(list(corrs.values())))
                max_corr_cell = max(corrs, key=corrs.get)

                model.train()
                wandb.log({
                    'validation_step': i + epoch * len(train_loader),
                    'validation_mean_corr': mean_corr,
                    'validation_median_corr': median_corr,
                    'validation_max_corr': max_corr,
                    'validation_min_corr': min_corr
                })
                save_cell_correlations(corrs, f'{save_dir}/all_cell_corrs_epoch_{epoch+1}.txt')
        avg_epoch_loss = epoch_loss / batch_count
        wandb.log({
            'avg_epoch_loss': avg_epoch_loss,
            'epoch': epoch
        })

        # Update learning rate
        scheduler.step(avg_epoch_loss)

        # Periodic model checkpointing
        if epoch % save_intervals[1] == 0:
            checkpoint_path = f'{save_dir}/epoch_{epoch+1}.pt'
            torch.save(model, checkpoint_path)
            wandb.save(checkpoint_path)

    # Run full validation at end of epoch
    model.eval()
    corrs = gather_paramresponses(
        model, 
        test_loader, 
        signal, 
        device, 
        save_dir
    )
    model.train()

    # Compute correlation metrics
    mean_corr = np.nanmean(np.array(list(corrs.values())))
    median_corr = np.nanmedian(np.array(list(corrs.values())))
    max_corr = np.nanmax(np.array(list(corrs.values())))
    min_corr = np.nanmin(np.array(list(corrs.values())))

    # Track and save best models
    if mean_corr > best_mean_corr:
        best_mean_corr = mean_corr
        best_mean_path = f'{save_dir}/best_mean.pt'
        torch.save(model, best_mean_path)
        wandb.log({'best_mean_corr': best_mean_corr})
        wandb.save(best_mean_path)

    if median_corr > best_median_corr:
        best_median_corr = median_corr
        best_median_path = f'{save_dir}/best_median.pt'
        torch.save(model, best_median_path)
        wandb.log({'best_median_corr': best_median_corr})
        wandb.save(best_median_path)

    if max_corr > best_max_corr:
        best_max_corr = max_corr
        best_max_path = f'{save_dir}/best_max.pt'
        torch.save(model, best_max_path)
        wandb.log({'best_max_corr': best_max_corr})
        wandb.save(best_max_path)

    # Log epoch summary
    wandb.log({
        'final_mean_corr': mean_corr,
        'final_median_corr': median_corr,
        'final_max_corr': max_corr,
        'final_min_corr': min_corr,
    })

    # Final model save
    final_path = f'{save_dir}/final_model.pt'
    torch.save(model, final_path)
    wandb.save(final_path)


def get_cells_above_threshold(file_path, threshold):
    cells_above_threshold = []
    all_cells = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    cell_id = int(parts[0])
                    corr_value = float(parts[1])
                    all_cells.append((cell_id, corr_value))
                    
                    if corr_value > threshold:
                        cells_above_threshold.append((cell_id, corr_value))
                except (ValueError, IndexError):
                    # Skip lines that don't follow expected format
                    continue
    
    return all_cells, cells_above_threshold



        
def style_plot(ax):
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ensure linewidth is applied to remaining spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Adjust tick parameters
    ax.tick_params(width=2, size=6)# Configure global matplotlib parameters
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 12
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16

def corr_scatter(x, y, marker='ko', show=False, ax=None, alpha=0.60):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        pass
    ax.plot(x, y, marker, alpha=alpha)
    ax.set_aspect('equal')
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=0.5)
    if show:
        plt.show()
    return ax

def frames2gif(frame_array, filename, duration=150, scale=None):
    print('GIF dtype and shape: ', frame_array.dtype, frame_array.shape)
    if scale is None:
        minimum = np.min(frame_array)
        frame_array = frame_array - minimum
        frame_array = frame_array / np.max(frame_array)
        frame_array = frame_array * 255
        frame_array = frame_array.astype(np.uint8)

        frame_array[0,0:5,0:5]=255
        frame_array[1,0:5,0:5]=0

        imageio.mimsave(filename, frame_array, duration=duration, subrectangles=False, loop=2100)
    else:
        frame_array[frame_array>scale[1]] = scale[1]
        frame_array[frame_array<scale[0]] = scale[0]
        minimum = np.min(frame_array)
        frame_array = frame_array - minimum
        frame_array = frame_array / np.max(frame_array)
        frame_array = frame_array * 255
        frame_array = frame_array.astype(np.uint8)

        frame_array[0,0:5,0:5]=255
        frame_array[1,0:5,0:5]=0

        imageio.mimsave(filename, frame_array, duration=duration, subrectangles=False, loop=2100)


def eval_repeats(model_path, h5_filepath, series, direction, device, signal, selected_unit, which_clusters, num_epochs=10, plot=True, plot_signals=False):
    """
    Evaluate and plot true responses, model responses, velocity, and locomotion for a specific cell across all epochs.
    The function loads all clusters but only plots the responses for the selected cell.
    """
    # Load the model
    model = torch.load(model_path, map_location=device)
    model.eval()


    num_epochs = 10  # Number of epochs to evaluate

    epochs = [f'{series}/epoch_{str(i+1).zfill(3)}' for i in range(num_epochs)]

    print(epochs)
    all_data = []  # Store tuples of (resps, true, velo, loco) for each epoch
    # Iterate over each epoch and evaluate
    for epoch in epochs:
        
        test_dataset = bopt.CorticalDataset(h5_filepath,
                                        [epoch],
                                        num_before=25,
                                        num_after=5,
                                        start_idx=0,
                                        end_idx=-1,
                                        stimulus_key=direction,
                                        grayscale=True,
                                        normalize_signals=False,
                                        signals=['locomotion', 'azimuth'],
                                        which_clusters=which_clusters,
                                        zero_blinks=True)
        
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=256,
                                                shuffle=False)
        
        resps, true, velo_agg, loco_agg = [], [], [], []
        for i, data in enumerate(tqdm.tqdm(test_loader, desc=f"Evaluating {epoch}")):
            stimulus, response, locomotion, velocity = (
                data[0].to(device).float(),
                data[1].to(device),
                data[2].to(device).float(),
                data[3].to(device).float(),
            )
            
            with torch.no_grad():
                if signal == 'cnn':
                    output = model(stimulus)
                elif signal == 'gaze':
                    output = model(stimulus, velocity)
                elif signal == 'locomotion':
                    output = model(stimulus, locomotion)
                elif signal == 'both':
                    combined_signals = torch.cat([velocity, locomotion], dim=1)
                    output = model(stimulus, combined_signals)
                
            resps.append(output.detach().cpu().numpy())
            true.append(response.cpu().numpy())
            velo_agg.append(velocity)
            loco_agg.append(locomotion)
        
        # Aggregate results
        resps = np.concatenate(resps, axis=0).T
        true = np.concatenate(true, axis=0).T
        velo_agg = torch.cat(velo_agg, dim=0)
        loco_agg = torch.cat(loco_agg, dim=0)*2000 +(100)


        all_data.append((resps, true, velo_agg, loco_agg, epoch, test_loader.dataset.cluster_ids))

    

    # Convert to matrix format (cells x epochs)
    num_cells = len(all_data[0][5])  # Number of cells (from first epoch)
    num_epochs = len(all_data)

    corr_matrix = np.full((num_cells, num_epochs), np.nan)  # Initialize NaN matrix


    for epoch_idx, (resps, true, _, _, epoch, cluster_ids) in enumerate(all_data):
        for cell_idx, (R, T) in enumerate(zip(resps, true)):
            corr = np.corrcoef(R, T)[0, 1]
            corr_matrix[cell_idx, epoch_idx] = corr  # Store correlation


    avg_corr_per_cell = np.nanmean(corr_matrix, axis=1)

    # Extract unit IDs (assume from first epoch since units remain the same)
    unit_ids = all_data[0][5]  # Cluster IDs from the first epoch

    # Create figure
    plt.figure(figsize=(12, 6))

    # Show correlation matrix using imshow
    plt.imshow(corr_matrix, aspect='auto', cmap='PuOr', vmin=-1, vmax=1)

    # Add colorbar
    plt.colorbar(label='Correlation Coefficient')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel('Unit IDs')
    plt.title(f'Cell-wise Correlation Across Epochs of {series} ')

    # Set x-ticks as epoch numbers
    plt.xticks(ticks=np.arange(num_epochs), labels=np.arange(1, num_epochs + 1))

    # Set y-ticks as actual unit IDs (adjust step if too many)
    step = max(1, len(unit_ids) // 20)  # Show every Nth unit to avoid clutter
    plt.yticks(ticks=np.arange(len(unit_ids))[::step], labels=np.array(unit_ids)[::step])

    # Show the plot
    bopt.style_plot(plt.gca())
    plt.show()

    plt.figure(figsize=(12, 6))
    
    # Create scatter plot with cell IDs on x-axis and average correlations on y-axis
    plt.scatter(unit_ids, avg_corr_per_cell, color='black', s=30)

    for i, txt in enumerate(unit_ids):
        plt.text(unit_ids[i], avg_corr_per_cell[i], str(txt), fontsize=8, ha='right', va='bottom', alpha=0.7)
    # Add labels and title
    plt.xlabel('Cell ID')
    plt.ylabel('Average Correlation')
    plt.title('Average Correlation Across All Epochs Per Cell')
    
    # Set ylim to show the full range of correlations
    plt.ylim(-.2, 1.1)
    plt.xticks([])  # Removes x-tick labels
    plt.tick_params(axis='x', which='both', bottom=False)  # Hides x-axis ticks

    # Apply style
    bopt.style_plot(plt.gca())
    plt.tight_layout()
    plt.show()
    

    if plot:
        fig, axes = plt.subplots(len(all_data), 1, figsize=(10, len(all_data) * 3), sharex=True)

        for idx, (resps, true, velo_agg, loco_agg, epoch, cluster_ids) in enumerate(all_data):
            # Find the index of the selected unit (e.g., 340)
            if selected_unit not in cluster_ids:
                print(f"Unit {selected_unit} not found in epoch {epoch}, skipping...")
                continue
            
            unit_idx = cluster_ids.index(selected_unit)  # Get index of unit in data
            model_resp = resps[unit_idx]
            true_resp = true[unit_idx]

            # Compute time vector
            time = np.arange(len(model_resp)) / 50  # Assuming 50Hz sampling rate

            ax = axes[idx] if len(all_data) > 1 else axes  # Handle single subplot case

            ax.plot(time, true_resp, 'k-', linewidth=1, label='True response')  # Ground truth in black
            ax.plot(time, model_resp, 'r-', linewidth=1, label='Model response')  # Model response in red
            corr = np.corrcoef(model_resp, true_resp)[0, 1]  # Compute correlation
            if plot_signals:
                # Twin the axis for locomotion and gaze speed
                ax2 = ax.twinx()
                ax2.plot(time, loco_agg.cpu().numpy(), 'g-', linewidth=1, alpha=0.7, label='Locomotion')
                ax2.plot(time, velo_agg.cpu().numpy(), 'b-', linewidth=1, alpha=0.7, label='Gaze speed')
                ax2.set_ylabel('Locomotion and Azimuth', fontsize=8)

            ax.set_ylabel('Firing rate (Hz)')
            ax.set_title(f'Unit {selected_unit} - {epoch}, Corr: {corr:.2f}')
            ax.legend(loc='upper right', fontsize=8)


        ax.set_xlabel('Time (s)')  # Set X-axis label only on the last plot
        bopt.style_plot(plt.gca())
        plt.tight_layout()
        plt.xticks(fontsize=8)
        plt.show()

def eval_model(model_path, dataloader, signal, device, title, plot=False, plot_signals=True):

    model = torch.load(model_path, map_location=device)
    model.eval()
    
    resps = []
    true = []
    velo_agg = []
    loco_agg = []
    for i, data in enumerate(tqdm.tqdm(dataloader, desc="Validating")):
        stimulus, response, locomotion, velocity = data[0].to(device).float(), data[1].to(device), data[2].to(device).float(), data[3].to(device).float()
        
        # Handle different signal types
        if signal == 'cnn':
            with torch.no_grad():
                output = model(stimulus).detach().cpu().numpy()  # Move to CPU and convert to NumPy
        
        elif signal == 'gaze':
            with torch.no_grad():
                output = model(stimulus, velocity).detach().cpu().numpy()  # Move to CPU and convert to NumPy
        
        elif signal == 'locomotion':
            with torch.no_grad():
                output = model(stimulus, locomotion).detach().cpu().numpy()  # Move to CPU and convert to NumPy
        
        elif signal == 'both':
            combined_signals = torch.cat([velocity, locomotion], dim=1)
            with torch.no_grad():
                output = model(stimulus, combined_signals).detach().cpu().numpy()  # Move to CPU and convert to NumPy

        resps.append(output)
        true.append(response.cpu().numpy())  # Ensure response is moved to CPU before converting to NumPy
        velo_agg.append(velocity)
        loco_agg.append(locomotion)
    
    # The rest of your validation code stays the same
    resps = np.concatenate(resps, axis=0).T
    true = np.concatenate(true, axis=0).T
    velo_agg = torch.cat(velo_agg, dim=0) 
    loco_agg = torch.cat(loco_agg, dim=0)*2000 +(100)
    
    corrs = {}
    cell = 0
    for CELL, (R, T) in enumerate(zip(resps, true)):
        corr = np.corrcoef(R, T)[0, 1]
        unit_id = dataloader.dataset.cluster_ids[CELL]
        corrs[unit_id] = corr
        print(f'Cell: {unit_id}, Correlation: {corr:.4f}')
        cell += 1
    
    mean_corr = np.nanmean(np.array(list(corrs.values())))
    median_corr = np.nanmedian(np.array(list(corrs.values())))
    max_corr = np.nanmax(np.array(list(corrs.values())))
    min_corr = np.nanmin(np.array(list(corrs.values())))
    max_corr_cell = max(corrs, key=corrs.get)
    
    print(f'Mean correlation: {mean_corr:.4f}')
    print(f'Median correlation: {median_corr:.4f}')
    print(f'Max correlation: {max_corr:.4f}, Cell: {max_corr_cell}')
    print(f'Min correlation: {min_corr:.4f}')
    print(f'Mean correlation: {mean_corr:.4f}')
    
    if plot:
        corr_pairs = []
        for cell_idx, (model_resp, true_resp) in enumerate(zip(resps, true)):
            corr = np.corrcoef(model_resp, true_resp)[0, 1]
            unit_id = dataloader.dataset.cluster_ids[cell_idx]
            corr_pairs.append((unit_id, corr, model_resp, true_resp))
            print(f'Cell: {unit_id}, Correlation: {corr:.4f}')

        # Sort by correlation (descending) and take top 3
        corr_pairs.sort(key=lambda x: x[1], reverse=True)
        # Assume corr_pairs is a list of tuples: (cluster_id, corr, model_resp, true_resp)
        batch_size = 3  # Plot 3 at a time

        time = np.arange(len(model_resp)) / 50  # Use the length of model_resp to generate time



        # Loop through the correlation pairs in batches of 3
        for batch_start in range(0, len(corr_pairs), batch_size):
            fig, axes = plt.subplots(1, batch_size, figsize=(18, 5))

            for i, (cluster_id, corr, model_resp, true_resp) in enumerate(corr_pairs[batch_start:batch_start + batch_size]):
                ax = axes[i]
                # Plot the true and model responses
                ax.plot(time, true_resp, 'k-', linewidth=1, label='True response')  # Ground truth in black
                ax.plot(time, model_resp, 'r-', linewidth=1, label='Model response')  # Model response in red

                # Normalize locomotion and velocity to the max of true response
                # Plot normalized locomotion and velocity
                if plot_signals:
                    # Twin the axis for locomotion and gaze speed
                    ax2 = ax.twinx()
                    ax2.plot(time, loco_agg.cpu().numpy(), 'g-', linewidth=1, alpha=0.7, label='Locomotion')
                    ax2.plot(time, velo_agg.cpu().numpy(), 'b-', linewidth=1, alpha=0.7, label='Gaze speed')
                    ax2.set_ylabel('Locomotion and Azimuth', fontsize=8)


                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Firing rate (Hz)')
                ax.set_title(f'Cluster: {cluster_id}\nCorr: {corr:.2f}', fontsize=14)
                ax.set_xlim(0, max(time))
                ax.set_ylim(0, max(true_resp))  # Add some headroom for visualization

                ax.legend(loc='upper right', fontsize=8)
                bopt.style_plot(ax)  # Apply consistent styling

            # Add a super title for the batch of plots
            plt.suptitle(f'{title}', fontsize=12, y=0.98)  # Increase y value to move title up
            plt.subplots_adjust(top=0.7)  # Lower value creates more space at the top
            
            # Adjust layout and show the plots
            plt.tight_layout()
            plt.show()

            plt.close()
    return corrs


def compare_models(model1_path, model2_path, signal1, signal2, model1_name, model2_name, 
                  test_loader1, test_loader2, device, plot1=False, plot2=False):
    """
    Compare two models by plotting their correlations in a scatter plot.
    
    Parameters:
    -----------
    model1_path : str
        Path to the first model
    model2_path : str
        Path to the second model
    model1_name : str
        Name to display for the first model
    model2_name : str
        Name to display for the second model
    test_loader1 : DataLoader
        DataLoader for the first model
    test_loader2 : DataLoader
        DataLoader for the second model
    which_clusters : list
        List of cluster IDs to analyze
    device : torch.device
        Device to run the model on (CPU or GPU)
    
    Returns:
    --------
    tuple
        (corrs_model1, corrs_model2, plt.figure)
    """
    
    # Load models
    print(f"Loading {model1_name} from {model1_path}")
    model1 = torch.load(model1_path, map_location=device)
    print(f"Loading {model2_name} from {model2_path}")
    model2 = torch.load(model2_path, map_location=device)

    # load signals from test loader
    model1.eval().to(device)
    model2.eval().to(device)
    
    # Use the passed test loaders or ensure they're provided
    if test_loader1 is None or test_loader2 is None:
        raise ValueError("Both test_loader1 and test_loader2 must be provided")
    
    # Get correlations for each model
    print(f"Getting correlations for {model1_name}")
    corrs_model1 = eval_model(model1_path, test_loader1, device=device, title=model1_name, plot=plot1, signal=signal1)
    print(f"Getting correlations for {model2_name}")
    corrs_model2 = eval_model(model2_path, test_loader2, device=device, title=model2_name, plot=plot2, signal=signal2)
    
    # Create comparison plot
    fig = plt.figure(figsize=(10, 10))
    
    plt.scatter(list(corrs_model1.values()), list(corrs_model2.values()), alpha=1, color='black')

    plt.plot([0, 1], [0, 1], 'r:', label='Unity line')
    plt.gca().set_aspect('equal', adjustable='box')
    
    bopt.style_plot(plt.gca())
    
    # Find the min and max values for the axes
    plt.gca().set_xlim(-0.2, 1)
    plt.gca().set_ylim(-0.2, 1)
    
    plt.xlabel(f'{model1_name} Correlations')
    plt.ylabel(f'{model2_name} Correlations')
    plt.title(f'Model Comparison: {model1_name} vs {model2_name}', fontsize=18)
    
    common_keys = set(corrs_model1.keys()).intersection(corrs_model2.keys())

    corrs_model1_values = np.array([corrs_model1[k] for k in common_keys])
    corrs_model2_values = np.array([corrs_model2[k] for k in common_keys])

    better_model2 = np.sum(corrs_model2_values > corrs_model1_values)

    total_cells = len(corrs_model2)
    plt.text(0.05, 0.95,
             f'Cells better with {model2_name}: {better_model2}/{total_cells}\n' +
             f'{model1_name} mean corr: {np.mean(list(corrs_model1.values())):.3f}\n' +
             f'{model2_name} mean corr: {np.mean(list(corrs_model2.values())):.3f}',
             transform=plt.gca().transAxes,
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

    return corrs_model1, corrs_model2, fig
    
            
