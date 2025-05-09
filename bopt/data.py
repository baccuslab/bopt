import atexit
import time
import h5py as h5
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class CorticalDataset(Dataset):

    def __init__(self,
                 h5_filepath,
                 series, 
                 num_before=30,
                 num_after=10,
                 stimulus_key="shifted",
                 start_idx=0,
                 end_idx=-1,
                 signals=None,
                 which_clusters=None,
                 normalize_signals=False,
                 grayscale=False,
                 shuffle_signals=False,
                 zero_blinks=True):


        super().__init__()

        self.h5_filepath = h5_filepath
        self.signals = signals if signals is not None else []
        self.series = series
        self.grayscale = grayscale
        self.num_before = num_before
        self.num_after = num_after
        self.window = num_before + num_after 
        self.normalize_signals = normalize_signals
        self.zero_blinks = zero_blinks
        self.shuffle_signals = shuffle_signals
        self.stimulus_key = stimulus_key
        self.which_clusters = which_clusters
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.signal_dict = {}
        for sig in self.signals:
            self.signal_dict[sig] = []

        self.stimulus = []
        self.responses = []
        with h5.File(h5_filepath, "r") as f:
            self.height = f['data'][series[0]]['stimulus'][stimulus_key].shape[1]
            self.width = f['data'][series[0]]['stimulus'][stimulus_key].shape[2]
            self.cluster_ids = list(f['meta']['cluster_ids'][:])
            if self.which_clusters is not None:
                self.cluster_idxs = [
                    self.cluster_ids.index(c) for c in self.which_clusters
                ]
                self.cluster_ids = self.which_clusters
            else:
                self.cluster_idxs = np.arange(len(self.cluster_ids))
            for s in series:
                stim = f['data'][s]['stimulus'][stimulus_key][start_idx:end_idx]
                if self.grayscale:
                    stim = stim.mean(-1)  # grayscale
                if self.zero_blinks:
                    print("Zeroing out blinks in stimulus (at init).")
                    blinks = f["data"][s]["signals"]["blinks"][start_idx:end_idx]
                    blink_mask = blinks > 0
                    stim[blink_mask] = 0
                self.stimulus.append(torch.tensor(stim))
                fr = f['data'][s]['firing_rates'][:, start_idx:end_idx]  # Slice time first
                if self.which_clusters is not None:
                    fr = fr[self.cluster_idxs]  # Then apply cluster selection
                self.responses.append(torch.tensor(fr.T))  # Transpose to (T, C)

                for sig in self.signals:
                    sig_data = f["data"][s]["signals"][sig][start_idx:end_idx]
                    self.signal_dict[sig].append(torch.tensor(sig_data))

        self.stimulus = torch.cat(self.stimulus, dim=0)
        self.responses = torch.cat(self.responses, dim=0)

        for sig in self.signals:
            self.signal_dict[sig] = torch.cat(self.signal_dict[sig], dim=0)
        if self.normalize_signals:
            for sig in self.signals:
                x = self.signal_dict[sig]
                x = (x - x.mean()) / x.std()
                self.signal_dict[sig] = x

        if self.shuffle_signals:
            for sig in self.signals:
                print(f" Shuffling signal '{sig}' — temporal structure will be lost.")
                perm = torch.randperm(self.signal_dict[sig].shape[0])
                self.signal_dict[sig] = self.signal_dict[sig][perm]



    def __len__(self):
        return self.stimulus.shape[0] - self.window

    def __getitem__(self, idx):
        idx_start = idx
        idx_end = idx + self.window
        stimulus = self.stimulus[idx_start:idx_end]  # (T, H, W) → (T, H, W) if already grayscale, or (T, H, W, C) → (T, H, W)
        response = self.responses[idx + self.num_before]  # future response prediction

        signal_list = []
        for sig in self.signals:
            sig_chunk = self.signal_dict[sig][idx +self.num_before]
            signal_list.append(sig_chunk)

        return (stimulus, response, *signal_list)

    def get_example(self):
        return next(iter(self))[:]
    

class SmoothBrainDataset(Dataset):
    def __init__(self, h5_filepath, series, num_before=30, num_after=10, test=False):
        super().__init__()
        self.num_before = num_before
        self.num_after = num_after
        
        self.window = num_before + num_after 
        

        with h5.File(h5_filepath, 'r') as file:
            # grayscale
            self.stimulus = file['data'][series]['stimulus']['static'][:].mean(-1)
            self.pd = file['data'][series]['signals']['nidaq_pd'][:]
            self.pd = self.pd - self.pd.mean()
            self.pd = self.pd / self.pd.std()
            self.pd = self.pd - self.pd.min()


        if test:
            self.stimulus = self.stimulus[-1000:]
            self.pd = self.pd[-1000:]
        else:
            self.stimulus = self.stimulus[:-1000]
            self.pd = self.pd[:-1000]

        self.num_samples = self.stimulus.shape[0] - num_before - num_after

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        x = self.stimulus[idx:idx+self.window]
        x = torch.tensor(x).float()
        z = self.pd[idx+self.num_before]
        z = torch.tensor(z).float()
        return x, z

def create_dataloader(h5_filepath, series, start_idx, end_idx, direction, zero_blinks, grayscale=True,
                     shuffle=True, num_before=25, num_after=5, 
                     batch_size=256, which_clusters=None, num_workers=10):
    """
    Create a data loader for cortical data.
    
    Parameters:
    -----------
    h5_filepath : str
        Path to the H5 file containing the cortical data
    series : list
        List of series/epoch strings to load (e.g. ['series_005/epoch_001'])
    start_idx : int
        Starting index for the dataset
    end_idx : int
        Ending index for the dataset
    direction : str
        Stimulus key (e.g. 'shifted')
    shuffle : bool
        Whether to shuffle the data
    num_before : int
        Number of time bins before spike
    num_after : int
        Number of time bins after spike
    batch_size : int
        Batch size for data loader
    which_clusters : list or None
        List of cluster IDs to include, or None to include all
    num_workers : int
        Number of worker processes for data loading
        
    Returns:
    --------
    loader : DataLoader
        PyTorch DataLoader
    dataset : CorticalDataset
        The dataset used to create the loader
    """
    # Create dataset
    dataset = CorticalDataset(
        h5_filepath,
        series,
        num_before=num_before,
        num_after=num_after,
        start_idx=start_idx,
        end_idx=end_idx,
        stimulus_key=direction,
        grayscale=grayscale,
        normalize_signals=True,
        signals=['elevation', 'azimuth'],
        which_clusters=which_clusters,
        zero_blinks=zero_blinks
    )
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )
    
    return loader, dataset
