import torch
import scipy.signal
import numpy as np

import torch
import subprocess
import re


def cuda_init():
    torch.cuda.empty_cache()
    device = get_free_device()
    return device


def get_free_device(threshold=1000):
    # Run nvidia-smi command and get the output
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ],
                            stdout=subprocess.PIPE,
                            text=True)
    # Parse the output
    memory_usage = [int(x) for x in result.stdout.strip().split('\n')]
    [
        print('--- device:{} has {}'.format(i, x))
        for i, x in enumerate(memory_usage)
    ]

    # Find the GPU with the lowest memory usage
    min_memory = min(memory_usage)
    gpu_index = memory_usage.index(min_memory)

    device = 'cuda:{}'.format(gpu_index)
    print('Selected device: {}'.format(device))
    return device


def estfr(bspk, time, sigma=0.01):
    """
    Estimate the instantaneous firing rates from binned spike counts.
    Parameters
    ----------
    bspk : array_like
        Array of binned spike counts (e.g. from binspikes)
    time : array_like
        Array of time points corresponding to bins
    sigma : float, optional
        The width of the Gaussian filter, in seconds (Default: 0.01 seconds)
    Returns
    -------
    rates : array_like
        Array of estimated instantaneous firing rate
    """
    # estimate the time resolution
    dt = float(np.mean(np.diff(time)))

    # Construct Gaussian filter, make sure it is normalized
    tau = np.arange(-5 * sigma, 5 * sigma, dt)
    filt = np.exp(-0.5 * (tau / sigma)**2)
    filt = filt / np.sum(filt)
    size = int(np.round(filt.size / 2))

    # Filter  binned spike times
    return (scipy.signal.fftconvolve(filt, bspk,
                                     mode="full")[size:size + time.size] / dt)
