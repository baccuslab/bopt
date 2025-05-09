import copy

from IPython import embed
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers

import scipy.stats
from torch.distributions import normal
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import embed
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class DatasetInfo:
    """
    Class to store information about a dataset
    that is used to construct and train
    subsequent models.
    """

    def __init__(self, dataset):
        ex = dataset.get_example()
        self.x_shape = ex[0].shape
        self.y_shape = ex[1].shape
        self.num_cells = self.y_shape[0]
        self.window = self.x_shape[0]
        self.height = self.x_shape[1]
        self.width = self.x_shape[2]
        self.h5_filepath = dataset.h5_filepath
        self.series = dataset.series
        self.num_before = dataset.num_before
        self.num_after = dataset.num_after
        self.start_idx = dataset.start_idx
        self.end_idx = dataset.end_idx
        self.which_clusters = dataset.cluster_ids


class EncodingModel(nn.Module):
    """
    Base class for encoding models. Contains
    a DatasetInfo object that stores information
    about the dataset used to construct and reconstruct the model.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = DatasetInfo(dataset)
        self.dataset.h5_filepath = None
        self.train_start_time = datetime.datetime.now()

        self.layers = nn.ModuleDict()


class LN(EncodingModel):
    """
    Linear nonlinear model with layer normalization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers['flatten'] = nn.Flatten()
        self.layers['fc'] = nn.Linear(self.dataset.window *
                                      self.dataset.height * self.dataset.width,
                                      self.dataset.num_cells,
                                      bias=True)
        self.layers['softplus'] = nn.Softplus()

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

    def get_filters(self):
        weights = self.layers['fc'].weight.detach().cpu().numpy()
        weights = weights.reshape(self.num_cells, self.window, self.height,
                                  self.width)

        filters = []
        for linear_filter in weights:

            linear_filter -= linear_filter.mean()
            linear_filter /= linear_filter.std()
            filters.append(linear_filter)

        return filters


class CNN2(EncodingModel):

    def __init__(self, num_layers, kernel_sizes, num_channels, padding,
                 mlp_size, **kwargs):
        super().__init__(**kwargs)

        self.padding = padding
        self.mlp_size = mlp_size

        og_temp = torch.randn(1, self.dataset.window, self.dataset.height,
                              self.dataset.width)

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * num_layers
        if type(num_channels) == int:
            num_channels = [num_channels] * num_layers

        for i in range(num_layers):
            if i == 0:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    self.dataset.window,
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)
            else:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    num_channels[i - 1],
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)

            temp = og_temp.clone()
            for name, layer in self.layers.items():
                print(name)
                temp = layer(temp)

            self.layers['layernorm_{}'.format(i)] = nn.LayerNorm(
                [temp.shape[1], temp.shape[2], temp.shape[3]],
                elementwise_affine=False)

            self.layers['dropout' + str(i)] = nn.Dropout(0.1)
            self.layers['nl' + str(i)] = nn.Softplus()

        self.layers['flatten'] = nn.Flatten()

        temp = og_temp.clone()
        for name, layer in self.layers.items():
            temp = layer(temp)

        self.layers['fc1'] = nn.Linear(temp.shape[1], self.mlp_size, bias=True)
        self.layers['out_nl1'] = nn.Softplus()
        self.layers['fc2'] = nn.Linear(self.mlp_size,
                                       self.dataset.num_cells,
                                       bias=True)
        self.layers['output'] = nn.Softplus()

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x


class CNN(EncodingModel):

    def __init__(self, num_layers, kernel_sizes, num_channels, padding,
                 **kwargs):
        super().__init__(**kwargs)

        self.padding = padding

        og_temp = torch.randn(1, self.dataset.window, self.dataset.height,
                              self.dataset.width)

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * num_layers
        if type(num_channels) == int:
            num_channels = [num_channels] * num_layers

        for i in range(num_layers):
            if i == 0:
                if self.padding is not None:
                    self.layers['conv' + str(i)] = nn.Conv2d(
                        self.dataset.window,
                        num_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        padding=padding)
                else:
                    self.layers['conv' + str(i)] = nn.Conv2d(
                        self.dataset.window,
                        num_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=1)
            else:
                if self.padding is not None:
                    self.layers['conv' + str(i)] = nn.Conv2d(
                        num_channels[i - 1],
                        num_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        padding=padding)
                else:
                    self.layers['conv' + str(i)] = nn.Conv2d(
                        num_channels[i - 1],
                        num_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=1)

            temp = og_temp.clone()
            for name, layer in self.layers.items():
                temp = layer(temp)
            self.layers['batchnorm'] = nn.BatchNorm2d(temp.shape[1])
            self.layers['dropout' + str(i)] = nn.Dropout(0.1)
            self.layers['nl' + str(i)] = nn.Softplus()

        self.layers['flatten'] = nn.Flatten()

        temp = og_temp.clone()
        for name, layer in self.layers.items():
            temp = layer(temp)

        self.layers['fc'] = nn.Linear(temp.shape[1],
                                      self.dataset.num_cells,
                                      bias=True)
        self.layers['output'] = nn.Softplus()

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x


class GaussianNoise(nn.Module):

    def __init__(self, std=0.1):
        """
        Additive Gaussian noise layer
        """
        super(GaussianNoise, self).__init__()
        self.std = std
        self.sigma = nn.Parameter(torch.ones(1) * std, requires_grad=False)

    def forward(self, x):
        # If training, add noise
        if self.training:
            noise = self.sigma * torch.randn_like(x)
            return x + noise
        else:
            return x



class SBPDLN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.ModuleDict()
        self.layers['flatten'] = nn.Flatten()
        self.layers['fc1'] = nn.Linear(30*30*30, 1, bias=True)
        nn.init.xavier_uniform_(self.layers['fc1'].weight)
        self.layers['nl1'] = nn.Softplus()

    
    def forward(self, x):
        batch_size = x.size(0)
        for name, layer in self.layers.items():
            x = layer(x)


        return x




class GridSampler(nn.Module):
    """
    Grid sampling module that resamples the input image based on two parameters:
    - scale: Controls zoom (values > 1 zoom in, values < 1 zoom out)
    - rotation: Controls rotation in degrees
    """

    def __init__(self):
        super(GridSampler, self).__init__()

    def forward(self, x, scale, rotation):
        batch_size, _, height, width = x.size()

        # Convert rotation from degrees to radians
        rotation_rad = rotation * np.pi / 180.0

        # Create affine transformation matrix
        # [scale*cos(θ), -scale*sin(θ), 0]
        # [scale*sin(θ), scale*cos(θ), 0]
        theta = torch.zeros(batch_size, 2, 3, device=x.device)
        theta[:, 0, 0] = scale * torch.cos(torch.tensor(rotation_rad))
        theta[:, 0, 1] = -scale * torch.sin(torch.tensor(rotation_rad))
        theta[:, 1, 0] = scale * torch.sin(torch.tensor(rotation_rad))
        theta[:, 1, 1] = scale * torch.cos(torch.tensor(rotation_rad))

        # Create sampling grid and resample
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        sampled_x = F.grid_sample(x, grid, align_corners=False)

        return sampled_x


class LatentGridSampler(nn.Module):
    """
    Grid sampling module that learns to transform input images using two latent variables.
    The latent variables don't have predefined meanings (like scale or rotation),
    but instead the model learns how to use them to distort the input image.
    """

    def __init__(self):
        super(LatentGridSampler, self).__init__()

        # Network to convert 2 latent variables into a 2×3 transformation matrix
        self.transform_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32,
                      6)  # 6 elements for the 2×3 affine transformation matrix
        )

        # Initialize to identity transformation
        self.transform_net[-1].weight.data.zero_()
        self.transform_net[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, latent_var1, latent_var2):
        batch_size = x.size(0)

        # Combine latent variables
        latent_vars = torch.stack([latent_var1, latent_var2], dim=1)

        # Generate transformation matrix from latent variables
        theta_flat = self.transform_net(latent_vars)
        theta = theta_flat.view(batch_size, 2, 3)

        # Create sampling grid and resample
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        sampled_x = F.grid_sample(x, grid, align_corners=False)

        return sampled_x


class NewGridSampler(nn.Module):
    """
    Grid Sampler module that selects a variable field from the input image
    based on two parameters: center_x and center_y.
    
    The sampler creates a sampling grid centered at (center_x, center_y)
    with a size determined by the scale parameter.
    """

    def __init__(self, output_size=(224, 224)):
        super(NewGridSampler, self).__init__()
        self.output_size = output_size

    def forward(self, x, center_x, center_y, scale=0.5):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            center_x (float): x-coordinate of the center of the sampling grid, in range [-1, 1]
            center_y (float): y-coordinate of the center of the sampling grid, in range [-1, 1]
            scale (float): Scale factor for the sampling grid, smaller values zoom in
        
        Returns:
            torch.Tensor: Sampled tensor of shape (batch_size, channels, output_size[0], output_size[1])
        """
        batch_size = x.size(0)

        # Create normalized 2D grid
        h, w = self.output_size
        y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, h),
                                        torch.linspace(-1, 1, w))

        # Adjust grid based on center and scale
        x_grid = (x_grid * scale + center_x).view(1, h, w, 1).expand(
            batch_size, -1, -1, -1)
        y_grid = (y_grid * scale + center_y).view(1, h, w, 1).expand(
            batch_size, -1, -1, -1)

        # Combine grids
        grid = torch.cat([x_grid, y_grid], dim=3).to(x.device)

        # Sample from the input image using the grid
        sampled = F.grid_sample(x,
                                grid,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True)

        return sampled



class GridSampler3(nn.Module):
    """
    Grid Sampler module that learns to select a variable field from the input image
    based on two input parameters.
    
    The model learns to map these two parameters to appropriate sampling coordinates.
    """

    def __init__(self, height, width, factor):
        super(GridSampler3, self).__init__()

        output_size = (height - factor, width - factor)
        self.output_size = output_size
        self.factor = factor

        # Network to learn the mapping from input parameters to sampling coordinates
        self.coordinate_mapper = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * 3)  # Outputs [center_x, center_y, scale]
        )

    def forward(self, x, param1, param2):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            param1 (torch.Tensor): First parameter tensor of shape (batch_size,)
            param2 (torch.Tensor): Second parameter tensor of shape (batch_size,)
        
        Returns:
            torch.Tensor: Sampled tensor of shape (batch_size, channels, output_size[0], output_size[1])
        """

        grid_params = self.coordinate_mapper(
            torch.stack([param1, param2], dim=1))

        shape = [
            x.shape[0], x.shape[1], self.output_size[0] - self.factor,
            self.output_size[1] - self.factor
        ]

        theta = grid_params.view(-1, 2, 3)
        affine_grid = F.affine_grid(theta, shape, align_corners=True)
        sampled = F.grid_sample(x, affine_grid, align_corners=True)
        return sampled


class GS_CNN(EncodingModel):

    def __init__(self, num_layers, kernel_sizes, num_channels, padding,
                 mlp_size, **kwargs):
        super().__init__(**kwargs)

        self.padding = padding
        self.mlp_size = mlp_size

        og_temp = torch.randn(1, self.dataset.window, self.dataset.height,
                              self.dataset.width)

        self.grid_sampler = GridSampler3(self.dataset.height,
                                         self.dataset.width, 10)

        out = self.grid_sampler(og_temp, torch.tensor([0.0]),
                                torch.tensor([0.0]))

        og_temp = torch.randn(1, out.shape[1], out.shape[2], out.shape[3])

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * num_layers
        if type(num_channels) == int:
            num_channels = [num_channels] * num_layers

        for i in range(num_layers):
            if i == 0:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    self.dataset.window,
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)
            else:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    num_channels[i - 1],
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)

            temp = og_temp.clone()
            for name, layer in self.layers.items():
                print(name)
                temp = layer(temp)

            self.layers['layernorm_{}'.format(i)] = nn.LayerNorm(
                [temp.shape[1], temp.shape[2], temp.shape[3]],
                elementwise_affine=False)

            self.layers['dropout' + str(i)] = nn.Dropout(0.1)
            self.layers['nl' + str(i)] = nn.Softplus()

        self.layers['flatten'] = nn.Flatten()

        if self.mlp_size is None:
            self.layers['fc'] = nn.LazyLinear(self.dataset.num_cells,
                                              bias=True)
        else:
            self.layers['fc1'] = nn.LazyLinear(self.mlp_size, bias=True)
            self.layers['out_nl1'] = nn.Softplus()
            self.layers['fc2'] = nn.Linear(self.mlp_size,
                                           self.dataset.num_cells,
                                           bias=True)
        self.layers['output'] = nn.Softplus()

    def forward(self, x, elevation, azimuth):
        x = self.grid_sampler(x, elevation, azimuth)

        for name, layer in self.layers.items():
            x = layer(x)
        return x




class CNNComponent(EncodingModel):
    """The CNN component of the model that processes visual input."""
    
    def __init__(self, num_layers, kernel_sizes, num_channels, padding, input_layernorm, **kwargs):
        super().__init__(**kwargs)
        
        self.padding = padding
        self.input_layernorm = input_layernorm
        
        og_temp = torch.randn(1, self.dataset.window, self.dataset.height, self.dataset.width)
        
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * num_layers
        if type(num_channels) == int:
            num_channels = [num_channels] * num_layers
        
        if input_layernorm:
            self.layers['input_layernorm'] = nn.LayerNorm(
                [self.dataset.window, self.dataset.height, self.dataset.width],
                elementwise_affine=False)
        
        for i in range(num_layers):
            if i == 0:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    self.dataset.window,
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)
            else:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    num_channels[i - 1],
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)
            
            temp = og_temp.clone()
            for name, layer in self.layers.items():
                temp = layer(temp)
            
            self.layers['layernorm' + str(i)] = nn.LayerNorm(
                [temp.shape[1], temp.shape[2], temp.shape[3]],
                elementwise_affine=False)
            self.layers['dropout' + str(i)] = nn.Dropout(0.1)
            if i == num_layers - 1:
                pass
            else:
                self.layers['nl' + str(i)] = nn.Softplus()
        
        self.layers['flatten'] = nn.Flatten()
        
        temp = og_temp.clone()
        for name, layer in self.layers.items():
            temp = layer(temp)
        
        self.layers['fc'] = nn.Linear(temp.shape[1], self.dataset.num_cells, bias=True)
        self.layers['output'] = nn.Softplus()
    
    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

class GainParamModel(nn.Module):

    def __init__(self, model):
        super(GainParamModel, self).__init__()
        self.model = model
        num_cells = self.model.dataset.num_cells
        self.A = nn.Parameter(torch.zeros(num_cells), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(num_cells), requires_grad=True)

    def forward(self, x, signal):
        with torch.no_grad():
            y = self.model(x)

        A_pos = F.softplus(self.A)
        B_pos = F.softplus(self.B)

        v= signal.mean(dim=1)
        v = v.unsqueeze(1)  # Changes shape from [256] → [256, 1]
        A_pos = A_pos.unsqueeze(0)  # [1, 32]
        B_pos = B_pos.unsqueeze(0)  # [1, 32]

        output = y + (A_pos * v * y) + B_pos * v

        output = F.softplus(output)  # This will set any negative values to a smooth positive range

        return output


class LNComponent(EncodingModel):
    """
    Linear-Nonlinear (LN) component that processes input with a single pathway:
    linear transformations followed by a final nonlinearity.
    """
    
    def __init__(self, hidden_dims, activation_fn=nn.Softplus, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.layers = nn.ModuleDict()
        
        # Calculate input dimension from dataset
        input_dim = self.dataset.window * self.dataset.height * self.dataset.width
        
        # Input flattening
        self.layers['flatten'] = nn.Flatten()
        
        # Parse hidden dimensions
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        # Create linear layers
        layer_dims = [input_dim] + hidden_dims
        for i in range(len(layer_dims) - 1):
            self.layers[f'fc{i}'] = nn.Linear(
                layer_dims[i], 
                layer_dims[i+1], 
                bias=True
            )
            self.layers[f'ln{i}'] = nn.LayerNorm(
                layer_dims[i+1],
                elementwise_affine=False
            )
            self.layers[f'dropout{i}'] = nn.Dropout(dropout_rate)
        
        # Output layer
        self.layers['output_fc'] = nn.Linear(layer_dims[-1], self.dataset.num_cells, bias=True)
        self.layers['output_act'] = activation_fn()
    
    def forward(self, x):
        # Initial flattening
        x = self.layers['flatten'](x)
        
        # Process through linear layers (no activations between them)
        for i in range(len([l for l in self.layers if l.startswith('fc')])):
            x = self.layers[f'fc{i}'](x)
            x = self.layers[f'ln{i}'](x)
            x = self.layers[f'dropout{i}'](x)
        
        # Final output with nonlinearity
        x = self.layers['output_fc'](x)
        x = self.layers['output_act'](x)
        
        return x


class PDLN(nn.Module):
    """
    Linear nonlinear model with layer normalization.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.layers['flatten'] = nn.Flatten()
        self.layers['fc'] = nn.LazyLinear(1, 
                                      bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        for name, layer in self.layers.items():
            x = layer(x)
        return x.view(batch_size)


