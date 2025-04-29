import os

import numpy as np
import torch
from experiment import Experiment
from run import Run

# Get Path
path = os.path.dirname(os.getcwd())

# Create folders for experiments and plots
if not os.path.exists(os.path.join(path, "experiments")):
    os.makedirs(os.path.join(path, "experiments"))
if not os.path.exists(os.path.join(path, "plots")):
    os.makedirs(os.path.join(path, "plots"))

# Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### Set parameters ###
n_x = 10 # number of equispaced samples at each time step
n_t = 10 # number of time steps
c_x = 100 # factor of increased resolution for PDE solution
c_t = 100 # factor of increased resolution for PDE solution
r = 0.001 # thermal diffusivity * time_increment / 2 / space_increment**2
u_min = 0 # lower limit for initial values
u_max = 1 # upper limit for initial values
num_epochs = 10000
model = "LSTM" # LSTM, GILR, or Attention
batch_size = "full"
plot_only = False  # change to True if you want to plot existing experimental results, assuming experiment pkl file already exists
lstm_size = 5 # size of RNN hidden dimension
fnn_size_range = 10 * np.arange(10) + 11 # size of FNN hidden dimension, for LSTM
# fnn_size_range = 10 * np.arange(10) + 15 # for GILR
# fnn_size_range = 10 * np.arange(10) + 12 # for Attention
num_samples_range = 10 * (np.arange(10) + 1) # number of data points is num_samples * (n_t - 1)

runs = []
# Create runs
for num_samples in num_samples_range:
    for fnn_size in fnn_size_range:
        run = Run(fnn_size=fnn_size, lstm_size=lstm_size, num_samples = num_samples)
        runs.append(run)

# Run experiment
ex = Experiment(
    n_x=n_x,
    n_t=n_t,
    c_x=c_x,
    c_t=c_t,
    r=r,
    u_min=u_min,
    u_max=u_max,
    num_samples = np.max(num_samples_range),
    num_epochs=num_epochs,
    model=model,
    batch_size=batch_size,
    runs=runs,
)
ex.run_experiment(plot_only=plot_only, path=path, device=device)
