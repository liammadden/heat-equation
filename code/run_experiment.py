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
n_x = 10
n_t = 10
c_x = 100
c_t = 100
r = .001
u_min = 0
u_max = 1
num_samples = 100
print("Number of data points: " + str(num_samples*(n_t-1)))
num_epochs = 10000
batch_size = "full"
plot_only = False # change to True if you want to plot existing experimental results, assuming experiment pkl file already exists
m_vals = 5*(np.arange(9)+2)

runs = []
# Create runs
for m in m_vals:
    run = Run(m=m)
    runs.append(run)

# Run experiment
ex = Experiment(
    n_x=n_x,
    n_t=n_t,
    c_x=c_x,
    c_t=c_t,
    r = r,
    u_min=u_min,
    u_max=u_max,
    num_samples=num_samples,
    num_epochs=num_epochs,
    batch_size=batch_size,
    runs=runs,
)
ex.run_experiment(plot_only=plot_only, path=path, device=device)