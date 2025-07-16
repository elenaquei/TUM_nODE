# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Set up the node package and the standardly used items

# %%
import sys
import torch
from plots.gifs import trajectory_and_vectorfield_gif
from plots.plots import vector_field, plot_all_vectorfields, plot_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader
# Import of the model dynamics that describe the neural ODE
# The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.nODE import nODE, classification_levelsets
from models.training import easyTrainer, visualize_dataloader

# %%
torch.backends.cudnn.deterministic = True
seed = np.random.randint(1, 200)
seed = 56
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
g = torch.Generator()
g.manual_seed(seed)

data_noise = 0.05
plotlim = [-3, 3]
subfolder = 'traj_moons'

# T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
T = 1
non_linearity = 'tanh'  # 'relu' #
architecture = 'outside_weights'  # inside

num_epochs = 130  # number of optimization runs in which the dataset is used for gradient decent
eps = 0.4

ODE_dim = 2
n_layers = 4

# %%
dataloader_train, dataloader_test = create_dataloader('moons', noise=data_noise, plotlim=plotlim, random_state=seed,
                                                      label='vector')

anode = nODE(ODE_dim, n_layers, architecture=architecture, time_interval=[0, T], non_linearity=non_linearity,
             first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)
anode.info()
print(anode)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)
trained = False # tested the training
if trained:
    trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=1)
    trainer_anode.train(dataloader_train, num_epochs)

# %%
eigs = plot_all_vectorfields(anode)
print(eigs)

# %% jupyter={"is_executing": true}
x, y = dataloader_test.dataset.tensors
trajectory_and_vectorfield_gif(anode, x, y)

# %%
plot_dataloader(dataloader_test, n_points=20)

# %%

# %%





# %%

# %%

# %%
from torchdiffeq import odeint

x, y = dataloader_test.dataset.tensors
n_times = 50
time_steps = torch.linspace(anode.time_interval[0], anode.time_interval[1], n_times)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
orbits = []
for i in range(100):
    xi = x[i, :]
    yi = y[i, :]
    out = odeint(anode.right_hand_side, xi, time_steps, method='euler')
    orbits.append(out)
    X = out[:,0]
    Y = out[:,1]
    if yi[0]>0:
        color = 'b'
    else:
        color = 'orange'
    ax.plot(time_steps.detach().numpy(), X.detach().numpy(), Y.detach().numpy(),  color=color, alpha=0.5)
    ax.set(xlabel='time', ylabel='X', zlabel='Y')
plt.show()

# %%

# %%

# %%
