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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE, CLV, ADJ
viridis = plt.get_cmap('viridis')



# %%
def compute_orbit(dynamical_system, n_steps):
    y = np.zeros([dynamical_system.dim, n_steps])
    for i in range(n_steps):
        dynamical_system.next()
        #print(dynamical_system.x)
        y[:,i] = dynamical_system.x
    return y


# %% [markdown]
# Setting up the nODE to compute the Lyapunov exponent with the new class for Lyapunov exponents (hoping for better stability w.r.t. the in-house code)

# %%
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader
# Import of the model dynamics that describe the neural ODE
# The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.nODE import nODE, classification_levelsets
from models.training import easyTrainer, visualize_dataloader

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
architecture = 'outside_weights'  # inside_weights both

num_epochs = 130  # number of optimization runs in which the dataset is used for gradient decent
eps = 0.4

ODE_dim = 2
n_layers = 4
dataloader_train, dataloader_test = create_dataloader('moons', noise=data_noise, plotlim=plotlim, random_state=seed,
                                                      label='vector')

anode = nODE(ODE_dim, n_layers, architecture=architecture, time_interval=[0, T], non_linearity=non_linearity,
             first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)
anode.info()
print(anode)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)
trained = True  # tested the training
if trained:
    trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=1)
    trainer_anode.train(dataloader_train, num_epochs)

# %%
x0 = np.arange(0,anode.ODE_dim)+ 0.0001
t0 = anode.time_interval[0]
dt = anode.compute_dt()

dynamical_nODE = ContinuousDS(x0, t0,
                              lambda x, t: anode.right_hand_side(torch.Tensor([t]), torch.from_numpy(x).float()).detach().numpy(),
                              lambda x, t : anode.derivative(torch.Tensor([t]), torch.from_numpy(x).float()).detach().numpy(), dt)

# %%
max_iters = int((anode.time_interval[1] - anode.time_interval[0])/dt -1)

LCE_nODE, history = mLCE(dynamical_nODE, 0, max_iters, True)
print(LCE_nODE)

# %% [markdown]
# How to reload for Jupyter

# %%
import importlib
import models.nODE
importlib.reload(models.nODE) # Reload the module
from models.nODE import nODE


# %% [markdown]
# A function to create a heat map once you know how to compute the "heat" item through a numpy function

# %%
def heat_map(n_1Dpoints, lambda_func, savefile = None):
    x = np.linspace(-2, 2, n_1Dpoints)
    y = np.linspace(-2, 2, n_1Dpoints)
    X, Y = np.meshgrid(x, y)
    computed_func = 0*X
    for i in range(n_1Dpoints):
        for k in range(n_1Dpoints):
            x = np.array([X[i, k], Y[i, k]])
            computed_func[i, k] = lambda_func(x)
        print(f'Completed : {i/n_1Dpoints:.2f}')

    anodeimg = plt.imshow(np.rot90(computed_func), origin='upper', extent=(-2, 2, -2, 2), cmap = viridis)
    vmin, vmax = anodeimg.get_clim()
    plt.colorbar()
    if savefile:
        plt.savefig(savefile)
    plt.show()
    return computed_func



# %%
def LE_func(x0):
    t0 = anode.time_interval[0]
    dt = anode.compute_dt()

    dynamical_nODE = ContinuousDS(x0, t0,
                                  lambda x, t: anode.right_hand_side(torch.Tensor([t]),
                                                                     torch.from_numpy(x).float()).detach().numpy(),
                                  lambda x, t: anode.derivative(torch.Tensor([t]),
                                                                torch.from_numpy(x).float()).detach().numpy(), dt)
    max_iters = int((anode.time_interval[1] - anode.time_interval[0]) / dt - 1)

    LCE_nODE = mLCE(dynamical_nODE, 0, max_iters, False)
    return LCE_nODE



# %%
computed_LE = heat_map(30, LE_func, savefile ='LE_heatmap_library.png')


# %% [markdown]
# # longer time integration for autonomous components
#
# We can think that, for every autonomous component of the nODE, we want to compute not short term LE, but medium term (we don't want to compute long term LE to avoid long term averaging). to achieve this, we will introduce a helper function that frezzes the time in the nODE to the wanted layer, thus the LE computation can concentrate on a sngle layer at a time, before averaging them all

# %%
def LE_nODE_fixed_layer(x0, layer):
    t0 = 0
    dt = 0.02

    layer_length = (anode.time_interval[1] - anode.time_interval[0]) / anode.n_layers
    fixed_time = anode.time_interval[0] + layer * layer_length

    dynamical_nODE = ContinuousDS(x0, t0,
                                  lambda x, t: anode.right_hand_side(torch.Tensor([fixed_time]),torch.from_numpy(x).float()).detach().numpy(),
                                  lambda x, t: anode.derivative(torch.Tensor([fixed_time]),
                                        torch.from_numpy(x).float()).detach().numpy(), dt)

    # with this hack, we are not limited to the original maximum number of iterates, but can take longer to ocnverge to the expected LE
    n_iters = 50
    LCE_nODE = mLCE(dynamical_nODE, 0, n_iters, False)
    for i in range(50):
        LCE_nODE = max(LCE_nODE, mLCE(dynamical_nODE, 0, n_iters, False))
        # rerun 5 times and pick the largest
    return LCE_nODE

def averaged_LCE(x0):
    LCE = np.empty(anode.n_layers)
    for i in range(anode.n_layers):
        LCE[i] = LE_nODE_fixed_layer(x0, i)
    averaged_LCE = np.mean(LCE)
    return averaged_LCE


# %%
from plots.plots import vector_field

for j in range(4):
    vector_field(anode, j/4+0.02)
    computed_LE = heat_map(4, lambda x: LE_nODE_fixed_layer(x, j))
    plt.show()

# %%
from plots.plots import plot_all_vectorfields

eigs = plot_all_vectorfields(anode)
print(eigs[0])

# %%
import importlib
import plots.plots
importlib.reload(plots.plots) # Reload the module
from plots.plots import plot_all_vectorfields

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
