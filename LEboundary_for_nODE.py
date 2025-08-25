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
import matplotlib.pyplot as plt
from LE_with_Jacobian import LEs
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
# Import of the model dynamics that describe the neural ODE
# The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.nODE import nODE, classification_levelsets
from minimal_neuralODE import build_neuralODE
from plots.plots import heat_map
from models.training import create_dataloader, easyTrainer, visualize_dataloader


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
def define_anode():
    torch.backends.cudnn.deterministic = True
    # seed = np.random.randint(1, 200)
    seed = 56
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    data_noise = 0.05
    plotlim = [-3, 3]

    # T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
    T = 1
    non_linearity = 'tanh'  # 'relu' #
    architecture = 'outside_weights'  # inside_weights both

    num_epochs = 100  # number of optimization runs in which the dataset is used for gradient decent
    ODE_dim = 2
    n_layers = 4

    dataloader_train, dataloader_test = create_dataloader('moons', noise=data_noise, plotlim=plotlim, random_state=seed,
                                                          label='vector')

    anode = nODE(ODE_dim, n_layers, architecture=architecture, time_interval=[0, T], non_linearity=non_linearity,
                 first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)

    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)
    trained = True  # tested the training
    if trained:
        trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=1)
        trainer_anode.train(dataloader_train, num_epochs)
    anode.info()
    return anode, dataloader_test

anode, data = define_anode()

# %% [markdown]
# How to reload for Jupyter

# %%
import importlib
import models.nODE
importlib.reload(models.nODE) # Reload the module
from models.nODE import nODE

# %%
def LE_with_Jac(x0):
    le = LEs(torch.from_numpy(x0).float(), anode).max().detach().numpy()
    return le


# %%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
visualize_dataloader(data, label = 'vector', plotlim = [-3, 3])
computed_LE = heat_map(15, LE_with_Jac, savefile ='LE_heatmap_Jacobian.png')


# %%
def decision_distance(x0):
    y, _ = anode(torch.from_numpy(x0).float())
    return np.abs(y[0].detach().numpy())

def error_margin(x0):
    LE_x0 = LE_with_Jac(x0)
    error = 1/LE_x0 * np.log(decision_distance(x0))
    return error

def plot_ball(ax, center, radius, color):
    t = np.linspace(0, 2*np.pi, 30)
    x = center[0] + np.sin(t) * radius
    y = center[1] + np.cos(t) * radius
    ax.plot(x, y, color=color)
    return

def plot_error_balls(anode, n_points):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.random.uniform(-3,3, size = [n_points, 2])
    for x in X:
        y, _ = anode(torch.from_numpy(x).float())
        ball_size = error_margin(x)
        print(f'point {x}, error margin = {ball_size}')
        if y[0] > 0:
            ax.plot(x[0], x[1], 'ob')
            plot_ball(ax, x, ball_size, 'b')
        else:
            ax.plot(x[0], x[1], 'or')
            plot_ball(ax, x, ball_size, 'r')

plot_error_balls(anode, 20)
plt.show()
