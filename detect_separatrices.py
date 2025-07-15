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
trained = True  # tested the training
if trained:
    trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=1)
    trainer_anode.train(dataloader_train, num_epochs)


# %%
def vector_field(anode, t_0):
    x = np.arange(-3,3,0.3)
    y = np.arange(-3,3,0.3)
    epsilon = 0.1
    for xi in x:
        for yi in y:
            velocity = anode.right_hand_side(torch.Tensor([t_0]), torch.Tensor([xi, yi]).float()).detach()
            #plt.plot(np.array([xi, xi+epsilon * velocity[0]]), np.array([yi, yi+epsilon * velocity[1]]), color='red')
            plt.arrow(xi, yi, velocity[0]*epsilon, velocity[1]*epsilon, head_width=0.5*epsilon, color='r')


# %%
oneD_iter = 20
n_plots = anode.n_layers + 1
plotlim = [-3, 3]

iter = oneD_iter**2
x_linspace, y_linspace = np.linspace(-3, 3, oneD_iter), np.linspace(-3, 3, oneD_iter)
X, Y = np.meshgrid(x_linspace, y_linspace)
positions = [torch.from_numpy(np.array([X.ravel()[i], Y.ravel()[i]])) for i in range(iter)]

time_points = np.linspace(anode.time_interval[0], anode.time_interval[1], n_plots)
dt = time_points[1] - time_points[0]
for t_0 in time_points[:-1]:
    # t_1 = t_0 + dt
    vector_field(anode, t_0)
    i = anode.layer_selection(torch.Tensor([t_0]))
    W = anode.inside_weights[i].weight.detach().numpy()
    b = anode.inside_weights[i].bias.detach().numpy()
    # print(W, b)
    x0 = np.linalg.solve(W, -b) # find equilibrium
    eigenvals, eigenvects = np.linalg.eig(W)
    #if eigenvals[0] * eigenvals[1] < 0:    # the equilibrium is a saddle
    plt.plot(x0[0], x0[1], '*')
    for index in [0,1]:
        if isinstance(eigenvals[index], np.complex64):
            continue
        #index = 0 + 1 * (eigenvals[0] > 0 ) # select the index associated with the negative eigenvalue
        eig = eigenvects[:, index]
        s_min = max(-3 - x0[0]/eig[0], -3 - x0[1]/eig[1])
        s_max = min(3 - x0[0]/eig[0], 3 - x0[1]/eig[1])
        x_plot = np.array([x0[0] + s_min * eig[0],x0[0] + s_max * eig[0]])
        y_plot = np.array([x0[1] + s_min * eig[1],x0[1] + s_max * eig[1]])
        if eigenvals[index] > 0:
            plt.plot(x_plot, y_plot, 'r')
        else:
            plt.plot(x_plot, y_plot, 'g')
        plt.axis('equal')
    plt.xlim(plotlim)
    plt.ylim(plotlim)
    plt.show()
    print(eigenvals, eigenvects)


# %%
dataloader_test

# %%
isinstance(eigenvals[index], np.complex64)

# %%
import os
import imageio

# Create GIF
def create_gif_from_files(file_names, image_dir, gif_name='gif.gif'):
    images = []
    for name in file_names:
        filename = os.path.join(image_dir, name)
        images.append(imageio.imread(filename))

    imageio.mimsave(gif_name, images, fps=1)

# Create GIF
def create_gif_from_iterator(file_names, ranges, image_dir, gif_name='gif.gif'):
    images = []
    for i in ranges:
        filename = os.path.join(image_dir, name)
        images.append(imageio.imread(filename))

    imageio.mimsave(gif_name, images, fps=1)



# %%
for in_datapoints, out_datapoints in dataloader_test:
    print(in_datapoints[5], out_datapoints[5])
    break


# %%
def plot_test_data(n_points = 10):
    x, y = dataloader_test.dataset.tensors
    plotlim = [-3, 3]
    data_0 = x[y[:, 0] > 0]
    data_1 = x[y[:, 0] < 0]
    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(data_0[:n_points, 0], data_0[:n_points, 1], edgecolor="#333", alpha=0.5)
    plt.scatter(data_1[:n_points, 0], data_1[:n_points, 1], edgecolor="#333", alpha=0.5)
    plt.show()



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
ax.view_init(elev=30, azim=-110+20)
plt.show()

# %%
from torchdiffeq import odeint
import imageio.v2 as imageio

x, y = dataloader_test.dataset.tensors
n_times = 100
time_steps = torch.linspace(anode.time_interval[0], anode.time_interval[1], n_times)
orbits = []
for i in range(100):
    xi = x[i, :]
    yi = y[i, :]
    out = odeint(anode.right_hand_side, xi, time_steps, method='euler')
    orbits.append(out)

images = []
filename = ('temp_for_gif.png')
for j in range(1, n_times):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(100):
        X = orbits[i][:j+1, 0]
        Y = orbits[i][:j+1, 1]
        if y[i][0]>0:
            color = 'b'
        else:
            color = 'orange'
        ax.plot(time_steps.detach().numpy()[:j+1], X.detach().numpy(), Y.detach().numpy(),  color=color, alpha=0.5)
        ax.plot(time_steps.detach().numpy()[j], X.detach().numpy()[j], Y.detach().numpy()[j], 'o',  color=color, alpha=0.5)
    ax.set(xlabel='time', ylabel='X', zlabel='Y', ylim=plotlim, xlim=anode.time_interval, zlim=plotlim)
    ax.view_init(elev=20, azim=-15)
    plt.savefig(filename)
    # plt.show()
    plt.close()
    images.append(imageio.imread(filename))
imageio.mimsave('forward_orbit.gif', images, fps=1)

# %%

# %%
