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

# %% [markdown]
# # minimal neuralODE creation
#
# This code is meant to build "any" 2D neuralODE, thus turnign the creation of a neuralODE into a black box

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader
#Import of the model dynamics that describe the neural ODE
#The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.neural_odes import NeuralODE
from models.training import doublebackTrainer, visualize_dataloader


# Juptyer magic: For export. Makes the plots size right for the screen 
# %matplotlib inline
# # %config InlineBackend.figure_format = 'retina'

# %config InlineBackend.figure_formats = ['svg'] 

def build_neuralODE(trained = True, data = False):
    torch.backends.cudnn.deterministic = True
    seed = np.random.randint(1,200)
    seed = 56
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    data_noise = 0.15
    plotlim = [-3, 3]
    subfolder = 'traj_moons'
    
    dataloader_train, dataloader_test = create_dataloader('moons', noise = data_noise, plotlim = plotlim, random_state = seed, label = 'vector')
    
    #for neural ODE based networks the network width is constant. In this example the input is 2 dimensional
    hidden_dim, data_dim = 2, 2 
    augment_dim = 0
    
    #T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
    T, num_steps = 4, 5
    bound = 0.
    fp = False #this recent change made things not work anymore
    cross_entropy = False
    turnpike = False
    
    non_linearity = 'tanh' #'relu' #
    architecture = 'outside' #outside
    
    num_epochs = 100 #number of optimization runs in which the dataset is used for gradient decent
    eps = 0.2
    
    anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=augment_dim, non_linearity=non_linearity, 
                        architecture=architecture, T=T, time_steps=num_steps, fixed_projector=fp, cross_entropy=cross_entropy)
    
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3) 
    if trained:
        trainer_anode = doublebackTrainer(anode, optimizer_anode, device, cross_entropy=cross_entropy, turnpike = turnpike,
                                 bound=bound, fixed_projector=fp, verbose = True, eps_comp = 0.2) 
        trainer_anode.train(dataloader_train, num_epochs)
    if data:
        return anode, dataloader_train
    else:
        return anode

def plot_data(data):
    visualize_dataloader(data, label = 'vector', plotlim = [-3, 3])
    return

if __name__ == "__main__":
    build_neuralODE

# %%
