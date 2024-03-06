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
# # Antisymmetric nODE with perturbation
#
# the goal of this code is to train a nODE with right hand side 
#      $$\tanh( \hat W x + b)$$
# with 
#     $$\hat W = W - W^T + \epsilon P + \gamma I,   $$ 
# where the norm of P is small

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np

# Juptyer magic: For export. Makes the plots size right for the screen 
# %matplotlib inline
# # %config InlineBackend.figure_format = 'retina'

# %config InlineBackend.figure_formats = ['svg'] 


torch.backends.cudnn.deterministic = True
seed = np.random.randint(1,200)
# seed = 56
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(seed)
g = torch.Generator()
g.manual_seed(seed)

# %% [markdown]
#
# # Data preparation

# %%
data_noise = 0.15
plotlim = [-3, 3]
subfolder = 'traj_moons'


from models.training import create_dataloader
dataloader, dataloader_viz = create_dataloader('moons', noise = data_noise, plotlim = plotlim, random_state = seed)


# %% [markdown]
# ## Training and generating level sets

# %%
from models.neural_odes import convolutionalNeuralODE

hidden_dim = 10
data_dim = 2 
augment_dim = 0

num_epochs = 80 #number of optimization runs in which the dataset is used for gradient decent
eps = 0.2
output_dim = 2

#T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
T, num_steps = 1, 1
bound = 0.
fp = False 
cross_entropy = True
turnpike = False

non_linearity = 'tanh' #'relu' #
architecture =  'outside' # 'inside' # 

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cnode = convolutionalNeuralODE(device, data_dim, hidden_dim, output_dim, non_linearity=non_linearity, 
                    architecture=architecture, T=T, time_steps=num_steps)

optimizer_cnode = torch.optim.Adam(cnode.parameters(), lr=1e-3) 


# %%
from models.training import convolutionalTrainer

antisymm_diag = .03
trainer_cnode = convolutionalTrainer(cnode, optimizer_cnode, device, cross_entropy=cross_entropy,
                        verbose = True, antisymm_diag = antisymm_diag)
trainer_cnode.train(dataloader, num_epochs)

# %%
from plots.plots import classification_levelsets
classification_levelsets(cnode)
plt.plot(trainer_cnode.histories['epoch_loss_history'])
plt.xlim(0, len(trainer_cnode.histories['epoch_loss_history']) - 1)
plt.ylim(0)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# %%
from scipy.io import savemat

W = cnode.flow.dynamics.fc2_time[0].weight.detach().numpy()
b = cnode.flow.dynamics.fc2_time[0].bias.detach().numpy()

#print(W,b)
dic = {"W": W, "b": b, "gamma": antisymm_diag}
savemat("coefs_convolutionalNode.mat", dic)

# %%
added_loss = 0
for W in trainer_cnode.model.parameters():
    if len(list(W.size())) == 2 and list(W.size())[0] == list(W.size())[1]:
        added_loss += torch.norm(W + W.T) + torch.linalg.vector_norm(W.diag() - trainer_cnode.antisymm_diag)
        print(torch.linalg.vector_norm(W.diag() - trainer_cnode.antisymm_diag))


# %%
added_loss.item()

# %%
torch.linalg.vector_norm(W.diag() - trainer_cnode.antisymm_diag)

# %%
torch.norm(W + W.T)

# %%
