# ---
# jupyter:
#   jupytext:
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
# # Definitions and set up

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

data_noise = 0.15
plotlim = [-3, 3]
subfolder = 'traj_moons'

# T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
T = 3
non_linearity = 'tanh'  # 'relu' #
architecture = 'inside_weights'  # outside

num_epochs = 130  # number of optimization runs in which the dataset is used for gradient decent
eps = 0.2

ODE_dim = 2
n_layers = 4


# %%
def heat_plot(func, x_amount = 10, verbose = True):
    x = torch.linspace(-2, 2, x_amount)
    y = torch.linspace(-2, 2, x_amount)
    X, Y = torch.meshgrid(x, y)
    lyap_inf_int_mat = 0*X.detach().numpy()
    for i in range(x_amount):
        for k in range(x_amount):
            x = torch.Tensor([X[i, k], Y[i, k]])
            lyap_inf_int_mat[i, k] = func(x)
        #sys.stdout.write("\033[F")
        if verbose:
            print('Completed :', (i+1)/x_amount)
    
    anodeimg = plt.imshow(np.rot90(lyap_inf_int_mat), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
    vmin, vmax = anodeimg.get_clim()
    plt.colorbar()
    plt.show()


# %%
def input_to_output(input, node, time_interval = torch.tensor([0, T], dtype=torch.float32)):
    return node.forward_integration(input, time_interval)

def LEs(input, node, time_interval = torch.tensor([0, T], dtype=torch.float32)):
    #fix the node so it is just a input to output of the other variable
    input_to_output_lambda = lambda input: input_to_output(input, node, time_interval)
    # Compute the Jacobian matrix
    J = torch.autograd.functional.jacobian(input_to_output_lambda, input)
    
    # Perform Singular Value Decomposition
    U, S, V = torch.svd(J)
    
    # Return the maximum singular value
    return 1/(time_interval[1]-time_interval[0]) * np.log(S)


# %% [markdown]
# # the nODE set up 

# %%
dataloader_train, dataloader_test = create_dataloader('moons', noise=data_noise, plotlim=plotlim, random_state=seed,
                                                      label='vector')

anode = nODE(ODE_dim, n_layers, architecture='inside_weights', time_interval=[0, T], non_linearity='tanh',
             first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)
anode.info()
print(anode)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)
trained = True  # tested the training
if trained:
    trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=1)
    trainer_anode.train(dataloader_train, num_epochs)

    classification_levelsets(anode)

# %%
for x_vector, y_vector in dataloader_test:
    for x, y in zip(x_vector[:100, :], y_vector[:100, :]):
        if all(y == torch.Tensor([-2, 0])):
            color = 'orange'
        else:
            color = 'b'
        trajectory_x = anode.trajectory(x).detach().numpy()
        # plt.plot(trajectory_x[:, 0], trajectory_x[:, 1], '.', color=color)
        plt.plot(trajectory_x[:, 0], trajectory_x[:, 1], color=color)
    break
plt.show()


# %% [markdown]
# # Experiments

# %%
t = torch.Tensor([0.3])
xy = torch.Tensor([1, 2, 3, 4, 5, 6])
anode.lyapunov_system(t, xy)

x = torch.Tensor([1, 2])
lyap_int = np.max(anode.lyapunov_integration(x).detach().numpy())
lyap_approx = anode.lyapunov_approx(x).detach().numpy()
lyap_iterated = np.max(anode.lyapunov_informed_integration(x)[0].detach().numpy())
print('Lyapunov comparison: ', lyap_int, lyap_approx, lyap_iterated)

# %%
func = lambda x : np.max(np.array([anode.lyapunov_approx(x).detach().numpy() for _ in range(10)]))
heat_plot(func)
func = lambda x : np.max(anode.lyapunov_integration(x).detach().numpy())
heat_plot(func)
func = lambda x : np.max(anode.lyapunov_informed_integration(x)[0].detach().numpy())
heat_plot(func)

# %%
func = lambda x : np.min(np.array([anode.lyapunov_approx(x).detach().numpy() for _ in range(10)]))
heat_plot(func)
func = lambda x : np.min(anode.lyapunov_integration(x).detach().numpy())
heat_plot(func)
func = lambda x : np.min(anode.lyapunov_informed_integration(x)[0].detach().numpy())
heat_plot(func)

# %%
func = lambda x : np.max(anode.lyapunov_integration(x).detach().numpy())
heat_plot(func)

# %%
epsilon = 3. # 10 ** -2
func = lambda x : np.max(np.array([anode.lyapunov_approx(x, integration_time = [0, epsilon]).detach().numpy() for _ in range(60)]))
heat_plot(func)

# %%
epsilon = 10 ** - 2
func = lambda x : np.max(np.array([anode.lyapunov_approx(x, integration_time = [0, epsilon]).detach().numpy() for _ in range(60)]))
heat_plot(func)
func = lambda x : np.max(anode.lyapunov_integration(x, integration_time = [0, epsilon]).detach().numpy())
heat_plot(func)

# %%
input1 = torch.tensor([[1, 0]], dtype=torch.float32)
input2 = torch.tensor([0, 1], dtype=torch.float32)
time_interval = torch.tensor([0, T], dtype=torch.float32)

func = lambda x : torch.max(torch.squeeze(LEs(x[None, :], anode)))
func(input1)
heat_plot(func, x_amount = 20)

# %%
ODE_dim = 2
T = 3

seed = 56
data_noise = 0.15
plotlim = [-3, 3]
subfolder = 'traj_moons'
num_epochs = 100

dataloader_train, dataloader_test = create_dataloader('moons', noise=data_noise, plotlim=plotlim, random_state=seed,
                                                      label='vector')

for n_layers in range(1,6):
    print(str('number of layers = ' + str(n_layers)))
    anode = nODE(ODE_dim, n_layers, architecture='inside_weights', time_interval=[0, T], non_linearity='tanh',
                 first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)
    
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)
    
    trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=0)
    trainer_anode.train(dataloader_train, num_epochs)
    classification_levelsets(anode)
    func = lambda x : torch.max(torch.squeeze(LEs(x[None, :], anode)))
    heat_plot(func, x_amount = 10, verbose = False)

# %%
example_T = 4
anode1 = nODE(ODE_dim, 1, architecture='inside_weights', time_interval=[0, example_T], non_linearity='tanh',
                 first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)
    
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)

trainer_anode = easyTrainer(anode1, optimizer_anode, device, verbose=0)
trainer_anode.train(dataloader_train, num_epochs)
classification_levelsets(anode1)


    

# %%
for time in range(1,example_T+1):
    time_interval = torch.tensor([0, time], dtype=torch.float32)
    func = lambda x : torch.max(torch.squeeze(LEs(x[None, :], anode1, time_interval=time_interval)))
    print('time 0 to T = ', time)
    heat_plot(func, x_amount = 20, verbose = False)

# %%
for time in range(1,example_T+1):
    time_interval = torch.tensor([time-1, time], dtype=torch.float32)
    func = lambda x : torch.min(torch.squeeze(LEs(x[None, :], anode1, time_interval=time_interval)))
    print('time ', time_interval[0], ' to T = ', time_interval[1])
    heat_plot(func, x_amount = 20, verbose = False)

# %%
time_interval = torch.tensor([0,1], dtype=torch.float32)
input_to_output_lambda = lambda input: input_to_output(input[None, :], anode1, time_interval)
input1 = torch.Tensor([1.2,1.6])
J = torch.autograd.functional.jacobian(input_to_output_lambda, input1)
print(J)

# %%
time_interval = torch.tensor([0,2], dtype=torch.float32)
input_to_output_lambda = lambda input: input_to_output(input[None, :], anode1, time_interval)
input1 = torch.Tensor([1.2,1.6])
J2 = torch.autograd.functional.jacobian(input_to_output_lambda, input1)
print(J2)

# %%
J.matmul(J)

# %%
U, S, V = torch.svd(J)
print( 1/1* np.log(S) )

# %%
U, S, V = torch.svd(J2)
print( 1/ 2 * np.log(S) )

# %%
U, S, V = torch.svd(J.matmul(J))
print( 1/2 * np.log(S) )

# %%
