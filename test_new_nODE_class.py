import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import numpy as np
from models.training import create_dataloader
# Import of the model dynamics that describe the neural ODE
# The dynamics are based on the torchdiffeq package, that implements ODE solvers in the pytorch setting
from models.nODE import nODE
from models.training import easyTrainer, visualize_dataloader

torch.backends.cudnn.deterministic = True
seed = np.random.randint(1, 200)
seed = 56
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(seed)
g = torch.Generator()
g.manual_seed(seed)

data_noise = 0.15
plotlim = [-3, 3]
subfolder = 'traj_moons'

dataloader_train, dataloader_test = create_dataloader('moons', noise=data_noise, plotlim=plotlim, random_state=seed,
                                                      label='vector')

# T is the end time of the neural ODE evolution, num_steps are the amount of discretization steps for the ODE solver
T = 3
non_linearity = 'tanh'  # 'relu' #
architecture = 'inside_weights'  # outside

num_epochs = 200  # number of optimization runs in which the dataset is used for gradient decent
eps = 0.2

ODE_dim = 2
n_layers = 2

anode = nODE(ODE_dim, n_layers, architecture='both', time_interval=[0, T], non_linearity='tanh',
             first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None)
anode.info()
print(anode)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3)
trained = True  # tested the training
if trained:
    trainer_anode = easyTrainer(anode, optimizer_anode, device, verbose=1)
    trainer_anode.train(dataloader_train, num_epochs)


def plot_data(data):
    visualize_dataloader(data, label='vector', plotlim=[-3, 3])
    return


print(anode)

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

for x_vector, y_vector in dataloader_test:
    for x, y in zip(x_vector[:100, :], y_vector[:100, :]):
        lyap = anode.lyapunov_approx(x).detach().numpy()
        plt.scatter(x.detach().numpy()[0], x.detach().numpy()[1], c=lyap)
    break
plt.show()

x_amount = 2
x = torch.linspace(-2, 2, x_amount)
y = torch.linspace(-2, 2, x_amount)
X, Y = torch.meshgrid(x, y)
lyap_mat = 0*X.detach().numpy()
for i in range(x_amount):
    for k in range(x_amount):
        x = torch.Tensor([X[i, k], Y[i, k]])
        lyap_mat[i, k] = anode.lyapunov_approx(x).detach().numpy()
    print('\nCompleted :', i/x_amount)

anodeimg = plt.imshow(np.rot90(lyap_mat), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()
plt.show()
print('All done!')

t = torch.Tensor([0.3])
xy = torch.Tensor([1, 2, 3, 4, 5, 6])
anode.lyapunov_system(t, xy)

x = torch.Tensor([1, 2])
lyap_int = np.max(anode.lyapunov_integration(x))
lyap_approx = anode.lyapunov_approx(x).detach().numpy()
lyap_iterated = np.max(anode.lyapunov_informed_integration(x)[0].detach().numpy())
print('Lyapunov comparison: ', lyap_int, lyap_approx, lyap_iterated)


x_amount = 20
x = torch.linspace(-2, 2, x_amount)
y = torch.linspace(-2, 2, x_amount)
X, Y = torch.meshgrid(x, y)
lyap_approx_mat = 0*X.detach().numpy()
lyap_int_mat = 0*X.detach().numpy()
lyap_inf_int_mat = 0*X.detach().numpy()
for i in range(x_amount):
    for k in range(x_amount):
        x = torch.Tensor([X[i, k], Y[i, k]])
        lyap_approx_mat[i, k] = np.max(np.array([anode.lyapunov_approx(x).detach().numpy() for _ in range(6)]))
        lyap_int_mat[i, k] = np.max(anode.lyapunov_integration(x))
        lyap_inf_int_mat[i, k] = np.max(anode.lyapunov_informed_integration(x)[0].detach().numpy())
    print('\nCompleted :', i/x_amount)

anodeimg = plt.imshow(np.rot90(lyap_approx_mat), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()
plt.show()

anodeimg = plt.imshow(np.rot90(lyap_int_mat), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()
plt.show()

anodeimg = plt.imshow(np.rot90(lyap_inf_int_mat), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()
plt.show()
print('All done!')