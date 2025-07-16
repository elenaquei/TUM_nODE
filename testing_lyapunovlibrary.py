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

# Definition of a continuous dynamical system, here Lorenz63.
sigma = 10.
rho = 28.
beta = 8./3.
x0 = np.array([1.5, -1.5, 20.])
t0 = 0.
dt = 1e-2

def f(x,t):
    res = np.zeros_like(x)
    res[0] = sigma*(x[1] - x[0])
    res[1] = x[0]*(rho - x[2]) - x[1]
    res[2] = x[0]*x[1] - beta*x[2]
    return res

def jac(x,t):
    res = np.zeros((x.shape[0], x.shape[0]))
    res[0,0], res[0,1] = -sigma, sigma
    res[1,0], res[1,1], res[1,2] = rho - x[2], -1., -x[0]
    res[2,0], res[2,1], res[2,2] = x[1], x[0], -beta
    return res

Lorenz63 = ContinuousDS(x0, t0, f, jac, dt)
Lorenz63.forward(10**2, False)

mLCE, history = mLCE(Lorenz63, 0, 10**2, True)
print(mLCE)


# %%
# Computation of LCE
LCE, history = LCE(Lorenz63, 3, 0, 10**6, True)

# Plot of LCE
plt.figure(figsize = (10,6))
plt.plot(history[:5000])
plt.xlabel("Number of time steps")
plt.ylabel("LCE")
plt.title("Evolution of the LCE for the first 5000 time steps")
plt.show()

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE, CLV, ADJ


Lorenz63 = ContinuousDS(x0, t0, f, jac, dt)
Lorenz63.forward(10**3, False)

mLCE63, history = mLCE(Lorenz63, 0, 10**3, True)
print(mLCE63)

# %%
print('short computation - high variability')
mLCE63, history = mLCE(Lorenz63, 0, 10**3, True)
print(mLCE63)
mLCE63, history = mLCE(Lorenz63, 0, 10**3, True)
print(mLCE63)
mLCE63, history = mLCE(Lorenz63, 0, 10**3, True)
print(mLCE63)

# %%
print('long computation - low variability')
mLCE63, history = mLCE(Lorenz63, 0, 10**5, True)
print(mLCE63)
mLCE63, history = mLCE(Lorenz63, 0, 10**5, True)
print(mLCE63)
mLCE63, history = mLCE(Lorenz63, 0, 10**5, True)
print(mLCE63)


# %%
def rossler():
    a = 0.1
    b = 0.1
    c = 14
    def func(x_func, t):
        #print(t)
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        ross = np.array([-y-z, x+a*y, b+z*(x-c)])
        return ross
    def derivative(x_func, t):
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        return np.array([[0, -1, -1],[1, a, 0],[z, 0, x-c]])
    return func, derivative

x0 = np.array([1.,2.,3.])
t_0 = 0
dt = 0.05
ross_func, ross_der = rossler()
Rossler = ContinuousDS(x0, t0, ross_func, ross_der, dt)



# %%
LCE_rossler, history = LCE(Rossler, 3, 0, 10**5, True)
print(LCE_rossler)
print('VS 0.0714, 0, -5.3943')


# %%
def compute_orbit(dynamical_system, n_steps):
    y = np.zeros([dynamical_system.dim, n_steps])
    for i in range(n_steps):
        dynamical_system.next()
        #print(dynamical_system.x)
        y[:,i] = dynamical_system.x
    return y


# %%
Rossler = ContinuousDS(x0, t0, ross_func, ross_der, dt)
y = compute_orbit(Rossler, 1000)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(y[0,:], y[1,:], y[2,:], '-')
plt.show()

# %% [markdown]
#

# %%

# %%

# %%

# %% [markdown]
#

# %%

# %%

# %% [markdown]
#

# %%



# %%

# %%

# %%

# %%

# %%
