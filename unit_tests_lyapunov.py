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
# # Tests on Lyapunov exponents, using the Lyapunov toolbox
#
# Background: there seems to still be a bug in there, so this file is build to find bugs in the Lyapunov Exponent algorithm, specifically local_FTLE
# At the same time, the other function MLE is rougher but produces a more reliable estimate. It could be used as benchmark.
#

# %%
from Lyapunov_toolbox import local_FTLE
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt

# %%
import time

# testing ODE with different MLE in phase space
def multiple_exponents_ODE2D():
    def func(x, t):
        if t<1/2 and np.linalg.norm(x-np.array([1,1]))<1/2:
            return -x
        else:
            return 2*x
    def der(x,t):
        if t<1/2 and np.linalg.norm(x-np.array([1,1]))<1/2:
            return -np.eye(2)
        else:
            return 2*np.eye(2)
    return func, der


func, der = multiple_exponents_ODE2D()

x_amount = 5
integration_time = 10
dt = 0.1

x = np.linspace(-2,2,x_amount)
y = np.linspace(-2,2,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

lyap_small_exp = np.zeros(x_amount**2)
start_time = time.time()
for i in range(x_amount**2):
    lyap_small_exp[i] = np.max(local_FTLE(func, XY[:,i], 1, 0.1, der))
    iteration_time = time.time() - start_time
    if np.mod(i, 10) == 0:
        print(i+1,' out of ', x_amount**2, ' after ', iteration_time)
    # print(lyap[i])
    #break
lyap_small_exp = np.reshape(lyap_small_exp, (x_amount,x_amount))
print(lyap_small_exp)

# Create heatmap using imshow
from IPython.display import Image

file_name = 'MLE_analytic_2DExp.png'

anodeimg = plt.imshow(np.rot90(lyap_small_exp), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
vmin, vmax = anodeimg.get_clim()
plt.colorbar()  # Show color scale
plt.savefig(file_name, bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
plt.close()

img1 = Image(file_name, width = 400)
display(img1)


# %%
## the simplest example: 1D linear 

def linear1D():
    def func(x_func, t):
        #print(t)
        return x_func
    def derivative(x_func, t):
        return np.array([[1.0]])
    return func, derivative


# %%
## a little harder test case : the Lorentz system
def lorenz():
    sigma = 10
    beta = 8/3
    rho = 28
    def func(x_func, t):
        #print(t)
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        lor = np.array([sigma*(y-x), x*(rho - z)-y, x*y - beta*z])
        return lor
    def derivative(x_func, t):
        x = x_func[0]
        y = x_func[1]
        z = x_func[2]
        return np.array([[-sigma, sigma, 0],[rho-z, -1, -x],[y, x, -beta]])
    return func, derivative


# %%
## a little harder test case: the Roessler attractor
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


# %%
import numpy as np
import scipy

print('\nResults for 1D linear')
lin1D_func, lin1D_der = linear1D()
print('Refined implementation \n', local_FTLE(lin1D_func, np.array([1]), 60, 0.1, lin1D_der), 'VS 1\n\n') 

test = True
if test:
    lor_func, lor_der = lorenz()
    print('Results for Lorentz')
    print('Refined implementation \n', local_FTLE(lor_func, np.array([1,2,3]), 100, 0.1, lor_der), 'VS 0.90')
    
    print('\nResults for Rossler show convergence over time')
    ross_func, ross_der = rossler()
    print('Refined implementation t=100\n', local_FTLE(ross_func, np.array([1,2,3]), 100, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
    print('Refined implementation t=500\n', local_FTLE(ross_func, np.array([1,2,3]), 500, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
    print('Refined implementation t=1500\n', local_FTLE(ross_func, np.array([1,2,3]), 1500, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')

# %%
print('Refined implementation \n', local_FTLE(lor_func, np.array([1,2,3]), 1000, 0.0001, lor_der), 'VS 0.90')

# %%
import time

# testing ODE with different MLE in phase space
def weirdODE1D():
    def func(x, t):
        if x<0:
            return -x # contracting
        else:
            return x # expanding
    def der(x,t):
        if x<0:
            return np.array([[-1]]) # contracting
        else:
            return np.array([[1]]) # expanding
    return func, der


def weirdODE2D():
    def func(x, t):
        return np.abs(x) # expanding
    def der(x,t):
        return np.diagflat(np.sign(x)) # expanding
    return func, der



weird1D_func, weird1D_der = weirdODE1D()
local_FTLE(weird1D_func, np.array([1]), 60, 0.1, weird1D_der)

x_amount = 10
integration_time = 10
dt = 0.1

x = np.linspace(-2,2,x_amount)

lyap_weird = np.zeros(x_amount)
start_time = time.time()
for i in range(x_amount):
    lyap_weird[i] = np.max(local_FTLE(weird1D_func, np.array([x[i]]), 20, 0.1, weird1D_der))
    iteration_time = time.time() - start_time
    if np.mod(i, 10) == 0:
        print(i+1,' out of ', x_amount, ' after ', iteration_time)

print(lyap_weird)
