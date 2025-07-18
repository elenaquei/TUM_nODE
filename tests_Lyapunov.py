# Tests on Lyapunov exponents, using the Lyapunov toolbox  

""" Background: there seems to still be a bug in there, so this file is build to find bugs in the Lyapunov Exponent algorithm, specifically local_FTLE
At the same time, the other function MLE is rougher but produces a more reliable estimate. It could be used as benchmark.\n  """

from Lyapunov_toolbox import local_FTLE
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
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
plt.display(img1)
## the simplest example: 1D linear

def linear1D():
  def func(x_func, t):
      #print(t)
      return x_func
  def derivative(x_func, t):
      return np.array([[1.0]])
  return func, derivative
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
import numpy as np
import scipy

print('\\nResults for 1D linear')
lin1D_func, lin1D_der = linear1D()
print('Refined implementation \\n', local_FTLE(lin1D_func, np.array([1]), 60, 0.1, lin1D_der), 'VS 1\\n\\n')

test = True
if test:
  lor_func, lor_der = lorenz()
  print('Results for Lorentz')
  print('Refined implementation \\n', local_FTLE(lor_func, np.array([1,2,3]), 100, 0.1, lor_der), 'VS 0.90')

  print('\\nResults for Rossler show convergence over time')
  ross_func, ross_der = rossler()
  print('Refined implementation t=100\\n', local_FTLE(ross_func, np.array([1,2,3]), 100, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
  print('Refined implementation t=500\\n', local_FTLE(ross_func, np.array([1,2,3]), 500, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
  print('Refined implementation t=1500\\n', local_FTLE(ross_func, np.array([1,2,3]), 1500, 0.1, ross_der), 'VS 0.0714, 0, -5.3943')
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
# Tests involving the NeuralODE code

# First, we need to set up the NeuralODE and test that we can access all that matters
from models.neural_odes import NeuralODE
from models.training import visualize_dataloader
from minimal_neuralODE import build_neuralODE, plot_data
import torch
import numpy as np



nODE, dataloader = build_neuralODE(trained = False, data = True)
# Then, we want to test that the derivative we are retrieving makes sense w.r.t. the right hand side

# We check this by comparign with a numerical derivative taken with the standard finite difference method

def finite_differences(f, t, x, eps = 0.1):
  x_temp = torch.clone(x)
  der = torch.empty(f(t,x).shape[0], x.shape[0])
  for i in range(len(x)):
      x_temp[i] = x[i] + eps
      der[:,i] = (f(t,x_temp) - f(t,x))/eps
      x_temp[i] = x[i]  # change back
  return der
t, x = 0., torch.Tensor([1,2.3])

finite_differences(nODE.flow.dynamics.forward, t, x)
x = torch.Tensor([2.3, 1.4])
y = torch.Tensor([[2.3, 1.4],[12.3, 1.1]])
x_mat = torch.diag_embed(x)

silly_function = lambda t, x : x*8 + y.matmul(x)
derivative = lambda t, x : 8*torch.eye(2) + y

error = torch.empty(7)
for i in range(7):
  print(i,finite_differences(silly_function, t, x, eps=10**-i), '\\n', derivative(t, x))
  error[i] = torch.norm(finite_differences(silly_function, t, x, eps=10**-i) - derivative(t, x))
  print(error[i])
plt.plot(error)
x = torch.Tensor([2.5889,0.245678])

finite_diff = finite_differences(nODE.flow.dynamics.forward, t, x, eps=10**-3).detach()
print('finite diff\\n', finite_diff)

# other option: autograd
x_torch = x
x_torch.requires_grad=True
f_x_t = lambda x: nODE.flow.dynamics.forward(t, x)
autograd = torch.autograd.functional.jacobian(f_x_t, x_torch) # symbolical (I think)
print('autograd\\n', autograd)

analytical = nODE.flow.dynamics.derivative(t, x)
print('analytical\\n', analytical)

print('error autograd VS finite diff', torch.norm(autograd - finite_diff))
print('error autograd VS analytic', torch.norm(autograd - analytical))
# Now the actual FTLE test starts
x_amount = 10
T = 4
integration_time = 20
dt = 0.1
plotlim = [-3, 3]
boundary = np.abs(plotlim[0])

x = np.linspace(-boundary,boundary,x_amount)
y = np.linspace(-boundary,boundary,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

lyap = np.zeros(x_amount**2)
start_time = time.time()
for i in range(x_amount**2):
  lyap[i] = np.max(local_FTLE(nODE, XY[:,i], integration_time, 0.1))
  iteration_time = time.time() - start_time
  if np.mod(i, 10) == 0:
      print(i+1,' out of ', x_amount**2, ' after ', iteration_time)
  # print(lyap[i])
  #break l
lyap2 = np.reshape(lyap, (x_amount,x_amount))
import matplotlib.pyplot as plt

lyap2 = np.reshape(lyap, (x_amount,x_amount))
plt.imshow(lyap2, origin='lower', extent=(-boundary, boundary, -boundary, boundary), cmap='viridis')
plt.colorbar()  # Show color scale

x, y = dataloader.dataset.tensors
plotlim = [-3, 3]
data_0 = x[y[:,0] > 0]
data_1 = x[y[:,0] < 0]
plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333"  ,  alpha = 0.15)
plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333"  , alpha = 0.15)
plt.xlim(plotlim)
plt.ylim(plotlim)
ax = plt.gca()
ax.set_aspect('equal')

plt.savefig('Lyapunov_for_moons')
plt.show()
plt.scatter(XY[0, :], XY[1, :], c = lyap, s = 700, marker = 's')
ax = plt.gca()
ax.set_aspect('equal')
def easy_lyap(x_amount, integration_time, f = nODE, jac = None):
  dt = 0.1

  x = np.linspace(-boundary,boundary,x_amount)
  y = np.linspace(-boundary,boundary,x_amount)
  X, Y = np.meshgrid(x, y)
  XY = np.array([X.flatten(), Y.flatten()])

  lyap = np.zeros(x_amount**2)
  start_time = time.time()
  for i in range(x_amount**2):
      if jac is None:
          lyap[i] = np.max(local_FTLE(f, XY[:,i], integration_time, 0.1))
      else:
          lyap[i] = np.max(local_FTLE(f, XY[:,i], integration_time, 0.1, der = jac))
      iteration_time = time.time() - start_time
  return lyap
# analytical derivatives

f = lambda x, t: nODE.flow.dynamics.forward(t,torch.Tensor(x)).detach()
jac_anal = lambda x, t: nODE.flow.dynamics.derivative(t,torch.Tensor(x)).detach()
def jac_auto(x,t):
  x_torch = torch.from_numpy(x).type(torch.float32)
  x_torch.requires_grad=True
  f_x_t = lambda x: nODE.flow.dynamics.forward(t, x)
  Df_x = torch.autograd.functional.jacobian(f_x_t, x_torch) # symbolical (I think)
  Dfx = Df_x.numpy()
  return Dfx
jac_findiff = lambda x, t: finite_differences(nODE.flow.dynamics.forward, t, torch.Tensor(x))

boundary = 3
T = 4
x_amount = 20
x = np.linspace(-boundary,boundary,x_amount)
y = np.linspace(-boundary,boundary,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

print('analytical derivatives, normal integration time')

lyap = easy_lyap(x_amount, T, f = f, jac = jac_anal)
plt.scatter(XY[0, :], XY[1, :], c = lyap, s = 700, marker = 's')
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig('Lyapunov_for_moons_short_integration_time')
plt.show()

print('analytical, long integration time')

integration_time = 20
lyap = easy_lyap(x_amount, integration_time, f = f, jac = jac_anal)
plt.scatter(XY[0, :], XY[1, :], c = lyap, s = 700, marker = 's')
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
x_amount = 10
x = np.linspace(-boundary,boundary,x_amount)
y = np.linspace(-boundary,boundary,x_amount)
X, Y = np.meshgrid(x, y)
XY = np.array([X.flatten(), Y.flatten()])

print('analytical derivatives, normal integration time')

lyap = easy_lyap(x_amount, T, f = f, jac = jac_anal)
plt.scatter(XY[0, :], XY[1, :], c = lyap, s = 700, marker = 's')
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig('Lyapunov_for_moons_short_integration_time')
plt.show()

print('auto, normal integration time')

integration_time = 20
lyap_auto = easy_lyap(x_amount, T, f = f, jac = jac_auto)
plt.scatter(XY[0, :], XY[1, :], c = lyap_auto, s = 700, marker = 's')
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

print('fin_diff, normal integration time')

integration_time = 20
lyap_findiff = easy_lyap(x_amount, T, f = f, jac = jac_findiff)
plt.scatter(XY[0, :], XY[1, :], c = lyap_findiff, s = 700, marker = 's')
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

# Using the lyapynov library on nODEs
f = lambda x, t: np.array(nODE.flow.dynamics.forward(t,torch.Tensor(x)).detach())
jac_anal = lambda x, t: np.array(nODE.flow.dynamics.derivative(t,torch.Tensor(x)).detach())
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE, CLV, ADJ

# lyap = easy_lyap(x_amount, T, f = f, jac = jac)
x0 = np.array([1,2.3])
t0 = 0
dt = 0.01
nODE_continuousSystem = ContinuousDS(x0, t0, f, jac_anal, dt)
mLCE_x, history = mLCE(nODE_continuousSystem, 0, int(T/dt), True)
print(mLCE_x)
from lyapynov import mLCE, LCE, CLV, ADJ

def mLE(x0, end_time = T):
  t0, dt = 0, 0.01
  nODE_continuousSystem = ContinuousDS(x0, t0, f, jac_anal, dt)
  mle, history = mLCE(nODE_continuousSystem, 0, int(end_time/dt), True)
  return mle, history


mle, history = mLE(np.array([1,2.3]), 100)
plt.plot(history)
def lyapynov_lyap(x_amount, end_time = T):
  x = np.linspace(-boundary,boundary,x_amount)
  y = np.linspace(-boundary,boundary,x_amount)
  X, Y = np.meshgrid(x, y)
  XY = np.array([X.flatten(), Y.flatten()])
  #print(XY)

  lyap = np.zeros(x_amount**2)
  for i in range(x_amount**2):
      x0 = XY[:,i]
      #print(x0)
      lyap[i], h = mLE(x0, end_time)

  #plt.scatter(XY[0, :], XY[1, :], c = lyap, s = 700, marker = 's')
  #ax = plt.gca()
  #ax.set_aspect('equal')
  #plt.show()
  return lyap

for end_time in range(20):
  lyap_test = lyapynov_lyap(10, 1+end_time)
  #overlay(lyap_test)
  plt.show()

lyap_test2 = lyapynov_lyap(13)

def overlay(lyap_vec):
  x_amount = int(np.sqrt(lyap_test.shape[0]))
  lyap2 = np.reshape(lyap_test, (x_amount,x_amount))
  plt.imshow(lyap2, origin='lower', extent=(-boundary, boundary, -boundary, boundary), cmap='viridis')
  plt.colorbar()  # Show color scale

  x, y = dataloader.dataset.tensors
  plotlim = [-3, 3]
  data_0 = x[y[:,0] > 0]
  data_1 = x[y[:,0] < 0]
  plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333" ,  alpha = 0.15)
  plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333"  , alpha = 0.15)
  plt.xlim(plotlim)
  plt.ylim(plotlim)
  ax = plt.gca()
  ax.set_aspect('equal')
overlay(lyap_test)  

lyap_test
nODE.linear_layer.weight  

np.linalg.eig(nODE.linear_layer.weight.detach().numpy())  
