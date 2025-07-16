'''
a selection of ways to compute the Maximal Lyapunov Exponent (MLE) and connected topics.


Main functions:
    local_FTLE(node, x, integration_time, dt = 0.5, der = torch.autograd.functional.jacobian):
          input: either a function or a neuralODE, an initial value for integration, an integration time, and optionally
          a time step and the derivative of the first argument. 
          output: a double, indicating the computed MLE, using integration
          of the variational equation + numerical stabilisation techniques
             needs: linear_dynamics, the adds the variational equation to an ODE
    MLE(traj, t, eps = 0.1)
        input: trajectories of size (x_amount,x_amount,x_dim,time_dim), each trajectory has initial value x_0 = traj[i,j,:,0]
        output: MLE for each trajectory as averaged LE for all initial values that is eps close to the initial value, 
        size (x_amount,x_amount,t_dim)
            needs: le, computing the exponential expansion between two points after time t has passed
'''

import numpy as np
import scipy
import torch
from models.neural_odes import NeuralODE
import warnings


def linear_dynamics(node, derivative):
    '''
    Takes as input a nODE and return the augmented (flattened) linear system (f(x), Df(x)Y) as a 
    function of (x,Y) and t (flattened)
    '''
    if isinstance(node, NeuralODE):
        f_x = node.flow.dynamics.forward
        isnode = True
    else:
        f_x = node
        isnode = False
    
    def composed_lin_ODE(x_and_Y,t):
        '''
        Build the extended right hand side of the augmented (flattened) linear system (f(x), Df(x)Y)
        '''
        size_x = int((-1 + np.sqrt(1 + 4 * x_and_Y.size))/2)
        x = x_and_Y[:size_x]
        Y = np.reshape(x_and_Y[size_x:], [size_x, size_x])
        # print('Y = ', Y)
        if isnode:
            x_torch = torch.from_numpy(x).type(torch.float32)
            x_torch.requires_grad=True
        else:
            x_torch = x
        if isnode:
            f_x_torch = f_x(t, x_torch)
            f_x_torch = f_x_torch.detach().numpy()
        else:
            f_x_torch = f_x(x_torch, t)
        
        Df_x = derivative(x_torch, t)
        
        if isnode: 
            Df_x = Df_x.numpy()
        
        Dfx_Y = np.matmul(Df_x, Y).flatten()
        rhs = np.concatenate((f_x_torch, Dfx_Y))
        return rhs
    
    return composed_lin_ODE


def local_FTLE(node, x, integration_time, dt = 0.5, der = torch.autograd.functional.jacobian):
    '''
    Full Maximal Lyapunov exponent computation, based on the linear_dynamics function
    '''
    def numpy_rhs_from_node():
        def func(x,t):
            x_torch = torch.from_numpy(x).type(torch.float32)
            x_torch.requires_grad=True
            f_x_torch = node.flow.dynamics.forward(t, x_torch)
            f_x_torch = f_x_torch.detach().numpy()
            return f_x_torch
        def der(x,t):
            x_torch = torch.from_numpy(x).type(torch.float32)
            x_torch.requires_grad=True
            f_x_t = lambda x: node.flow.dynamics.forward(t, x)
            Df_x = torch.autograd.functional.jacobian(f_x_t, x_torch) # symbolical (I think)
            Dfx = Df_x.numpy()
            return Dfx
        return func, der
    
    if isinstance(node, NeuralODE):
        func, der = numpy_rhs_from_node()
        #der = torch.autograd.functional.jacobian
    else:
        func = node
    
    # set up of initial values
    Jac = np.identity(x.size)
    Q_t = np.identity(x.size)
    x_and_Jac_t = np.concatenate((x, Jac.flatten()))
    L = np.zeros(x.size)

    #selection functionalities
    select_Jac_int = lambda mat: np.reshape(mat[x.size:], [x.size, x.size])
    select_x_int = lambda mat: mat[:x.size]
    
    warnings.warn("only one iteration to test results")
    
    iters = 1  # number of iterations done
    length_iter = integration_time/iters # length of each iteration
    start_time = 0
    
    for i in range(iters):
        time_array_iter = start_time + np.array([0, length_iter])
        x_and_Jac_t = scipy.integrate.odeint(linear_dynamics(func, der), x_and_Jac_t, time_array_iter)[-1,:]
        
        Jac_t = select_Jac_int(x_and_Jac_t)
        
        Q_t, R_t = np.linalg.qr(np.matmul(Jac_t, Q_t)) # Qi Ri = Yi Qi-1
        x_and_Jac_t = np.concatenate((select_x_int(x_and_Jac_t), np.identity(x.size).flatten()))
        # for numerical stability, reset Y to the identity

        # lyapunov computation takes place
        L += np.log(np.abs(np.sort(np.diagonal(R_t))))
        
    return L/integration_time


'''
input: two trajectories x(t), y(t) where each has size (x_dim,len_t)
output: maximum lyapunov exponent for each time t, size (1,len_t)
'''
def le(x,y,t):
    d = x - y
    d = torch.linalg.norm(x - y, dim = 0)
    # print(f'{d.shape = }')
    d = d/d[0]
    d = np.log(d)

    d = d/t
    return d



'''
Compute the maximum Lyapunov exponent with initial value tolerance eps
input: 
trajectories of size (x_amount,x_amount,x_dim,time_dim)
each trajectory has initial value x_0 = traj[i,j,:,0]

output:
MLE for each trajectory as averaged LE for all initial values that is eps close to the initial value
output size (x_amount,x_amount,t_dim)
'''
def MLE(traj, t, eps = 0.1):
    x_amount = traj.size(0)
    x_amount, y_amount, x_dim, t_dim = traj.shape
    le_val = torch.zeros((x_amount, y_amount, t_dim)) 
    print(f'{le_val.shape = }')
    for i in range(x_amount):
        for j in range(y_amount):
            count = 0
            for i_comp in range(x_amount):
                for j_comp in range(y_amount):
                    if (torch.norm(traj[i,j,:,0] - traj[i_comp,j_comp,:,0]) < eps and not(i == i_comp and j == j_comp)):
                        count += 1
                        le_val[i,j] += le(traj[i,j,:,:],traj[i_comp,j_comp,:,:],t)
                        print('avg le update with count ',count,' and value ',le_val[i,j,-1])
            if count > 1:
                le_val[i,j,:] = le_val[i,j,:]/count
    return le_val
