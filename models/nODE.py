#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
further adapted by Elena Queirolo
"""
##------------#
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from warnings import warn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from matplotlib.colors import to_rgb
import imageio

from matplotlib.colors import LinearSegmentedColormap
import os

# from adjoint_neural_ode import adj_Dynamics

# odeint Returns:
#         y: Tensor, where the first dimension corresponds to different
#             time points. Contains the solved value of y for each desired time point in
#             `t`, with the initial value `y0` being the first element along the first
#             dimension.


MAX_NUM_STEPS = 1000


def tanh_prime(input):
    # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
    return 1 - torch.tanh(input) * torch.tanh(
        input)


def identity_prime(input):
    # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions
    return 1 - 0*input

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class Tanh_Prime(nn.Module):
    '''
    Applies tanh'(x) function element-wise:

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''

    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()  # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return tanh_prime(input)  # simply apply already implemented SiLU


# Useful dicos:
activations = {'tanh': nn.Tanh(),
               'relu': nn.ReLU(),
               'sigmoid': nn.Sigmoid(),
               'leakyrelu': nn.LeakyReLU(negative_slope=0.25, inplace=True),
               'tanh_prime': tanh_prime,
               'identity' : nn.Identity()
               }
derivatives_activations = {'tanh': tanh_prime,
                           'identity': identity_prime
                           }
architectures = {'inside_weights': -1, 'outside_weights': 0, 'both': 1}


class nODE(nn.Module):

    def __init__(self, ODE_dim, n_layers, architecture='inside_weights', time_interval=None, non_linearity='tanh',
                 first_layer_bool=False, last_layer_bool=False, start_dim=None, end_dim=None, interpolation = None):
        super(nODE, self).__init__()
        if time_interval is None:
            time_interval = [0, 1]
        self.ODE_dim = ODE_dim
        self.n_layers = n_layers
        self.architecture = architecture
        self.time_interval = time_interval
        self.non_linearity = activations[non_linearity]
        self.non_linear_derivative = derivatives_activations[non_linearity]
        self.first_layer_bool = first_layer_bool
        self.first_layer = None
        self.last_layer_bool = last_layer_bool
        self.last_layer = None
        if first_layer_bool and start_dim is None:
            start_dim = ODE_dim

        if last_layer_bool and start_dim is None:
            end_dim = ODE_dim
        self.start_dim = start_dim  # this should describe the amount of piecewise constant parameters exist. i.e. T/num_params
        self.end_dim = end_dim
        self.inside_weights = None
        self.outside_weights = None
        self.setup_weights()
        self.interpolation = interpolation
        return

    def setup_weights(self):
        blocks_inside = [nn.Linear(self.ODE_dim, self.ODE_dim) for _ in range(self.n_layers)]
        self.inside_weights = nn.Sequential(*blocks_inside)
        blocks_outside = [nn.Linear(self.ODE_dim, self.ODE_dim) for _ in range(self.n_layers)]
        self.outside_weights = nn.Sequential(*blocks_outside)
        if self.first_layer_bool:
            self.first_layer = nn.Linear(self.start_dim, self.ODE_dim)
        if self.last_layer_bool:
            self.last_layer = nn.Linear(self.ODE_dim, self.end_dim)
        return

    def layer_selection(self, t):
        if t < self.time_interval[0]:
            return 0
        if t > self.time_interval[1]:
            return self.n_layers - 1
        time_length = self.time_interval[1] - self.time_interval[0]
        dt = time_length / self.n_layers
        return int(torch.floor(t / dt))

    def layers_and_interpolation(self, t):
        if t < self.time_interval[0]:
            return 0
        if t > self.time_interval[1]:
            return self.n_layers - 1
        time_length = self.time_interval[1] - self.time_interval[0]
        dt = time_length / self.n_layers
        time_from_start = t - self.time_interval[0]
        layer_left = int(torch.floor(time_from_start/ dt))
        layer_right = int(torch.floor(time_from_start / dt)) + 1
        interpolation = time_from_start - dt * layer_left
        if layer_right > self.n_layers - 1:
            layer_right = self.n_layers - 1
        return layer_left, layer_right, interpolation

    def right_hand_side(self, t, x):
        if self.interpolation:
            layer_left, layer_right, time = self.layers_and_interpolation(t)
            interpolation = lambda x0, x1 : (1 - time) * x0 + time * x1
            ## it was there for a C0 vectorfield
            if architectures[self.architecture] == 1:  # outside architecture ahs no inside layer
                out = x
            else:
                w1_t0 = self.inside_weights[layer_left].weight
                b1_t0 = self.inside_weights[layer_left].bias
                w1_t1 = self.inside_weights[layer_right].weight
                b1_t1 = self.inside_weights[layer_right].bias

                w1_t = interpolation(w1_t0, w1_t1)
                b1_t = interpolation(b1_t0, b1_t1)
                out = x.matmul(w1_t.t()) + b1_t
            out = self.non_linearity(out)
            if architectures[self.architecture] == 0:  # inside architecture has no outside layer
                out = out
            else:
                w2_t0 = self.outside_weights[layer_left].weight
                b2_t0 = self.outside_weights[layer_left].bias
                w2_t1 = self.outside_weights[layer_right].weight
                b2_t1 = self.outside_weights[layer_right].bias

                w2_t = interpolation(w2_t0, w2_t1)
                b2_t = interpolation(b2_t0, b2_t1)

                out = out.matmul(w2_t.t()) + b2_t
        else:
            layer = self.layer_selection(t)
            if architectures[self.architecture] == 1:  # outside architecture ahs no inside layer
                out = x
            else:
                w1_t = self.inside_weights[layer].weight
                b1_t = self.inside_weights[layer].bias
                out = x.matmul(w1_t.t()) + b1_t
            out = self.non_linearity(out)
            if architectures[self.architecture] == 0:  # inside architecture has no outside layer
                out = out
            else:
                w2_t = self.outside_weights[layer].weight
                b2_t = self.outside_weights[layer].bias

                out = out.matmul(w2_t.t()) + b2_t
        return out

    def derivative(self, t, x):
        """
        The output of the class -> D_xf(x(t), u(t))
        """
        def rowKronecker(x_vector, y_matrix):
            temp = [x_vector[i].detach() * y_matrix[i, :].detach() for i in range(len(x))]
            result = torch.Tensor()
            result = torch.cat(temp, out=result).reshape(y_matrix.shape)
            return result

        k = self.layer_selection(t)
        if architectures[self.architecture] == 0:
            w_t = self.outside_weights[k].weight
            b_t = self.outside_weights[k].bias
            # w(t)\sigma(x(t))+b(t)  inner
            # # #     -> derivative is w(t)\sigma'(x(t))
            out = w_t.matmul(torch.diag(self.non_linear_derivative(x)))
        elif architectures[self.architecture] == -1:
            w_t = self.inside_weights[k].weight
            b_t = self.inside_weights[k].bias
            out = rowKronecker(self.non_linear_derivative(w_t.matmul(x) + b_t), w_t)
        else:
            # w1(t)\sigma(w2(t)x(t)+b2(t))+b1(t) bottle-neck
            # # #     -> derivative is w1(t)\sigma'(w2(t)x(t)+b2(t))\row dy row kronecked product w2(t)
            w1_t = self.inside_weights[k].weight
            b1_t = self.inside_weights[k].bias
            w2_t = self.outside_weights[k].weight
            # b2_t = self.fc3_time[k].bias
            out = rowKronecker(self.non_linear_derivative(w1_t.matmul(x) + b1_t), w1_t)
            out = w2_t.matmul(out)

            # x.matmul(w1_t.t()) is the same as torch.matmul(w1_t,x) simple matrix-vector multiplication
        return out

    def compute_dt(self):
        dt = (self.time_interval[1] - self.time_interval[0]) / (20 * self.n_layers)
        return dt

    def forward(self, x, return_features=False):
        if return_features:
            time_intervals = torch.linspace(self.time_interval[0], self.time_interval[1], 300)
            integration_interval = torch.tensor(time_intervals).float().type_as(x)
        else:
            integration_interval = torch.tensor(self.time_interval).float().type_as(x)
        if self.first_layer_bool:
            x_in = self.first_layer(x)
        else:
            x_in = x
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x_in, integration_interval, method='euler', options={'step_size': dt})
        out = out[1, :, :]
        if self.last_layer_bool:
            x_out = self.last_layer(out)
        else:
            x_out = out
        return x_out

    def forward_integration(self, x, integration_time=None, outer_layers=True):
        if integration_time is None:
            time_intervals = torch.tensor([self.time_interval[0], self.time_interval[1]])
            integration_interval = torch.tensor(time_intervals).clone().float().type_as(x)
        else:
            integration_interval = torch.tensor([integration_time[0], integration_time[1]])
        if self.first_layer_bool and outer_layers:
            x_in = self.first_layer(x)
        else:
            x_in = x
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x_in, integration_interval, method='euler', options={'step_size': dt})
        if len(out.shape) == 3:
            out = out[1, :, :]
        if self.last_layer_bool and outer_layers:
            x_out = self.last_layer(out)
        else:
            x_out = out
        return x_out

    def __str__(self):
        """a __str__ function for readability with print statements"""
        activation_string = [i for i in activations if activations[i] == self.non_linearity][0]
        string = str()
        if architectures[self.architecture] < 1:
            if architectures[self.architecture] == -1:
                string += str(
                    'w(t)' + activation_string + '(x(t))+b(t)    over the interval ' + str(self.time_interval) + ',\n')
                layers = self.inside_weights
            else:
                string += str(
                    activation_string + '(w(t)x(t)+b(t))    over the interval ' + str(self.time_interval) + '\n')
                layers = self.outside_weights
            for k in range(self.n_layers):
                string += str(
                    'W[' + str(k) + '] = ' + str(layers[k].weight.detach().numpy()) + ',        b[' + str(
                        k) + '] = ' + str(layers[k].bias.detach().numpy()) + '\n\n')
        else:
            string += str(
                'w1(t)' + activation_string + '(w2(t)x(t)+b2(t))+b1(t)    over the interval ' + str(self.time_interval) + '\n')
            for k in range(self.n_layers):
                string += str(
                    'W1[' + str(k) + '] = ' + str(self.outside_weights[k].weight.detach().numpy()) + ',        b1[' + str(
                        k) + '] = ' + str(self.outside_weights[k].bias.detach().numpy()) + '\n\n')
                string += str(
                    'W2[' + str(k) + '] = ' + str(self.inside_weights[k].weight.detach().numpy()) + ',        b2[' + str(
                        k) + '] = ' + str(self.inside_weights[k].bias.detach().numpy()) + '\n\n')
        return string

    def layer(self, n):
        if n > self.n_layers:
            raise ValueError('Layer requested does not exist')
        if architectures[self.architecture] == 0:
            return self.inside_weights[n]
        if architectures[self.architecture] == 1:
            return self.outside_weights[n]
        return self.outside_weights[n], self.inside_weights

    def info(self):
        print('Time interval: ', self.time_interval, '\n')
        print('Number of layers: ', self.n_layers, '\n')
        print('Non-linearity: ', self.non_linearity, '\n')
        print('Architecture: ', self.architecture, '\n')
        print('ODE dimension: ', self.ODE_dim, '\n')
        print('First linear layer: ', self.first_layer_bool, '\n')
        if self.first_layer_bool:
            print('Dimension input: ', self.start_dim, '\n')
        print('Last linear layer: ', self.last_layer_bool, '\n')
        if self.last_layer_bool:
            print('Dimension output: ',  self.end_dim, '\n')

    def set_inside_weigths(self, linear_layers):
        self.inside_weights = linear_layers
        return

    def set_outside_weights(self, linear_layers):
        self.outside_weights = linear_layers
        return

    def trajectory(self, x, n_evals=100):
        time_intervals = torch.linspace(self.time_interval[0], self.time_interval[1], n_evals)
        integration_interval = time_intervals.clone().detach().float().type_as(x)

        if self.first_layer_bool:
            x_in = self.first_layer(x)
        else:
            x_in = x
        dt = self.compute_dt()
        out = odeint(self.right_hand_side, x_in, integration_interval, method='euler', options={'step_size': dt})

        return out

    def lyapunov_approx(self, x, eps=10**-4, integration_time=None):
        if len(x.size()) == 1:
            x = x.view([1, x.shape[0]])
        perturbation = torch.Tensor(torch.rand(x.size()))
        perturbation_x = x + eps * perturbation/torch.norm(perturbation)
        y, perturbation_y = self.forward_integration(x, integration_time), self.forward_integration(perturbation_x, integration_time)
        if integration_time is None:
            time_interval = self.time_interval[1]-self.time_interval[0]
        else:
            time_interval = integration_time[1] - integration_time[0]
        lyap = torch.log(torch.norm(y-perturbation_y)/eps)/time_interval
        return lyap

    def lyapunov_system(self, t, xy):
        x = xy[:self.ODE_dim]
        y = xy[self.ODE_dim:].reshape((self.ODE_dim, self.ODE_dim))
        xdot = self.right_hand_side(t, x)
        ydot = self.derivative(t, x).matmul(y)
        xdot_ydot = torch.cat([xdot, torch.reshape(ydot, (-1,))])
        return xdot_ydot

    def lyapunov_integration(self, x, lyap_approx=None, integration_time=None):
        if integration_time is None:
            time_intervals = torch.tensor([self.time_interval[0], self.time_interval[1]])
            integration_interval = torch.tensor(time_intervals).float().type_as(x)
        else:
            integration_interval = torch.tensor([integration_time[0], integration_time[1]])
        if self.first_layer_bool:
            x_in = self.first_layer(x)
        else:
            x_in = x
        if lyap_approx is None:
            Y = torch.eye(self.ODE_dim)
        else:
            Y = torch.diag(torch.exp(lyap_approx))
        xy0 = torch.cat([x_in, torch.reshape(Y, (-1,))])
        dt = self.compute_dt()
        out = odeint(self.lyapunov_system, xy0, integration_interval, method='euler', options={'step_size': dt})
        out = out[1, :]
        Y_end = out[self.ODE_dim:].reshape((self.ODE_dim, self.ODE_dim))
        Lmat = Y_end.T.matmul(Y_end).detach().numpy()
        time = integration_interval[1] - integration_interval[0]
        lyap = np.log(np.linalg.eig(Lmat)[0])/(2*time)
        return lyap

    def lyapunov_informed_integration(self, x, tol=10**-2):
        if self.first_layer_bool:
            x_in = self.first_layer(x)
        else:
            x_in = x
        temp_first_layer = self.first_layer_bool  # only concentrate on the ODE component
        self.first_layer_bool = False
        old_lyap = 1 + 0*x
        new_lyap = torch.Tensor(self.lyapunov_integration(x_in, lyap_approx=old_lyap))
        maxnIter = 10
        nIter = 1
        while torch.linalg.norm(old_lyap - new_lyap) > tol and nIter < maxnIter:
            old_lyap = new_lyap
            new_lyap = torch.Tensor(self.lyapunov_integration(x_in, lyap_approx=old_lyap))
            nIter += 1
        self.first_layer_bool = temp_first_layer
        return new_lyap, nIter


def grad_loss_inputs(model, data_inputs, data_labels, loss_module):
    data_inputs.requires_grad = True

    data_inputs_grad = torch.tensor(0.)

    preds, _ = model(data_inputs)

    loss = loss_module(preds, data_labels)

    data_inputs_grad = torch.autograd.grad(loss, data_inputs)[0]
    data_inputs.requires_grad = False
    return data_inputs_grad


@torch.no_grad()
def classification_levelsets(model, fig_name=None, footnote=None, contour = True, plotlim = [-2, 2]):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(5, 5), dpi=100)
    
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)

    
   
    model.to(device)

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    
    preds = model(model_inputs)
    
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = torch.nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    #we only need the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])

    ax = plt.gca()
    ax.set_aspect('equal') 
    
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,8).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')
    

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
    else: plt.show()