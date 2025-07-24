import numpy as np
import torch
from models.nODE import nODE
from models.neural_odes import NeuralODE

def input_to_output(input, node, time_interval):
    return node.flow(input, time_interval)[-1]


def LEs(input, node, time_interval=None):
    if time_interval is None:
        if isinstance(node, nODE):
            time_interval = node.time_interval
        elif isinstance(node, NeuralODE):
            T = node.T
            time_interval = torch.tensor([0, T], dtype=torch.float32)
        else:
            Exception(f'Unrecognised input, expected nODE or NeuralODE, got {type(node)}')
    if isinstance(node, NeuralODE):
        # fix the node so it is just a input to output of the other variable
        input_to_output_lambda = lambda x: node.flow(x, time_interval)[-1]
    else:
        input_to_output_lambda = lambda x: node.forward_integration(x)

    # Compute the Jacobian matrix
    J = torch.autograd.functional.jacobian(input_to_output_lambda, input)

    # Perform Singular Value Decomposition
    U, S, V = torch.svd(J)

    # Return the maximum singular value
    return 1 / (time_interval[1] - time_interval[0]) * np.log(S)


def input_to_output_nODE(input, anode : nODE):
    return anode.forward_integration(input)
