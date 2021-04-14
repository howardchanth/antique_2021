"""
Source: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

"""

"""
---------------------------------------------------------------------------
                                Triplet Loss
---------------------------------------------------------------------------
Reference: https://github.com/adambielski/siamese-triplet/blob/master/networks.py
"""


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class CNNEmbeddingNet(nn.Module):
    def __init__(self, in_channel=3, out_dim=128):
        super(CNNEmbeddingNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=(3, 3)), nn.PReLU(),
                                  nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(32, 64, kernel_size=(3, 3)), nn.PReLU(),
                                  nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 62 * 62, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, out_dim)
                                )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class CNNEmbeddingNetL2(CNNEmbeddingNet):
    """
    L2 Normalized CNN Embedding Net
    """

    def __init__(self, in_channel, out_dim):
        super(CNNEmbeddingNetL2, self).__init__(in_channel, out_dim)

    def forward(self, x):
        x = super(CNNEmbeddingNetL2, self).forward(x)
        x /= x.clone().norm(p=2, dim=1, keepdim=True)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    """
    Wrapper for an embedding network, processes triplets of inputs
    Triplet loss model from:
    https://github.com/adambielski/siamese-triplet/blob/master/networks.py
    """

    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


"""
---------------------------------------------------------------------------
                                Meta Learning
---------------------------------------------------------------------------
"""


def get_params(module, memo=None, pointers=None):
    """ Returns an iterator over PyTorch module parameters that allows to update parameters
        (and not only the data).
    ! Side effect: update shared parameters to point to the first yield instance
        (i.e. you can update shared parameters and keep them shared)
    Yields:
        (Module, string, Parameter): Tuple containing the parameter's module, name and pointer
    """
    if memo is None:
        memo = set()
        pointers = {}
    for name, p in module._parameters.items():
        if p not in memo:
            memo.add(p)
            pointers[p] = (module, name)
            yield module, name, p
        elif p is not None:
            prev_module, prev_name = pointers[p]
            module._parameters[name] = prev_module._parameters[prev_name]  # update shared parameter pointer
    for child_module in module.children():
        for m, n, p in get_params(child_module, memo, pointers):
            yield m, n, p


class MetaLearner(nn.Module):
    """ Bare Meta-learner class
        Should be added: Initialization, hidden states, more control over everything
    """

    def __init__(self, model):
        super(MetaLearner, self).__init__()
        self.weights = Parameter(torch.Tensor(1, 2))

    def forward(self, forward_model, backward_model):
        """ Forward optimizer with a simple linear neural net
        Inputs:
            forward_model: PyTorch module with parameters gradient populated
            backward_model: PyTorch module identical to forward_model (but without gradients)
              updated at the Parameter level to keep track of the computation graph for meta-backward pass
        """
        f_model_iter = get_params(forward_model)
        b_model_iter = get_params(backward_model)
        for f_param_tuple, b_param_tuple in zip(f_model_iter, b_model_iter):  # loop over parameters
            # Prepare the inputs, we detach the inputs to avoid computing 2nd derivatives (re-pack in new Variable)
            (module_f, name_f, param_f) = f_param_tuple
            (module_b, name_b, param_b) = b_param_tuple
            inputs = Variable(torch.stack([param_f.grad.data, param_f.data], dim=-1))
            # Optimization step: compute new model parameters, here we apply a simple linear function
            dW = F.linear(inputs, self.weights).squeeze()
            param_b = param_b + dW
            # Update backward_model (meta-gradients can flow) and forward_model (no need for meta-gradients).
            module_b._parameters[name_b] = param_b
            param_f.data = param_b.data
