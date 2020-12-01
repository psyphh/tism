import pyro
import torch


loading = torch.tensor([[.7, 0, 0],
                        [.7, 0, 0],
                        [.7, 0, 0],
                        [0, .7, 0],
                        [0, .7, 0],
                        [0, .7, 0],
                        [0, 0, .7],
                        [0, 0, .7],
                        [0, 0, .7]])

uniqueness = torch.tensor([.51, .51, .51,
                           .51, .51, .51,
                           .51, .51, .51])


import pyro
import torch
mvn = pyro.distributions.MultivariateNormal(
    loc=torch.Tensor([0, 1]),
    scale_tril=torch.cholesky(torch.Tensor([[1., 0.], [0., .5]])))

