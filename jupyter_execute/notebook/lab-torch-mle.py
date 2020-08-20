Lab: 最大概似估計
================

import torch

from torch.distributions import Normal
normal = Normal(loc=0., scale=1.)

print("random sample with shape ():\n",
      normal.sample())
print("random sample with shape (3,):\n",
      normal.sample(sample_shape=(3,)))
print("random sample with shape (2,3):\n",
      normal.sample(sample_shape=(2, 3)))

print("cumulative probability given value with shape ():\n",
      normal.cdf(value=0), "\n")
print("cumulative probability given value with (3,):\n",
      normal.cdf(value=torch.Tensor([-1, 0, .5])), "\n")
print("cumulative probability given value with (2,3):\n",
      normal.cdf(value=torch.Tensor([[-1, 0, .5], [-2, 1, 3]])))

print("cumulative probability given value with shape ():\n",
      normal.log_prob(value=0), "\n")
print("cumulative probability given value with (3,):\n",
      normal.log_prob(value=torch.Tensor([-1, 0, .5])), "\n")
print("cumulative probability given value with (2,3):\n",
      normal.log_prob(value=torch.Tensor([[-1, 0, .5], [-2, 1, 3]])))

print(normal)

print(normal.batch_shape)
print(normal.event_shape)

normal_batch = Normal(loc=torch.Tensor([0., 1.]), scale=torch.Tensor([1., 1.5]))
print(normal_batch)

print("random sample with sample_shape ():\n",
      normal_batch.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      normal_batch.sample(sample_shape=(3,)), "\n")
print("random sample with sample_shape (2,3):\n",
      normal_batch.sample(sample_shape=(2,3)))


print("log-probability given value with shape ():\n",
      normal_batch.log_prob(0), "\n")
print("log-probability given value with shape (2,):\n",
      normal_batch.log_prob(torch.Tensor([0, 0])), "\n")
print("log-probability given value with shape (2,1):\n",
      normal_batch.log_prob(torch.Tensor([[0], [0]])))

from torch.distributions import MultivariateNormal
mvn = MultivariateNormal(
    loc=torch.Tensor([0, 1]),
    scale_tril=torch.cholesky(torch.Tensor([[1., 0.], [0., .5]])))
print(mvn)


print("random sample with sample_shape ():\n",
      mvn.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      mvn.sample(sample_shape=(3,)), "\n")
print("random sample with sample_shape (2, 3):\n",
      mvn.sample(sample_shape=(2, 3)))

print("log-probability given value with shape (2,):\n",
      mvn.log_prob(torch.Tensor([0, 0])), "\n")
print("log-probability given value with shape (2,1):\n",
      mvn.log_prob(torch.Tensor([[0, 0], [0, 0]])))

from torch.distributions import Independent
normal_batch = Independent(normal_batch, reinterpreted_batch_ndims=1)
print(normal_batch.batch_shape)
print(normal_batch.event_shape)


mvn_batch = MultivariateNormal(
    loc=torch.Tensor([[0, 1],[1, 2],[2, 3]]),
    scale_tril=torch.cholesky(torch.Tensor([[1., .2], [.2, .5]])))
mvn_batch

print("random sample with sample_shape ():\n",
      mvn_batch.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      mvn_batch.sample(sample_shape=(3,)), "\n")
print("random sample with sample_shape (2, 3):\n",
      mvn_batch.sample(sample_shape=(2, 3)))


mu_true = torch.tensor([5.])
sigma_true = torch.tensor([2.])
normal_model_true = Normal(
    loc=mu_true,
    scale=sigma_true)
print("normal model:\n", normal_model_true, "\n")

sample_size = 1000
x = normal_model_true.sample(sample_shape=(sample_size,))
loss_value = -torch.mean(torch.sum(normal_model_true.log_prob(x), dim = 1))
print("negative likelihood value is", loss_value)


epochs = 200
lr = 1.0
mu = torch.tensor([0.], requires_grad=True)
sigma = torch.tensor([1.], requires_grad=True)
optimizer = torch.optim.Adam([mu, sigma], lr=.5)
for epoch in range(epochs):
    normal_model = Normal(loc=mu, scale=sigma)
    loss_value = -torch.mean(normal_model.log_prob(x))
    optimizer.zero_grad()
    loss_value.backward() # compute the gradient
    optimizer.step()

print("ML mean by gradient descent:", mu)
print("ML std by gradient descent:", sigma)

print("ML mean by formula:", torch.mean(x))
print("ML std by formula:", torch.std(x, unbiased=False))

mu_true = torch.tensor([-1., 0., 1.])
sigma_tril_true = torch.tensor([[3., 0., 0.], [2., 1., 0.], [.4, .5, .5]])
mvn_model_true = MultivariateNormal(
    loc=mu_true,
    scale_tril=sigma_tril_true)
print("true mean vector: \n", mvn_model_true.mean)
print("true covariance matrix: \n", mvn_model_true.covariance_matrix)

sample_size = 1000
x = mvn_model_true.sample(sample_shape=(sample_size,))
loss_value = -torch.mean(mvn_model_true.log_prob(x))
print("negative likelihood value is", loss_value)


epochs = 500
lr = .1
mu = torch.tensor(
    [0., 0., 0.], requires_grad=True)
sigma_tril = torch.tensor(
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    requires_grad=True)
optimizer = torch.optim.Adam([mu, sigma_tril], lr=lr)

for epoch in range(epochs):
    mvn_model = MultivariateNormal(
    loc=mu,
    scale_tril=sigma_tril)
    loss_value = -torch.mean(mvn_model.log_prob(x))
    optimizer.zero_grad()
    loss_value.backward() # compute the gradient
    optimizer.step()

print("ML mean by gradient descent: \n",
      mu)
print("ML covariance by gradient descent: \n",
      sigma_tril @ torch.transpose(sigma_tril, 0, 1))

sample_mean = torch.mean(x, dim = 0)
sample_moment2 = (torch.transpose(x, 0, 1) @ x) / sample_size
sample_cov = sample_moment2 - torch.ger(sample_mean, sample_mean)
print("ML mean by formula: \n",
      sample_mean)
print("ML covariance by formula: \n",
      sample_cov)