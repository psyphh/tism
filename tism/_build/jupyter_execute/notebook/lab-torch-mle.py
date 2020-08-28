Lab: 最大概似估計
================

import torch

## `torch` 分配物件

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

## 計算最大概似估計值

mu_true = torch.tensor([5.])
sigma_true = torch.tensor([2.])
model_normal_true = Normal(
    loc=mu_true,
    scale=sigma_true)
print("normal model:\n", model_normal_true, "\n")

sample_size = 1000
x = model_normal_true.sample(sample_shape=(sample_size,))
loss_value = -torch.mean(torch.sum(model_normal_true.log_prob(x), dim = 1))
print("negative likelihood value is", loss_value)


epochs = 200
lr = 1.0
mu = torch.tensor([0.], requires_grad=True)
sigma = torch.tensor([1.], requires_grad=True)
opt = torch.optim.Adam([mu, sigma], lr=.5)
for epoch in range(epochs):
    model_normal = Normal(loc=mu, scale=sigma)
    loss_value = -torch.mean(model_normal.log_prob(x))
    opt.zero_grad()
    loss_value.backward() # compute the gradient
    opt.step()

print("ML mean by gradient descent:", mu)
print("ML std by gradient descent:", sigma)

print("ML mean by formula:", torch.mean(x))
print("ML std by formula:", torch.std(x, unbiased=False))

mu_true = torch.tensor([-1., 0., 1.])
sigma_tril_true = torch.tensor([[3., 0., 0.], [2., 1., 0.], [.4, .5, .5]])
model_mvn_true = MultivariateNormal(
    loc=mu_true,
    scale_tril=sigma_tril_true)
print("true mean vector: \n", model_mvn_true.mean)
print("true covariance matrix: \n", model_mvn_true.covariance_matrix)

sample_size = 1000
x = model_mvn_true.sample(sample_shape=(sample_size,))
loss_value = -torch.mean(model_mvn_true.log_prob(x))
print("negative likelihood value is", loss_value)


epochs = 500
lr = .1
mu = torch.tensor(
    [0., 0., 0.], requires_grad=True)
sigma_tril = torch.tensor(
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    requires_grad=True)
opt = torch.optim.Adam([mu, sigma_tril], lr=lr)

for epoch in range(epochs):
    model_mvn = MultivariateNormal(
    loc=mu,
    scale_tril=sigma_tril)
    loss_value = -torch.mean(model_mvn.log_prob(x))
    opt.zero_grad()
    loss_value.backward() # compute the gradient
    opt.step()

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


## 實徵範例

### 產生邏吉斯迴歸資料

torch.manual_seed(48)

from torch.distributions import Bernoulli
def generate_data(n_sample,
                  weight,
                  bias = 0,
                  mean_feature = 0,
                  std_feature = 1,
                  dtype = torch.float64):
    weight = torch.tensor(weight, dtype = dtype)
    n_feature = weight.shape[0]
    x = torch.normal(mean = mean_feature,
                     std = std_feature,
                     size = (n_sample, n_feature),
                     dtype = dtype)
    weight = weight.view(size = (-1, 1))
    logit = bias + x @ weight
    bernoulli = Bernoulli(logits = logit)
    y = bernoulli.sample()
    return x, y

# run generate_data
x, y = generate_data(n_sample = 1000,
                     weight = [-5, 3, 0],
                     bias = 2,
                     mean_feature = 10,
                     std_feature = 3,
                     dtype = torch.float64)

### 建立一進行邏吉斯迴歸分析之物件

# define a class to fit logistic regression
class LogisticRegression():
    def __init__(self, dtype = torch.float64):
        self.dtype = dtype
        self.weight = None
        self.bias = None
    def log_lik(self, x, y):
        logit = self.bias + x @ self.weight
        bernoulli = Bernoulli(logits = logit)
        return torch.mean(bernoulli.log_prob(y))
    def fit(self, x, y, epochs = 200, lr = .1):
        if x.dtype is not self.dtype:
            x = x.type(dtype = self.dtype)
        if y.dtype is not self.dtype:
            y = y.type(dtype = self.dtype)
        n_feature = x.size()[1]
        self.bias = torch.zeros(size = (1,),
                                dtype = self.dtype,
                                requires_grad = True)
        self.weight = torch.zeros(size = (n_feature, 1),
                                  dtype = self.dtype,
                                  requires_grad = True)
        opt = torch.optim.Adam([self.bias, self.weight], lr=lr)
        for epoch in range(epochs):
            loss_value = - self.log_lik(x, y)
            opt.zero_grad()
            loss_value.backward() # compute the gradient
            opt.step()
        return self

### 計算模型參數

# fit logistic model
model_lr = LogisticRegression()
model_lr.fit(x, y, epochs = 2000, lr = 1)
print(model_lr.bias)
print(model_lr.weight)

# fit logistic model via sklearn
# please install sklearn first
from sklearn import linear_model
model_lr_sklearn = linear_model.LogisticRegression(C=10000)
model_lr_sklearn.fit(x, y)
print(model_lr_sklearn.intercept_)
print(model_lr_sklearn.coef_)