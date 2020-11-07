Lab: 最大概似估計
================

import torch

## `torch` 分配物件

### 分配物件之基礎
`torch.distribution` 內建了許多機率分配物件（見[官方網頁](https://pytorch.org/docs/stable/distributions.html)），而分配物件可供使用者

1. 產生隨機樣本。
2. 給定實現值計算可能性或機率值。
2. 給定上界計算累積機率值（並非每個分配都可以）。

在產生一分配物件時，我們需給定該分配的參數。以常態分配為例，其參數包括了平均數與變異數，此兩參數亦稱作位置（location）參數與尺度（scale）參數

from torch.distributions import Normal
normal = Normal(loc=0., scale=1.)

再以Binomial分配為例，其參數為嘗試次數與成功之機率（也可以使用對數勝率來設定）

from torch.distributions import Binomial
binomial = Binomial(total_count = 10,
                    probs = 0.5)

我們可以透過對分配物件的列印，以了解其內部之參數設定：

print(normal)
print(binomial)

對於已建立之分配物件，我們可以利用其`.sample()`方法來產生隨機變數

print("random sample with shape ():\n",
      normal.sample())
print("random sample with shape (3,):\n",
      normal.sample(sample_shape=(3,)))
print("random sample with shape (2,3):\n",
      normal.sample(sample_shape=(2, 3)))

print("random sample with shape ():\n",
      binomial.sample())
print("random sample with shape (3,):\n",
      binomial.sample(sample_shape=(3,)))
print("random sample with shape (2,3):\n",
      binomial.sample(sample_shape=(2, 3)))

從前述的例子，我們可以看到 `sample_shape` 可用於設定產生樣本之個數與樣本張量之排列形狀。

給定一組實現值，`log_prob()`可用於計算該實現之對數可能性或機率

print("log-likelihood given value with shape ():\n",
      normal.log_prob(value=torch.Tensor([0])), "\n")
print("log-likelihood given value with (3,):\n",
      normal.log_prob(value=torch.Tensor([-1, 0, .5])), "\n")
print("log-likelihood given value with (2,3):\n",
      normal.log_prob(value=torch.Tensor([[-1, 0, .5], [-2, 1, 3]])))

print("log-probability given value with shape ():\n",
      binomial.log_prob(value=torch.Tensor([5])), "\n")
print("log-probability given value with (3,):\n",
      binomial.log_prob(value=torch.Tensor([5, 3, 7])), "\n")
print("log-probability given value with (2,3):\n",
      binomial.log_prob(value=torch.Tensor([[5, 3, 7], [2, 0, 10]])))


在給定上界之數值，常態分配之`.cdf()` 可用於計算該上界數值所對應之累積機率數值

print("cumulative probability given value with shape ():\n",
      normal.cdf(value=torch.Tensor([0])), "\n")
print("cumulative probability given value with (3,):\n",
      normal.cdf(value=torch.Tensor([-1, 0, .5])), "\n")
print("cumulative probability given value with (2,3):\n",
      normal.cdf(value=torch.Tensor([[-1, 0, .5], [-2, 1, 3]])))

不過，binomial分配並無 `cdf()` 方法可評估累積機率值。



### 分配物件之形狀
`pytorch` 分配物件之設計，乃參考 `tensorflow_probability`此套件，而分配物件在形狀上，牽涉到三類型之形狀：

1. 樣本形狀（sample shape）：為用於描述獨立且具有相同分配隨機樣本之形狀，先前產生隨機樣本時，所設定的 `sample_shape` 即為樣本形狀。
2. 批次形狀（batch shape）：為用於描述獨立，但不具有相同分配隨機樣本之形狀，其可以透過模型參數之形狀進行設定。
3. 事件形狀（event shape）：為用於描述多變量分配之形狀，各變數間可能不具有統計獨立之特性。

先前產生的常態分配，其在 `batch_shape` 與 `event_shape` 上，皆為純量，故其數值為0-d之張量。

from torch.distributions import Normal
normal = Normal(loc=0., scale=1.)
print(normal.batch_shape)
print(normal.event_shape)

接下來，我們設定一批次形狀為 `[2]` 之常態分配物件：

normal_batch = Normal(loc=torch.Tensor([0., 1.]),
                      scale=torch.Tensor([1., 1.5]))
print(normal_batch.batch_shape)
print(normal_batch.event_shape)

該分配可產生一形狀為 `[2]` 之常態隨機變數，第一個元素的平均數為0，變異數為1，第二個元素的平均數為1，變異數為1.5。接著，我們從該分配中產生不同樣本形狀之隨機樣本

print("random sample with sample_shape ():\n",
      normal_batch.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      normal_batch.sample(sample_shape=(3,)), "\n")
print("random sample with sample_shape (2,3):\n",
      normal_batch.sample(sample_shape=(2,3)))

我們可以看見，產生樣本的張量尺寸為 `sample_size + batch_size`，尺寸的最後一個維度皆為2。


當分配物件的 `batch_shape` 為 `[2]` 時，則在評估其對數機率時若僅輸入 `[0]`，則 `[0]` 會被廣播為 `[0, 0]` 評估，而 `[[0], [0]]` 會被廣播為 `[[0, 0], [0, 0]]`。

print("log-probability given value with shape ():\n",
      normal_batch.log_prob(torch.Tensor([0])), "\n")
print("log-probability given value with shape (2,):\n",
      normal_batch.log_prob(torch.Tensor([0, 0])), "\n")
print("log-probability given value with shape (2,1):\n",
      normal_batch.log_prob(torch.Tensor([[0], [0]])))

分配物件的 `event_shape`，可透過多變量分配之參數設定。以多元常態分配為例，我們可以透過其平均數向量與共變異數矩陣設定 `event_shape`

from torch.distributions import MultivariateNormal
mvn = MultivariateNormal(
    loc=torch.Tensor([0, 1]),
    scale_tril=torch.cholesky(torch.Tensor([[1., 0.], [0., .5]])))
print(mvn.batch_shape)
print(mvn.event_shape)

由於我們給定的平均數向量與共變異數矩陣適用於二維之多變量常態分配，因此，其 `event_shape` 為 `[2]`。這邊需特別注意的是，我們並非直接給定共變異數矩陣，取而代之的是，給定共變異數矩陣之 `cholesky` 拆解。

我們可以使用該多元常態分配來產生資料，以及評估其對數可能性數值

print("random sample with sample_shape ():\n",
      mvn.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      mvn.sample(sample_shape=(3,)), "\n")
print("random sample with sample_shape (2, 3):\n",
      mvn.sample(sample_shape=(2, 3)))

print("log-likelihood given value with shape (2,):\n",
      mvn.log_prob(torch.Tensor([0, 0])), "\n")
print("log-likelihood given value with shape (2,1):\n",
      mvn.log_prob(torch.Tensor([[0, 0], [0, 0]])))

這邊需要別注意的是，屬於同一事件之觀測值，僅會給予一對數機率值，方便用於建立概似函數。


另外，也可以透過 `Independent` 此函數，將分配之 `batch_size` 重新解釋為 `event_size`，`reinterpreted_batch_ndims` 用於設定有多少個面向要從 `batch_shape` 轉為 `event_shape`（從右至左）。

from torch.distributions import Independent
normal_event = Independent(normal_batch,
                           reinterpreted_batch_ndims = 1)
print(normal_event.batch_shape)
print(normal_event.event_shape)

最後，我們也可以對多元常態分配設定 `batch_shape`

mvn_batch = MultivariateNormal(
    loc=torch.Tensor([[0, 1],[1, 2],[2, 3]]),
    scale_tril=torch.cholesky(torch.Tensor([[1., .2], [.2, .5]])))
print(mvn_batch.batch_shape)
print(mvn_batch.event_shape)

此分配每次產生一形狀為 `(3, 2)` 之樣本，若進一步設定 `sample_shape`，則其產生之樣本張量形狀為 `smaple_shape + (3, 2)`：

print("random sample with sample_shape ():\n",
      mvn_batch.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      mvn_batch.sample(sample_shape=(3,)), "\n")
print("random sample with sample_shape (2, 3):\n",
      mvn_batch.sample(sample_shape=(2, 3)))

關於前述三種形狀之說明，讀者亦可參考此[網誌](https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/)。

## 最大概似估計法

### 建立概似函數

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

### 進行優化

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