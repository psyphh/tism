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
binomial = Binomial(total_count = 10, probs = 0.5)

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
利用 `torch` 的分配物件，我們可以很容易地來建立概似函數。

首先，我們先以常態分配進行說明。為了產生樣本資料，我們先設定一平均數為5，標準差為4之常態分配

mu_true = torch.tensor([5.])
sigma_true = torch.tensor([2.])
model_normal_true = Normal(
    loc=mu_true,
    scale=sigma_true)
print("normal model:\n", model_normal_true, "\n")

接著，我們可以利用此常態分配來產生一樣本數為1000之資料，並評估該資料在平均數為5，標準差為4此常態分配下之負對數可能性（negative log-likelihood）

sample_size = 1000
x = model_normal_true.sample(sample_shape=(sample_size,))
loss_value = - model_normal_true.log_prob(x).mean()
print("negative likelihood value is", loss_value)

因此，只要在資料的形狀上可以匹配，我們即可使用分配物件的`log_prob()` 方法，再搭配 `sum()` 或是 `mean()` 來計算對數概似函數數值。

### 進行優化

在建立完概似函數後，我們就可以透過 `torch` 的優化器進行優化。在這邊需要特別注意的是，由於模型的參數會在優化過程中更新，因此，其必須使用一可為分之張量來儲存，並且，在每次更新完參數數值後，皆需再次產生一新的分配物件，以計算概述函數之數值

epochs = 200
lr = 1.0
mu = torch.tensor([0.], requires_grad=True)
sigma = torch.tensor([1.], requires_grad=True)
opt = torch.optim.Adam([mu, sigma], lr=.5)
for epoch in range(epochs):
    model_normal = Normal(loc=mu, scale=sigma)
    loss_value = - model_normal.log_prob(x).mean()
    opt.zero_grad()
    loss_value.backward() # compute the gradient
    opt.step()

print("ML mean by gradient descent:", mu.data.numpy())
print("ML std by gradient descent:", sigma.data.numpy())

我們可以比較前述使用梯度下降法所得到的結果，與直接帶公式計算結果間的差異，可以發現兩者間的差異主要展現在小數點後3位，是絕大多數情況下可以忽略的誤差。

print("ML mean by formula:", torch.mean(x).numpy())
print("ML std by formula:", torch.std(x, unbiased=False).numpy())

### 多元常態分配之最大概似估計

多元常態分配為統計建模中常被使用之分配，因此，我們在此對該分配之參數進行最大概似法之估計。

多元常態分配之最大概似估計最麻煩的部分在於，共變異數矩陣是對稱正定矩陣，因此，雖然共變異矩陣中有 $P \times P$ 個元素，但事實上，其僅有 $P(P+1)/2$ 個能夠自由估計之參數，並且，其數值需滿足正定矩陣之要求。為了處理此困難，在進行多元常態分配之參數設定時，我們不直接設定共變異數矩陣，取而代之的是，設定該矩陣之 Cholesky 拆解，即 `scale_tril`

mu_true = torch.tensor([-1., 0., 1.])
sigma_tril_true = torch.tensor([[3., 0., 0.], [2., 1., 0.], [.4, .5, .5]])
model_mvn_true = MultivariateNormal(
    loc=mu_true,
    scale_tril=sigma_tril_true)
print("true mean vector: \n", model_mvn_true.mean)
print("true covariance matrix: \n", model_mvn_true.covariance_matrix)


前一程式碼所展示的共變異數矩陣，可透過以下的公式獲得

sigma_tril_true @ sigma_tril_true.t()

前式可以確保所得到的共變異數矩陣為對稱矩陣，另外，如果說給定的對角線元素為正的，則可以進一步確保該共變異數矩陣為對稱正定矩陣。

接著，我們就可以利用該分配物件來產生資料、計算概似函數數值、以及計算最大概似估計值。

sample_size = 1000
x = model_mvn_true.sample(sample_shape=(sample_size,))
loss_value = -model_mvn_true.log_prob(x).mean()
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

print("ML mean by gradient descent: \n", mu)
print("ML covariance by gradient descent: \n", sigma_tril @ sigma_tril.t())

sample_mean = torch.mean(x, dim = 0)
sample_moment2 = (x.t() @ x) / sample_size
sample_cov = sample_moment2 - torch.ger(sample_mean, sample_mean)
print("ML mean by formula: \n", sample_mean)
print("ML covariance by formula: \n", sample_cov)


x = torch.tensor([1., 2., 3., 4., 5., 6.], requires_grad=True)
m = torch.zeros((3, 3))

tril_indices = torch.tril_indices(row=3, col=3, offset=0)
m[tril_indices[0], tril_indices[1]] = x
print(m)

## 實徵範例

### 產生邏吉斯迴歸資料

torch.manual_seed(246437)

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
        self.x = None
        self.y = None
        self.weight = None
        self.bias = None
    def log_lik(self, bias, weight):
        logit = bias + self.x @ weight
        bernoulli = Bernoulli(logits = logit)
        return bernoulli.log_prob(self.y).mean()
    def fit(self, x, y, optimizer = "LBFGS",
            epochs = 200, lr = .1, tol = 10**(-7)):
        if x.dtype is not self.dtype:
            x = x.type(dtype = self.dtype)
        if y.dtype is not self.dtype:
            y = y.type(dtype = self.dtype)
        self.x = x
        self.y = y
        self.n_sample = x.size()[0]
        self.n_feature = x.size()[1]
        bias = torch.zeros(size = (1,),
                           dtype = self.dtype,
                           requires_grad = True)
        weight = torch.zeros(size = (self.n_feature, 1),
                             dtype = self.dtype,
                             requires_grad = True)
        if optimizer == "LBFGS":
            opt = torch.optim.LBFGS([bias, weight],
                                    lr=lr, max_iter = epochs,
                                    tolerance_grad = tol,
                                    line_search_fn = "strong_wolfe")
            def closure():
                opt.zero_grad()
                loss_value = - self.log_lik(bias, weight)
                loss_value.backward()
                return loss_value
            opt.step(closure)
        else:
            opt = getattr(torch.optim, optimizer)([bias, weight], lr=lr)
            for epoch in range(epochs):
                opt.zero_grad()
                loss_value = - self.log_lik(bias, weight)
                loss_value.backward()
                with torch.no_grad():
                    grad_max = max(bias.grad.abs().max().item(),
                                   weight.grad.abs().max().item())
                if (grad_max < tol):
                    break
                opt.step()
        self.bias = bias.data.numpy()
        self.weight = weight.data.numpy()
        # print(opt.state_dict())
        return self
    def vcov(self):
        from torch.autograd.functional import hessian
        bias, weight = torch.tensor(self.bias), torch.tensor(self.weight)
        h = hessian(self.log_lik, (bias, weight))
        fisher_obs = -torch.cat([torch.cat([h[0][0],h[0][1].squeeze(dim = 2)], dim = 1),
                                torch.cat([h[1][0].squeeze(dim =0).squeeze(dim =1),
                                           h[1][1].squeeze()], dim = 1)],
                               dim = 0)

        vcov = torch.inverse(fisher_obs)/self.n_sample
        return vcov

### 計算模型參數

# fit logistic model
model_lr = LogisticRegression()
model_lr.fit(x, y, optimizer = "LBFGS", epochs = 2000, lr = 1)
print(model_lr.bias)
print(model_lr.weight)

# fit logistic model via sklearn
# please install sklearn first
from sklearn import linear_model
model_lr_sklearn = linear_model.LogisticRegression(C=10000)
model_lr_sklearn.fit(x, y.flatten())
print(model_lr_sklearn.intercept_)
print(model_lr_sklearn.coef_)

### 計算參數估計標準誤

vcov = model_lr.vcov()

vcov.diagonal().sqrt()

# fit logistic model via statsmodels
# please install statsmodels first
import statsmodels.api as sm
model_lr_sm = sm.Logit(y.numpy(), sm.add_constant(x.numpy()))
print(model_lr_sm.fit().summary())

### 練習
1. 請建立一類型，其可以使用最大概似法，執行線性回歸分析。

2. 請在前述的類型中，加入一摘要方法（summary()），該方法可以列印出參數估計的假設檢定與信賴區間。

3. 請建立一類型，其可以估計多元常態分配的平均數與共變異數矩陣，並且提供各參數估計之標準誤。