
Lab: `pyro`簡介
================
在 `colab` 上，請先使用以下指定安裝 `pyro`

`!pip3 install pyro-ppl`

import torch
import pyro

## `pyro` 產生隨機資料

`pyro` 產生隨機資料的方式與 `torch.distribution` 很類似，但其主要透過 `pyro.sample` 進行抽樣，且每次的抽樣，都可以對該隨機變數設定一名稱。在以下的例子中，我們先後產生滿足邏輯斯迴歸架構之 `data_x` 與 `data_y`：

torch.manual_seed(246437)
n_sample = 10000
n_feature = 3
data_x = pyro.sample("data_x", pyro.distributions.MultivariateNormal(
            loc = torch.zeros(n_sample, n_feature),
            scale_tril = torch.eye(n_feature)))

mu_x = data_x.mean(axis = 0)
sigma_x = (data_x - data_x.mean(axis = 0)).T @ (data_x - data_x.mean(axis = 0)) / n_sample
print(mu_x)
print(sigma_x)


weight_true = torch.tensor([[10.], [5.], [-5.]])
intercept_true = torch.tensor(-5.)
data_y = pyro.sample("data_y", pyro.distributions.Bernoulli(
    logits = intercept_true + data_x @ weight_true))
data = {"y":data_y, "x":data_x}



## `pyro` 對邏輯斯回歸進行最大概似法估計

使用 `pyro` 進行最大概似法估計，最簡單的方法是透過 `pyro.infer.SVI` 此物件進行，該物件主要用於進行變分推論（variational inference），在使用 `pyro.infer.SVI` 時，使用者需設定一模型（`model`）之機率分佈，以及一指引（`guide`）之機率分佈，由於我們在這邊使用最大概似法，因此，將指引設為什麼都沒做的函數。

def model_lr(data):
    weight = pyro.param("weight", torch.zeros((3,1)))
    intercept = pyro.param("intercept", torch.zeros(()))
    logits = intercept + data["x"] @ weight
    y = pyro.sample("y", pyro.distributions.Bernoulli(logits = logits),
                    obs = data["y"])
    return y

def guide_lr(data):
    pass

接著，我們就可以使用 `pyro.infer.SVI` 來進行優化。

lr = 50. / n_sample
n_steps = 201

pyro.clear_param_store()
optimizer = pyro.optim.SGD({"lr": lr})
svi = pyro.infer.SVI(model_lr, guide_lr, optimizer,
                     loss=pyro.infer.Trace_ELBO())

for step in range(n_steps):
    loss = svi.step(data)
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

for name in pyro.get_param_store():
    print(name)

print(pyro.param("weight"))
print(pyro.param("intercept"))



## `pyro` 對因素分析模型進行最大概似法估計
以下程式碼用於產生因素分析之資料

def create_fa_model(n_factor, n_item, ld, psi = None, rho = None):
    if (n_item % n_factor) != 0:
        n_item = n_factor * (n_item // n_factor)
    loading = torch.zeros((n_item, n_factor))
    item_per_factor = (n_item // n_factor)
    for i in range(n_factor):
        for j in range(i * item_per_factor,
                       (i + 1) * item_per_factor):
            loading[j, i] = ld
    if rho is None:
        cor = torch.eye(n_factor)
    else:
        unit = torch.ones((n_factor, 1))
        identity = torch.eye(n_factor)
        cor = rho * (unit @ unit.T) + (1 - rho) * identity
    if psi is None:
        uniqueness = 1 - torch.diagonal(loading @ cor @ loading.T)
    else:
        uniqueness = psi * torch.ones((n_item, ))
    return loading, uniqueness, cor

def generate_fa_data(n_sample, loading, uniqueness, cor):
    n_item = loading.size()[0]
    mean = torch.zeros((n_item, ))
    cov = loading @ cor @ loading.T + torch.diag_embed(uniqueness)
    mvn = torch.distributions.MultivariateNormal(
        loc = mean, scale_tril = torch.cholesky(cov))
    data = mvn.sample((n_sample,))
    return data

torch.manual_seed(246437)
n_factor = 4
n_item = 12
n_sample = 10000
loading_true, uniqueness_true, cor_true = create_fa_model(n_factor, n_item, ld = .7)
data = generate_fa_data(n_sample,
                        loading = loading_true,
                        uniqueness = uniqueness_true,
                        cor = cor_true)

接著，我們設定觀察變項之邊際分佈。這邊，我們在模型設定時，使用了 `pyro.plate` 來進行重複之設定。

def model_fa(data):
    loading = pyro.param("loading", 0.5 * loading_true)
    uniqueness = pyro.param("uniqueness", 0.5 * uniqueness_true)
    loading_mask = 1 *  (loading_true != 0)
    with pyro.plate("data", data.size(0)):
        pyro.sample("x", pyro.distributions.MultivariateNormal(
            loc = torch.zeros((loading.size()[0], )),
            scale_tril = torch.cholesky(
                (loading * loading_mask) @ (loading * loading_mask).T + torch.diag_embed(uniqueness))),
            obs=data)

def guide_fa(data):
    pass

lr = 1. / n_sample
n_steps = 201
pyro.clear_param_store()
optimizer = pyro.optim.SGD({"lr": lr})
svi = pyro.infer.SVI(model_fa, guide_fa, optimizer, loss=pyro.infer.Trace_ELBO())

for step in range(n_steps):
    loss = svi.step(data)
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

print(pyro.param("loading"))
print(pyro.param("uniqueness"))


前述設定的模型中，我們在設定模型與指引時，皆直接將樣本資料視為函數的輸入，事實上，此給定的動作可以事後在使用 `pyro.poutine.condition` 進行。


def model_fa():
    loading = pyro.param("loading", 0.5 * loading_true)
    uniqueness = pyro.param("uniqueness", 0.5 * uniqueness_true)
    loading_mask = 1 *  (loading_true != 0)
    with pyro.plate("data", data.size(0)):
        pyro.sample("x", pyro.distributions.MultivariateNormal(
            loc = torch.zeros((loading.size()[0], )),
            scale_tril = torch.cholesky(
                (loading * loading_mask) @ (loading * loading_mask).T + torch.diag_embed(uniqueness))))

def guide_fa():
    pass


model_fa_cond = pyro.poutine.condition(model_fa, data={"x": data})
guide_fa_cond = pyro.poutine.condition(guide_fa, data={"x": data})

接著，我們可以把 `model_fa_cond` 與 `guide_fa_cond` 丟到 `pyro.infer.SVI` 進行優化。

lr = 1. / n_sample
n_steps = 201
pyro.clear_param_store()
optimizer = pyro.optim.SGD({"lr": lr})
svi = pyro.infer.SVI(model_fa_cond, guide_fa_cond, optimizer, loss=pyro.infer.Trace_ELBO())

for step in range(n_steps):
    loss = svi.step()
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

print(pyro.param("loading"))
print(pyro.param("uniqueness"))


### 利用隨機 EM 進行最大概似法

def joint_model():
    loading = pyro.param("loading", 0.7 * loading_true)
    uniqueness = pyro.param("uniqueness", 0.7 * uniqueness_true)
    loading_mask = 1 *  (loading_true != 0)
    with pyro.plate("sample_plate", n_sample):
        eta = pyro.sample("eta", pyro.distributions.MultivariateNormal(
            loc = torch.zeros(n_factor),
            scale_tril = torch.eye(n_factor)))
        x = pyro.sample("x", pyro.distributions.MultivariateNormal(
                loc = eta @ (loading * loading_mask).T,
                scale_tril = torch.cholesky(torch.diag(uniqueness))))
    return x

def joint_guide():
    pass

lr = .1 / n_sample
n_steps = 1
pyro.clear_param_store()


for step in range(n_steps):
    model_cond_x = pyro.poutine.condition(joint_model, data = {"x": data})
    nuts_kernel = pyro.infer.NUTS(model_cond_x)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=1,
                           warmup_steps = 20, disable_progbar = True)
    mcmc.run()
    eta = mcmc.get_samples()["eta"].reshape((-1, 4))
    model_cond_x_eta = pyro.poutine.condition(joint_model,
                                              data = {"x": data,
                                            "eta":eta})
    guide_cond_x_eta = pyro.poutine.condition(joint_guide,
                                              data = {"x": data,
                                            "eta":eta})
    optimizer = pyro.optim.SGD({"lr": lr})
    svi = pyro.infer.SVI(model_cond_x_eta,
                         guide_cond_x_eta,
                         optimizer,
                         loss=pyro.infer.Trace_ELBO())
    loss = svi.step()
    if step % 5 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))



print(pyro.param("loading"))
print(pyro.param("uniqueness"))
