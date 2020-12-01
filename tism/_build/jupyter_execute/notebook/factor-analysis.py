因素分析
================================


## 因素分析模型

### 模型架構
令 $x_i$ 表示個體於第 $i$ 個測驗（或是試題）的觀測分數（observed score）（$i=1,2,...,I$），因素分析（factor analysis）試圖引入 $M$ 個潛在因素（latent factor）$\eta_1, \eta_2,...,\eta_M$，以解釋 $x_i$ 之變異

$$
x_i = \nu_i + \sum_{m=1}^M \lambda_{im} \eta_m + \epsilon_i
$$

這裡，$\eta_m$ 表示第 $m$ 個潛在因素，其對 $x_i$ 之效果 $\lambda_{im}$ 被稱作因素負荷量（factor loading），其反映 $\eta_m$ 每變動一單位，預期 $x_i$ 跟著變動的量，$\nu_i$ 為試題 $i$ 之截距，其反映當所有 $\eta_m = 0$時，$x_i$ 的預期數值，而 $\epsilon_i$ 則為試題 $i$ 所對應之測量誤差。


因素分析模型假設

1. 潛在因素 $\eta_m$ 與誤差分數 $\epsilon_i$ 為統計獨立。
2. $\eta_m \sim (0, 1)$，$\mathbb{C} \text{ov}(\eta_m, \eta_{m'}) = \phi_{mm'}$。當所有 $\phi_{mm'}=0$（$m \neq m'$）時，我們稱此因素結構為正交結構（orthogonal structure）。
3. $\epsilon_i \sim (0, \psi^2_i)$，$\mathbb{C} \text{ov}(\epsilon_i, \epsilon_{i'}) =  \psi_{ii'}$。多數情況下，模型假設$\psi_{ii'} = 0$（$i \neq i'$）。


在平均數與共變異數結構方面，當誤差分數間無相關的假設下，該結構為

1. $\mu_i(\theta) = \nu_i$
2. $\sigma_{i}^2(\theta) = \sum_{m=1}^M \lambda_{im}^2 + \psi_{i}^2$。
3. $\sigma_{ij}(\theta) = \sum_{m=1}^M \sum_{k=1}^M \lambda_{im}\lambda_{jk} \phi_{mk}$。





### 矩陣形式之模型架構
若我們將 $\eta_1, \eta_2,...,\eta_M$ 與 $\lambda_{i1}, \lambda_{i2},...,\lambda_{iM}$ 皆排成 $M$ 維之向量，即 $\eta = (\eta_1, \eta_2,...,\eta_M)$ 與 $\lambda_i = (\lambda_{i1}, \lambda_{i2},...,\lambda_{iM})$，則前述之方程式可以寫為

$$
x_i = \nu_i + \lambda_{i}^T \eta + \epsilon_i
$$

進一步，令 $x = (x_1, x_2, ..., x_I)$，$\nu = (\nu_1, \nu_2, ..., \nu_I)$，以及 $\epsilon = (\epsilon_1, \epsilon_2, ..., \epsilon_I)$ 皆表示一 $I \times 1$ 矩陣，而

$$
\Lambda =
\underbrace{\begin{pmatrix}
\lambda_1^T \\
\lambda_2^T \\
\vdots \\
\lambda_I^T
\end{pmatrix}}_{I \times M}
=
\underbrace{\begin{pmatrix}
\lambda_{11} & \lambda_{12} & \cdots & \lambda_{1M} \\
\lambda_{21} & \lambda_{22} & \cdots & \lambda_{2M} \\
\vdots & \vdots & \ddots & \vdots \\
\lambda_{I1} & \lambda_{I2} & \cdots & \lambda_{IM} \\
\end{pmatrix}}_{I \times M}
$$


在前述之符號表示下，觀察變項向量 $x$ 可以被寫為

$$
x = \nu + \Lambda \eta + \epsilon
$$

我們可以將前述因素分析模型之假設，轉為矩陣之形式：（1）$\eta$ 與 $\epsilon$ 兩者獨立；（2）$\eta \sim (0, \Phi)$ ；（3）$\epsilon \sim (0, \Psi)$。此時，平均數與共變異數結構可以寫為

$$
\begin{aligned}
\mu(\theta) &= \nu \\
\Sigma(\theta) &= \Lambda \Phi \Lambda^T + \Psi
\end{aligned}
$$


## 參數估計

### 最小平方法
給定一樣本共變異數矩陣 $S$，其第 $i,j$ 元素為 $s_{ij}$，則一般最小平方（ordinal least squares，簡稱OLS）法透過最小化以下準則以獲得模型參數之估計

$$
\mathcal{D}_{OLS}(\theta) =\sum_{i=j}^I \sum_{j=1}^I (s_{ij} - \sigma_{ij}(\theta))^2.
$$

一個與最小平方法有關的變形為最小殘差（minimum residual，簡稱MINRES）法，其僅考慮共變異數之非對角線元素進行估計


$$
\mathcal{D}_{MINRES}(\theta) =\sum_{i=j+1}^I \sum_{j=1}^{I-1} (s_{ij} - \sigma_{ij}(\theta))^2.
$$

當所有的 $\psi_i^2$ 皆可被自由估計時，MINRES與OLS兩者為等價的。

前述的OLS法可以進一步引入權重，即成為加權最小平方法（weighted least squares，簡稱WLS），其估計準則改為

$$
\mathcal{D}_{WLS}(\theta) =\sum_{i=j}^I \sum_{j=1}^I w_{ij} (s_{ij} - \sigma_{ij}(\theta))^2.
$$

這裡，$w_{ij}$ 表示對 $(s_{ij} - \sigma_{ij}(\theta))$ 此殘差給予的權重，其並非模型之參數，乃研究者於估計準則中給定的，當 $w_{ij}$ 越大，即表示研究者希望 $(s_{ij} - \sigma_{ij}(\theta))$ 之差異應越小越好。

### 最大概似法

在因素分析的模型假設下，$x$ 之平均數與共變異數為 $\mu(\theta)$ 與 $\Sigma(\theta))$，若再進一步引進多元常態分配之假設，則 $x \sim \text{Normal}(\mu(\theta), \Sigma(\theta))$，此時，$x$ 之對數機率密度函數為

$$
\log f(x|\theta) = -\frac{I}{2} \log{2\pi} - \frac{1}{2} |\Sigma(\theta)| - \frac{1}{2} (x - \mu(\theta))^T \Sigma(\theta) ^{-1} (x - \mu(\theta))
$$

因此，給定樣本資料 $x_1, x_2,...,x_N$下，最大概似估計準則可以寫為

$$
\ell(\theta) = C  -\frac{N}{2} |\Sigma(\theta)| - \frac{1}{2} \sum_{n=1}^N (x_n - \mu(\theta))^T \Sigma(\theta) ^{-1} (x_n - \mu(\theta))
$$

前述之最大概似準則可以簡化為

$$
\ell(\theta) = C  - \frac{N}{2} |\Sigma(\theta)| - \frac{N}{2} tr(\Sigma(\theta) ^{-1} S) - \frac{N}{2} (m - \mu(\theta))^T \Sigma(\theta) ^{-1} (m - \mu(\theta))
$$