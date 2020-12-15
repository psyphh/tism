試題反應理論
================================

試題反應理論（item response theory，簡稱IRT）乃一系列之測量模型，其用於刻畫個體於試題作答之反應。

## 基本試題反應理論模型

### 試題反應函數與作答機率
IRT的核心在於如何利用一數學函數，刻畫個體潛在能力（latent ability）與作答機率（response category）之關係，該數學函數被稱作試題反應函數（item response function，簡稱IRF），而其所對應之圖示，則被稱作試題特徵曲線（item characteristic curve，簡稱ICC）。

首先，我們考慮最簡單的IRT模型，其僅考慮單一之潛在變項，而各試題之作答為二分變項，1表示答對，0表示答錯。令 $\eta$ 表示個體的潛在能力，$x_i$ 表示個體於試題 $i$ 之作答反應，則試題 $i$ 之 IRF 被定義為給定 $\eta$ 之下，個體答對之機率，即

$$
\pi_i(\eta) = \mathbb{P}(x_i = 1|\eta)
$$

則個體答錯的機率則為 $1-\pi_i(\eta)$。根據伯努利分配，我們可以將個體的作答機率表示為

$$
\mathbb{P}(x_i|\eta) = \pi_i(\eta)^{x_i} [1-\pi_i(\eta)]^{1-x_i}
$$

令 $x_1, x_2,...,x_I$ 表示個體於 $I$ 道試題之作答反應，則在局部獨立的假設下，個體的作答機率可以寫為

$$
\mathbb{P}(x_1,x_2,...,x_I|\eta) = \prod_{i=1}^I \pi_i(\eta)^{x_i} [1-\pi_i(\eta)]^{1-x_i}
$$

### 單參數、二參數、三參數邏輯斯模型

IRT的發展過程中，最為著名的即單參數、二參數、以及三參數邏輯斯模型。

單參數邏輯斯（one-parameter logistic，簡稱1PL）模型，亦稱作 Rasch 模型，其假設每個題目有不同的難度，將 IRF 刻畫為

$$
\pi_i(\eta) = \frac{\exp(\eta-\alpha_i)}{1 + \exp(\eta-\alpha_i)}
$$

這裡，$\alpha_i$ 表示試題 $i$ 之難度參數。

二參數邏輯斯（two-parameter logistic，簡稱2PL）模型進一步引入鑑別力，其 IRF 為

$$
\pi_i(\eta) =  \frac{\exp[\beta_i(\eta-\alpha_i)]}{1 + \exp[\beta_i(\eta-\alpha_i)]}
$$

這裡，$\beta_i$ 表示試題 $i$ 之鑑別力參數。

三參數邏輯斯（three-parameter logistic，簡稱3PL）模型進一步引入個體之猜測機率，其 IRF 為

$$
\pi_i(\eta) = \gamma_i + (1-\gamma_i) \frac{\exp[\beta_i(\eta-\alpha_i)]}{1 + \exp[\beta_i(\eta-\alpha_i)]}
$$

這裡，$\gamma_i$ 表示試題 $i$ 之猜測機率。


### 誤差分配之拓展
令 $\tau_i = \beta_i(\eta - \alpha_i)$ 表示個體於試題 $i$ 的「真實分數」，而$\epsilon_i$ 表示一來自位置參數為0，尺度參數為1邏輯斯分配之誤差，則前述2PL之IRF可以被理解為

$$
\begin{aligned}
\mathbb{P}(x_i|\eta) &= \mathbb{P}(\tau_i + \epsilon > 0|\eta) \\
&= \mathbb{P}(\epsilon > - \tau_i|\eta) \\
&= \mathbb{P}(- \epsilon < \tau_i|\eta) \\
&= \mathbb{P}(\epsilon < \tau_i|\eta) \\
&=  \frac{\exp[\tau_i]}{1 + \exp[\tau_i]}
\end{aligned}
$$

這裡，$\frac{\exp(\tau_i)}{1 + \exp(\tau_i)}$ 為邏輯斯分配之CDF。

事實上，可以透過對於 $\epsilon_i$ 分配的改變獲得不同的IRT模型，比如說，常態肩型（normal ogive）模型即假設 $\epsilon_i$ 為標準常態分配，因此，其IRT 為

$$
\pi_i(\eta) = \int_{-\infty}^{\tau_i} \phi(z) dz
$$

這裡，$\phi(z)$ 表示標準常態分配的pdf