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
\begin{aligned}
p(x_i|\eta) &= \mathbb{P}(x_i|\eta) \\
&= \pi_i(\eta)^{x_i} [1-\pi_i(\eta)]^{1-x_i}
\end{aligned}
$$

令 $x_1, x_2,...,x_I$ 表示個體於 $I$ 道試題之作答反應，則在局部獨立的假設下，個體的作答機率可以寫為

$$
\begin{aligned}
p(x_1,x_2,...,x_I|\eta) &= \mathbb{P}(x_1,x_2,...,x_I|\eta) \\
&= \prod_{i=1}^I \pi_i(\eta)^{x_i} [1-\pi_i(\eta)]^{1-x_i}
\end{aligned}
$$

有時，我們會將多道試題的反應簡單寫為一向量，即 $x = (x_1, x_2,...,x_I)$ ，此時，我們亦可將前述之混合機率分佈簡化為 $p(x|\eta)=p(x_1,x_2,...,x_I|\eta)$。

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
\mathbb{P}(x_i = 1|\eta) &= \mathbb{P}(\tau_i + \epsilon > 0|\eta) \\
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

這裡，$\phi(z)$ 表示標準常態分配的pdf。

## 進階試題反應理論模型

### 等級反應模型
等級反應模型（graded response model，簡稱GRM）常用於刻畫 Likert 量尺之作答，令 $x_i \in \{0,1,...,C_i-1\}$ 表示一次序之試題反應變項，將 $\mathbb{P}(x_i \geq j | \eta)$ 刻畫為

$$
\begin{aligned}
\mathbb{P}(x_i \geq 0 | \eta) &= 1,\\
\mathbb{P}(x_i \geq 1 | \eta) &= \frac{\exp(\nu_{i1} + \lambda_{i} \eta)}{1 + \exp(\nu_{i1} + \lambda_{i} \eta)},\\
\vdots,\\
\mathbb{P}(x_i \geq C_i-1 | \eta) &= \frac{\exp(\nu_{i(C_i-1)} + \lambda_{i} \eta)}{1 + \exp(\nu_{i(C_i-1)} + \lambda_{i} \eta)},\\
\mathbb{P}(x_i \geq C_i | \eta) &= 0,
\end{aligned}
$$

這裡，$\nu_{ij}$ 表示對於 $x_i \geq 1$ 此事件之截距，$\lambda_{i}$ 則為因素負荷量。因此，我們有 $\mathbb{P}(x_i = j | \eta) = \pi_{ij}(\eta) = \mathbb{P}(x_i \geq j-1 | \eta) - \mathbb{P}(x_i \geq j | \eta)$。



### 部分給分模型
部分給分模型（partial credit model，簡稱PCM）假設個體在作答試題時，採用一序列的模式處理

$$
\begin{aligned}
\mathbb{P}(x_i = 1 |x_i = 0 \text{ or } x_i = 1, \eta)  &= \frac{\exp(\eta - \alpha_{i1})}{1+\exp(\eta - \alpha_{i1})},\\
\mathbb{P}(x_i = 2 |x_i = 1 \text{ or } x_i = 2, \eta)  &= \frac{\exp(\eta - \alpha_{i2})}{1+\exp(\eta - \alpha_{i2})},\\
\vdots,\\
\mathbb{P}(x_i = C_i-1 |x_i = C_i-2 \text{ or } x_i = C_i-1, \eta) &= \frac{\exp(\eta - \alpha_{i(C_i-1)})}{1+\exp(\eta - \alpha_{i(C_i-1)})},\\
\end{aligned}
$$

前述機率隱含著 $\pi_{ij}(\eta) = \pi_{i(j-1)}(\eta) \exp(\eta - \alpha_{ij})$，在透過 $\pi_{i0}(\eta) + \pi_{i1}(\eta) +...+\pi_{i(C_i -1)}(\eta) =1$ 此限制式，$\mathbb{P}(x_i = j | \eta) = \pi_{ij}(\eta)$ 可以寫為

$$
\mathbb{P}(x_i = j | \eta) = \frac{\exp(\sum_{k=0}^j (\eta - \alpha_{ik}))}{\sum_{l=0}^{C_i-1}\exp(\sum_{k=0}^l (\eta - \alpha_{ik}))}.
$$
這裡，我們將 $\eta - \beta_{i0}$ 設為 0。

### 多向度試題反應模型
前述的試題反應模型，皆可採用某種方式拓展為多向度模型，以3PL為例，其IRF可以寫為

$$
\pi_i(\eta) = \gamma_i + (1-\gamma_i) \frac{\exp(\nu_i + \sum_{m=1}^M \lambda_{im} \eta_m)}{1 + \exp(\nu_i + \sum_{m=1}^M \lambda_{im} \eta_m)}
$$
這裡，$\nu_i$ 表示試題 $i$ 之截距，$\lambda_{im}$ 表示因素 $m$ 對試題 $i$ 之因素負荷量， $\gamma_i$ 則為試題 $i$ 的猜測參數。

## 試題反應模型之估計

### 聯合最大概似法
令 $\eta_n$ 表示個體 $n$ 之潛在能力，$\text{H}$ 表示由 $\eta_n$ 作為其第 $n$ 個橫列所形成之矩陣矩陣，而 $\theta$ 則表示由試題參數組成之向量，聯合最大概似法（joint maximum maximum likelihood method，簡稱JML法）考慮以下的估計準則，同時對試題參數 $\theta$ 與個體能力 $\text{H}$ 進行估計

$$
\ell_{\text{joint}}(\theta, \text{H}; \mathcal{X}) = \sum_{n=1}^N \sum_{i=1}^I
\left[
x_{ni} \log \pi_i(\eta_n) + (1- x_{ni}) \log (1-\pi_i(\eta_n))
\right]
$$
這裡，$mathcal{X}$ 表示由 $\{x_n\}_{n=1}^N$ 所形成的觀察資料。JML 的計算相對容易，但其缺點在於，其參數數目會隨著觀測值個數的增加而上升，一般來說，其無法獲得一致性之參數估計式。

### 邊際最大概似法
邊際最大概似法（marginal maximum likelihood method，簡稱MML法）僅使用 $x$ 的邊際分配來建立概似函數。令 $p(\eta; \theta)$ 表示潛在能力的機率分佈，其常被假定為多元常態分配，MML考慮以下之估計準則

$$
\ell_{\text{marginal}}(\theta; mathcal{X}) = \sum_{n=1}^N \log p(x_n; \theta)
$$

這裡，

$$
\begin{aligned}
p(x_n; \theta) &=
\int
p(x_n, \eta_n; \theta)
d \eta_n \\
&=
\int \left[
\prod_{i=1}^I p(x_{n_i}|\eta_n; \theta)
\right]
p(\eta_n; \theta)
d \eta_n
\end{aligned}
$$

儘管此方法被稱作邊際最大概似法，但其事實上可以被視為一般的最大概似法，故具有最大概似法估計的理論特性。然而，由於前述的概似函數牽涉到積分，故在潛在能力向度大時不易計算。

### EM算則
實務上，IRT之參數估計多使用某種形式的EM算則，在此，我們以Cai（2009）的Metropolis–Hastings Robbins–Monro（MH-RM）算則為例。

在EM算則下，考慮的是完整資料之概似函數 $\ell_{\text{com}}(\theta)$，其可以寫為

$$
\ell_{\text{com}}(\theta; \mathcal{X}, \text{H}) =
\sum_{n=1}^N \sum_{i=1}^I \log p(x_{ni}|\eta_n; \theta) +
\sum_{n=1}^N \log p(\eta_n; \theta)
$$

而EM算則的 E 步驟在於計算該概似函數在給定觀察資料與當下參數估計下之條件期望值，即

$$
Q(\theta| \widehat{\theta}^{(t)}) = \mathbb{E}(\ell_{\text{com}}(\theta; \mathcal{X}, \text{H}) | \mathcal{X}, \widehat{\theta}^{(t)})
$$

而根據Fisher's identity，我們有

$$
\nabla_{\theta} \ell_{\text{marginal}}(\theta; \mathcal{X}) =
 \mathbb{E}( \nabla_{\theta}\ell_{\text{com}}(\theta;\mathcal{X}, \text{H}) | \mathcal{X},\theta)
$$

這意味著，我們可以直接透過完整資料該似函數的期望值來計算邊際概似函數之梯度，以進行參數的更新。

在 MH-RM 法的第 $t$ 個步驟時，我們使用 Metropolis-Hasting方法，於 $p(\eta_n|x_n; \widehat{\theta}^{(t)})$ 此分配中產生 $\eta^{(t)}_1, \eta^{(t)}_2, ..., \eta^{(t)}_N$，獲得對於MML估計準則梯度的估計

$$
d^{(t)} =
\sum_{n=1}^N \sum_{i=1}^I \nabla_{\theta}\log p(x_{ni}|\eta_n^{(t)}; \theta) +
\sum_{n=1}^N \nabla_{\theta}\log p(\eta_n^{(t)}; \theta)
$$

接著，再根據Robbins-Monro的方法，利用此梯度更新參數估計

$$
\widehat{\theta}^{(t + 1)} =
\widehat{\theta}^{(t)} + \frac{1}{t} d^{(t)}
$$

