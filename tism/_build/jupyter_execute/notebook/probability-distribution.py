機率分佈
================================




## 隨機變數與其分佈特徵

### 機率
如果一現象，我們事先可知道其可能之結果範疇，但無法得知其結果之特定結果的話，我們會稱該現象為隨機的現象（stochastic phenomenon），而機率（probability）則常用於描述特定結果之發生強度。

在了解機率時，我們須了解以下詞彙之意涵：

+ 對隨機現象進行單次的觀察，稱作嘗試（trial）或實驗（experiment）。
+ 一嘗試所觀察到或量測到的數值，被稱作結果（outcome）。
+ 所有結果之集合，被被稱作樣本空間（sample space），常以 $S$ 來表示。
+ 單一或多個結果形成之集合，被稱作一事件（event），常以 $E$ 來表示。

考慮扔躑一六面之骰子，則每扔躑一次骰子就是一嘗試或實驗，而此嘗試的可能結果為1、2、3、4、5、或6，共六種結果，因此，其樣本空間為$S =  \{1, 2, 3,4,5,6 \}$，而任一將不同的結果組合起來，如單數 $E=\{1, 3, 5\}$、小於3之結果$E=\{1,2\}$、沒有任何結果 $E = \{ \emptyset \}$，皆為事件的例子。

簡單的來說，機率是一個介於0到1之間的數字，用來度量特定事件 $E$ 發生的可能性，其必須滿足

1. $0 \leq \mathbb{P}(E) \leq 1$；
2. $\mathbb{P}(S) = 1$。

若從比較嚴格的角度來定義機率的話，我們必須先定義一sigma algebra（或稱Borel field），其符號為 $\mathcal{B}$，其由 $S$ 的子集合所構成並滿足

1. $\emptyset \in \mathcal{B}$；
2. 若 $E \in \mathcal{B}$，則 $E^C \in \mathcal{B}$；
3. 若 $E_1, E_2,... \in \mathcal{B}$，則 $\cup_{i=1}^{\infty} E_i \in \mathcal{B}$

此時，機率函數（probability function）為一定義於 $\mathcal{B}$之函數，表示為$\mathbb{P}$，其滿足

1. 對於所有的 $E \in \mathcal{B}$，$\mathbb{P}(E)\geq 0$；
2. $\mathbb{P}(S) = 1$；
3. 若 $E_1, E_2,...$ 互為配對不相交（pairwise disjoint）之事件，則 $\mathbb{P}(\cup_{i=1}^{\infty} E_i) = \sum_{i=1}^{\infty} \mathbb{P}(E_i)$。

### 隨機變數與分配函數
隨機變數 $X$ 為一函數，其將樣本空間 $S$ 中的事件 $A$，轉換為一實數 $x \in \mathcal{X}$。因此，完整的轉換歷程為 $x = X(A)$。

隨機變數 $X$ 的分配函數（distribution function），常以 $F_X(x)$ 來表示，其定義為

$$
F_X(x) = \mathbb{P}(X \leq x)
$$

因此，分配函數可用於刻畫一隨機變數的行為表現。由於分配函數刻畫的是該隨機變數 $X$ 小於等於 $x$ 之機率，因此，其又被稱作累積分配函數（cumulative distribution function，簡稱CDF）。

當 $X$ 為間斷之隨機變數（discrete random variable）時，意即，$X$ 的實現值僅為有限的數值時，則我們可以使用機率質量函數（probability mass function，簡稱PMF）來刻畫其行為，其被定義為

$$
f_X(x) = \mathbb{P}(X=x)
$$

當 $X$ 為連續隨機變數（continuous random variable）時，意即，$X$ 的實現值可為實數軸上某區間的任一數值時，此時，$X$ 於任一實現值的機率都是 0，因此，我們無法直接透過 $\mathbb{P}(X=x)$ 來刻畫其隨機行為。取而代之的是，使用 $\mathbb{P}(X \in [x - \delta, x + \delta])$ 來度量 $X$ 在 $x$ 附近的發生機率，這裡，$\delta$ 表示一很小的數值。因此，我們可使用機率密度函數（probability density function，簡稱PDF）來連續隨機變數於 $x$ 附近之發生機率，其被定義為

$$
f_X(x) = \frac{d F_X(x)}{d x}
$$

前式意味著CDF可以寫為

$$
F_X(x)= \int_{-\infty} ^{x} f_X(u) du
$$

不過，要特別注意的是，隨機變數一定存在其所對應之CDF，但不一定有PDF。


### 期望值與變異數
隨機變數之期望值（expectation）被定義為：

$$
\mathbb{E}(X) =
\begin{cases}
\sum_{x} x \cdot f_X(x) & \text{ if } X \text{ is discrete;} \\
\int_{x} x \cdot f_X(x) dx & \text{ if } X \text{ is continuous.}
\end{cases}
$$

期望值可以被理解為該隨機變數之平均值。前述之期望值，常使用 $\mu_X$ 來表示，意即，$\mu_X=\mathbb{E}(X)$。

由於連加運算子 $\sim$ 事實上可視為積分運算子 $\int$ 於間斷度量（discrete measure）時之特例，因此，在之後談及期望值時，不管是間斷或是連續的隨機變數，我們僅使用 $\mathbb{E}(X) = \int_{x} x \cdot f_X(x) dx$ 來表示。

隨機變數之變異數（variance）則被定義為：

$$
\mathbb{V}\text{ar}(X) = \mathbb{E}\left[(X - \mu_X)^2 \right] =
\int_{x} (x - \mu_X)^2 \cdot f_X(x) dx
$$

變異數反映該隨機變數離期望值距離平方之平均，其常使用 $\sigma_X^2$ 來表示。

由於積分具有線性之性質，因此，若定義一隨機變數 $Y = a + b X$，則

$$
\begin{aligned}
\mathbb{E}(Y) &= a + b \mu_X, \\
\mathbb{V}\text{ar}(Y) &= b^2 \sigma_X^2.
\end{aligned}
$$

此關係式在推導隨機變得共變結構時相當重要。


## 常見隨機變數之分配

### Binomial 分配
當一個嘗試滿足以下的條件時，其被稱作伯努利嘗試（Bernoulli trial）

1. 每次實驗的結果，只有成功與失敗兩種可能。
2. 隨機變數 $X$ 用來指示（indicate）是否成功，即成功時，$X=1$，失敗時，$X=0$。
3. 成功與失敗的機率分別為 $\pi$ 與 $1-\pi$。

此時，$X$ 表示一服從伯努利分配之隨機變數，我們寫作

$$
X \sim \text{Bernoulli}(\pi)
$$

而 $X$ 的PMF可以寫為

$$
f_X(x) = \pi^x (1-\pi)^{(1-x)}
$$

根據定義，我們可以計算伯努利隨機變數的期望值與變異數，即

$$
\begin{aligned}
\mathbb{E}(X) &= \sum_{x=0}^1 x \pi^x (1-\pi)^{(1-x)}\\
&=  \pi^1 (1-\pi)^{0}\\
&= \pi\\
\end{aligned}
$$

與

$$
\begin{aligned}
\mathbb{V} \text{ar}(X) &= \sum_{x=0}^1 (x - \pi)^2 \pi^x (1-\pi)^{(1-x)}\\
&=  (0 - \pi)^2 (1-\pi) + (1 - \pi)^2 \pi \\
&=  \pi^2 -\pi^3 + \pi - 2 \pi^2 + \pi^3 \\
&=  \pi -  \pi^2  \\
&=  \pi (1-  \pi)  \\
\end{aligned}
$$


伯努利的隨機變數具有一特別的性質，若 $X_1, X_2,...,X_N$ 皆來自 $\text{Bernoulli}(\pi)$，且 $X_1, X_2,...,X_N$ 彼此統計獨立（見下一小節），則定義 $Y = X_1 + X_2 + ... +X_N$，$Y$ 的分配被稱作二項式分配（binomial distribution），其 PMF 為

$$
f_Y(y) = {N \choose y} \pi^{y} (1-\pi)^{(N-y)}
$$

這裡，${N \choose y} = \frac{N!}{y! (N-y)!}$ 而 $y! = y(y-1)(y-2) \cdots \times 2 \times 1$。二項式分配的期望值與變異數，可以直接使用多個隨機變數之線性組合獲得（見隨機向量之期望值與變異數的部分），其表達式為

$$
\mathbb{E}(X) = N \pi
$$

與

$$
\mathbb{V} \text{ar}(X) = N \pi (1-\pi)
$$


### 常態分配
許多大自然的現象，皆展現常態分配的樣貌，常態分配的隨機變數 $X$，其PDF為一對稱的鐘形曲線，表達式為

$$
f_X(x) = \frac{1}{\sqrt{2 \pi} \sigma_X} e^{-(x - \mu_X)^2/2 \sigma_X^2}
$$

常態分配隨機變數的期望值為 $\mu_X$，變異數為 $\sigma_X^2$，然而，若要照定義進行計算的話，則需要使用到變數變換（change of variables）與分部積分法（integral by parts）的技巧，有興趣的讀者，可以自行去找一本數理統計學的課本來閱讀。

常態分配有個重要的特性是，任何常態分配隨機變數的線性組合，不論參數數值為何以及是否獨立，仍然會是常態分配，此特性在許多模型的建立與理論推導上，扮演重要的角色。

## 多個隨機變數之分佈

### 隨機向量
當我們有多個隨機變數 $X_1, X_2,...,X_P$ 時，我們可以將其排成一個隨機向量 $X = (X_1, X_2,...,X_P)$，$X_p$ 用於表示 $X$ 的第 $p$ 個隨機變數。$X$ 此隨機向量的行為可以使用分配函數表示：

$$
\begin{aligned}
F(x) &= F(x_1,x_2,...,x_P) \\
&=  \mathbb{P}(X_1 \leq x_1, X_2 \leq x_2,...,X_P \leq x_P)
\end{aligned}
$$
這裡，$x=(x_1,x_2,...,x_P)$ 表示一 $P$ 維向量，其內部元素為 $x_p$。

當 $X$ 的內部元素皆為間斷變數時，其PMF為

$$
\begin{aligned}
f(x) &= f(x_1,x_2,...,x_P) \\
& =  \mathbb{P}(X_1 = x_1, X_2 = x_2,...,X_P = x_P)
\end{aligned}
$$

而當$X$ 的內部元素皆為連續變數時，其PDF為

$$
\begin{aligned}
f(x) &= f(x_1,x_2,...,x_P) \\
&=\frac{\partial^P F(x)}{\partial x_1 \partial x_2 ... \partial x_P}
\end{aligned}
$$

### 聯合分佈、邊際分佈、與條件分佈
無論是 $F(x)$ 與 $f(x)$，其都用於刻畫 $X_1, X_2, ..., X_P$ 之聯合分佈（joint distribution），其反映的是 $X_1, X_2, ..., X_P$ 共 $P$ 個隨機變數聯合起來的隨機表現。

令 $f_{X_1X_2}(x_1, x_2)$ 表示 $X_1$ 與 $X_2$ 聯合分佈之 PMF/PDF，則 $X_1$ 邊際分佈（marginal distribution）的 PMF/PDF，可以寫為

$$
f_{X_1}(x_1) = \int f_{X_1X_2}(x_1, x_2) d x_2
$$

事實上，$f_{X_1}(x_1)$ 即為 $X_1$ 分佈之 PMF/PDF。一般來說，若 $X = (X_1, X_2, ...,X_P)$，則 $X_p$ 的邊際分佈可以透過以下方式獲得

$$
f_{X_p}(x_p) = \int  ... \int \int ... \int f_{X}(x) d x_1 ...d x_{p-1} d x_{p+1} ...d x_{P}
$$

令 $f_{X_1X_2}(x_1, x_2)$ 表示 $X_1$ 與 $X_2$ 聯合分佈之 PMF/PDF，$f_{X_1}(x_1)$ 表示 $X_1$ 之邊際分佈，則 $X_2$ 在給定 $X_1=x_1$ 之下的條件分佈（conditional distribution）為

$$
f_{X_2|X_1}(x_2| x_1) = \frac{f_{X_1X_2}(x_1, x_2)}{f_{X_1}(x_1)}
$$

這裡，我們假設 $f_{X_1}(x_1) \neq 0$。

條件分佈在統計建模中，扮演相當關鍵的角色，舉例來說，線性迴歸分析事實上可以被理解為

$$
Y | X = x \sim \mathcal{N}(w_0 +\sum_{p=1}^P w_p x_p , \sigma^2_{\epsilon})
$$

意即，在給定 $X = x$ 的條件下，$Y$ 的分配為平均數為 $w_0 +\sum_{p=1}^P w_p x_p$，變異數為 $\sigma^2_{\epsilon}$ 之常態分配。這裡，$w_p$ 為 $x_p$ 所對應之迴歸係數。

另外，邏輯斯迴歸亦可以被理解為

$$
Y | X = x \sim \text{Bernoulli}(\pi(x))
$$
意即，在給定 $X = x$ 的條件下，$Y$ 的分配為成功機率為 $\pi(x)$ 之伯努利分配。這裡，$\pi(x) = \frac{\exp(w_0 +\sum_{p=1}^P w_p x_p)}{1 + \exp(w_0 +\sum_{p=1}^P w_p x_p)}$。




### 隨機向量之期望值與變異數
令 $X = (X_1, X_2,...,X_P)$ 表示一隨機向量，則該向量之期望值向量（expectation vector）與共變數矩陣（covariance matrix）被定義為

$$
\mu_X = \mathbb{E} (X) 
= \begin{pmatrix}
\mathbb{E} (X_1) \\
\mathbb{E} (X_2) \\
\vdots \\
\mathbb{E} (X_P)
\end{pmatrix}
=
\begin{pmatrix}
\mu_{X_1} \\
\mu_{X_2} \\
\vdots \\
\mu_{X_P}
\end{pmatrix}
$$

與

$$
\begin{aligned}
\Sigma_X &= \mathbb{C}\text{ov} (X) \\
&=
\begin{pmatrix}
\mathbb{C}\text{ov} (X_1, X_1) & \mathbb{C}\text{ov} (X_1, X_2) & \cdot & \mathbb{C}\text{ov} (X_1, X_P)  \\
\mathbb{C}\text{ov} (X_2, X_1) & \mathbb{C}\text{ov} (X_2, X_2) & \cdot & \mathbb{C}\text{ov} (X_2, X_P) \\
\vdots & \vdots & \ddots & \vdots \\
\mathbb{C}\text{ov} (X_P, X_1) & \mathbb{C}\text{ov} (X_P, X_2) & \cdot & \mathbb{C}\text{ov} (X_P, X_P) 
\end{pmatrix} \\
&=
\begin{pmatrix}
\sigma_{X_1}^2 & \sigma_{X_1 X_2} & \cdot & \sigma_{X_1 X_P}  \\
\sigma_{X_2 X_1} & \sigma_{X_2}^2 & \cdot & \sigma_{X_2 X_P} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{X_P X_1} & \sigma_{X_P X_2} & \cdot & \sigma_{X_P}^2
\end{pmatrix}
\end{aligned}
$$

這裡，$\mathbb{E} (X_p)$ 與 $\mathbb{C}\text{ov} (X_p, X_q)$ 的定義分別為

$$
\begin{aligned}
\mathbb{E} (X_p) &= \int x_p f_X(x) dx \\
&= \int x_p f_{X_p}(x_p) dx_p
\end{aligned}
$$

與

$$
\begin{aligned}
\mathbb{C}\text{ov} (X_p, X_q) &= \int (x_p- \mu_{X_p}) (x_q- \mu_{X_q}) f_X(x) dx \\
&= \int \int (x_p- \mu_{X_p}) (x_q- \mu_{X_q}) f_{X_p X_q}(x_p,x_q) dx_p dx_q
\end{aligned}
$$

另外，如果我們對 $X$ 進行線性轉換，即 $Y = b + A X$，則我們有

$$
\mu_Y = \mathbb{E} (Y) = \mathbb{E} (b + AX) = b + A \mu_X
$$

與

$$
\Sigma_Y = \mathbb{C}\text{ov} (Y) = \mathbb{C}\text{ov} (b + AX) = A \Sigma_X A^T
$$

其中，一個最簡單的應用是 $Y= 1^T X = \sum_{p=1}^P X_p$，這裡，$1 = \underbrace{(1,1,...,1)}_{P \text{個} 1}$，此時，

$$
\begin{aligned}
\mu_Y &= 1^T\mu_X = \sum_{p=1}^P \mu_{X_p}, \\
\Sigma_Y &= 1^T \Sigma_X 1 = \sum_{p=1}^P \sum_{q=1}^P \sigma_{X_p X_q}.
\end{aligned}
$$



### 獨立與條件獨立
兩隨機變數 $X_1$ 與 $X_2$，若滿足以下條件時，則稱作統計獨立（statistically independent）

$$
f_{X_1X_2}(x_1, x_2) = f_{X_1}(x_1) f_{X_2}(x_2)
$$

因此，我們可以僅透過 $X_1$ 與 $X_2$ 各自的邊際分配，得到其聯合之分配。

然而，在進行統計建模時，獨立的假設大多過於強烈，因此，常採用的是條件獨立（conditional independence）之假設。對於所有的 $Z = z$，若滿足以下條件時，我們稱 $X_1$ 與 $X_2$ 兩者條件獨立：

$$
f_{X_1X_2|Z}(x_1, x_2|z) = f_{X_1|Z}(x_1|z) f_{X_2|Z}(x_2|z)
$$

利用條件獨立之假設，令 $\text{pa}(X_p)$ 表示所有對 $X_p$ 造成影響的變項，則利用條件獨立之假設，我們可以使用以下的方法來建構 $X_1, X_2,..., X_P$ 之聯合分佈

$$
f_X(x_1,x_2,...,x_P) = \prod_{p=1}^P f_{X_p|\text{pa}(X_p)}(x_p|\text{pa}(X_p))
$$

