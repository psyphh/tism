線性迴歸
================================

線性迴歸（linear regression）可說是統計建模的基礎，其試圖建立一線性函數（linear function），以描述兩變項 $x$ 與 $y$ 之間的關係。這裡，$x=(x_1,x_2,..., x_P)$為一 $P$ 維之向量，其常被稱獨變項（independent variable）、共變量（covariate），或是特徵（feature），而 $y$ 則為一純量（scalar），其常被稱作依變項（dependent variable）或是反應變項（response variable）。

在此主題中，我們將會學習以下的重點：

1. 使用線性函數刻畫兩變項間的關係。

2. 使用最小平方法（least squares method）來建立參數之估計準則。

3. 使用一階導數（derivative）來刻畫最小平方估計值（estimate）之最適條件（optimality condition）

4. 使用線性代數來解決線性迴歸之問題。



## 線性迴歸模型
廣義來說，迴歸分析試圖使用一 $P$ 維之向量 $x$，對於 $y$ 進行預測，其假設 $x$ 與 $y$ 存在以下的關係

$$y=f(x)+\epsilon$$

這裡，$f(x)$ 表示一函數，其描述了 $x$ 與 $y$ 系統性的關係，而 $\epsilon$ 則表示一隨機誤差，其平均數為0，變異數為 $\sigma_{\epsilon}^2$，即 $\epsilon \sim (0,\sigma_{\epsilon}^2)$。

線性迴歸模型假設 $f(x)$ 為一線性函數，即

$$f(x) = w_0 + w_1 x_1 + w_2 x_2 + ... + w_P x_P$$

這裡，$w_p$ 為 $x_p$ 所對應之迴歸係數（regression coefficient），其亦被稱作權重（weight），反映 $x_p$ 每變動一個單位時，預期 $y$ 跟著變動的量，$w_0$ 則稱作截距（intercept），亦稱作偏誤（bias），其反映當 $x_1,x_2,..,x_P$皆為0時，我們預期 $y$ 的數值。利用連加的符號，$f(x)$可簡單的寫為

$$f(x) = w_0 + \sum_{p=1}^P w_p x_p$$

無論是迴歸係數 $w_p$ 或是截距 $w_0$，由於其刻畫了 $f(x)$ 的形狀，故其皆被稱作模型參數（model parameter），文獻中常簡單以一 $P+1$ 維之向量 $w = (w_0, \beta_1, ..., \beta_P)$ 來表示模型中的所有參數，透過此向量的形式表徵，我們可以將 $f(x)$ 寫為以下更簡單的形式

$$
f(x) = x^T w
$$

這裡，$x^T$ 表示 $x=(1,x_1,x_2,...,x_P)$ 此 $P + 1$ 維向量之轉置（transpose）。


線性迴歸分析的主要目的乃透過一樣本資料，獲得對於 $w$ 之估計 $\widehat{w}$，一方面對於斜率與截距進行推論，二方面則是使用 $\widehat{f}(x) = \widehat{w}_0 + \sum_{p=1}^P \widehat{w}_p x_p$ 對 $y$ 進行預測。


## 最小平方估計法
線性迴歸分析主要採用最小平方法（least squares method，簡稱 LS 法）以對模型參數進行估計。令$(x_n, y_n)$ 表示第 $n$ 位個體於 $x$ 與 $y$ 之觀測值，則給定一隨機樣本 $\{(x_n, y_n) \}_{n=1}^N$，LS 估計準則定義為

$$\begin{aligned}
\mathcal{D}(w)
= &\frac{1}{N} \sum_{n=1}^N \left (y_n - x_n^T w \right )^2
\end{aligned}$$

由於迴歸模型假設 $y=f(x)+\epsilon$，且線性迴歸僅考慮線性的關係 $f(x) = x_n^T w$，因此，第 $n$ 筆觀測值對應之殘差可寫為 $\epsilon_n = y_n - x_n^T w$，而 LS 估計準則可簡單地寫為

$$
\begin{aligned}
\mathcal{D}(w)
= &\frac{1}{N} \sum_{n=1}^N \epsilon_n^2
\end{aligned}
$$

LS估計法的目標在於找到一估計值 $\widehat{w} = (\widehat{w}_0, \widehat{w}_1, ..., \widehat{w}_P)$，其最小化 LS 估計準則，意即，$\widehat{w}$ 可最小化樣本資料中所有殘差的平方和。


## 一階導數與最適條件
根據定義，$w$ 的 LS 估計式 $\widehat{w}$ 必須最小化 LS 估計準則，即 $\widehat{w}$ 需滿足

$$
\mathcal{D}(\widehat{w}) = \min_{w} \mathcal{D}(w)
$$

而當最小化 $\mathcal{D}(w)$ 的 $w$ 存在唯一性（uniqueness）時，則我們可以透過以下的表達式來說明 $\widehat{w}$ 之意涵：

$$
\widehat{w} = \text{argmin}_{w} \mathcal{D}(w)
$$

意即，$\widehat{w}$ 是那一個唯一最小化 $\mathcal{D}(w)$ 的數值。

根據優化理論（optimization theory），當 $\mathcal{D}(w)$ 為可微分之函數時，$\widehat{w}$ 為 $\mathcal{D}(w)$ 局部極小元（local minimizer）的必要條件（necessary condition）為

$$
\begin{aligned}
\nabla \mathcal{D}(\widehat{w}) =
\begin{pmatrix}
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_0}  \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_1} \\
   \vdots \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_P}
 \end{pmatrix}
 =\begin{pmatrix}
  0  \\
  0 \\
   \vdots \\
  0
 \end{pmatrix}
\end{aligned}
$$

意即，$\mathcal{D}(w)$ 的梯度（gradient），在 $\widehat{w}$ 的數值上必須等於0，此條件被稱作一階最適條件（first-order optimality condition）。由於一階最適條件，僅為 $\widehat{w}$ 作為 $\mathcal{D}(w)$ 局部極小元之必要條件，而非充分必要條件（sufficient and necessary condition），因此，我們仍需透過二階最適條件（second-order optimality condition）來檢驗，即判斷以下矩陣是否為正定矩陣（positive definite matrix）

$$
\nabla^2 \mathcal{D}(\widehat{w}) =
\begin{pmatrix}
  \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_0 \partial w_0} &
    \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_0 \partial w_1} &
    \cdots & \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_0 \partial w_P} \\
  \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_1 \partial w_0} &
    \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_1 \partial w_1} &
    \cdots & \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_1 \partial w_P} \\
   \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_P \partial w_0} &
    \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_P \partial w_1} &
    \cdots & \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_P \partial w_P}
 \end{pmatrix}
$$
這裡，$\nabla^2 \mathcal{D}(\widehat{w})$ 為 $\mathcal{D}(\widehat{w})$ 的二階微分矩陣，尺寸為 $(P +1) \times (P + 1)$，其亦被稱作黑塞矩陣（hessian matrix）。若 $\nabla^2 \mathcal{D}(\widehat{w})$為正定矩陣，則意味著對於所有不為0的 $P+1$ 維向量 $v$，我們有$v^T\nabla^2 \mathcal{D}(\widehat{w}) v >0$，這表示在 $\widehat{w}$ 附近，考慮任何方向的向量 $v$，其切線皆展現遞增的狀況，故 $\widehat{w}$ 為局部極小元。

因此，根據一階與二階最適條件，當要對 $\mathcal{D}(w)$ 進行最小化，求得 $\widehat{w}$時，其步驟為：

1. 計算 $\mathcal{D}(w)$ 之梯度，即 $\nabla \mathcal{D}(w)$。
2. 獲得 $\nabla \mathcal{D}(\widehat{w})=0$ 此聯立方程組之解。
3. 檢驗 $\nabla^2 \mathcal{D}(\widehat{w})$ 此矩陣是否為正定矩陣。




## 線性代數與迴歸
線性迴歸的問題可以簡單的使用矩陣與向量的方式來表徵

$$y = X w + \epsilon,$$
$$\mathop{\begin{pmatrix}
  y_{1} \\
  y_{2} \\
  \vdots \\
  y_{N}
 \end{pmatrix}}_{N \times 1}=
\mathop{\begin{pmatrix}
  1 & x_{11} & \cdots & x_{1P} \\
  1 & x_{21} & \cdots & x_{2P} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  1 & x_{N1} & \cdots & x_{NP}
 \end{pmatrix}}_{N \times (P+1)}
\mathop{\begin{pmatrix}
  b \\
  w_{1} \\
  \vdots \\
  w_{P}
 \end{pmatrix}}_{(P+1) \times 1}+
 \mathop{\begin{pmatrix}
  \epsilon_{1} \\
  \epsilon_{2} \\
  \vdots \\
  \epsilon_{N}
 \end{pmatrix}}_{N \times 1}.$$

在這邊我們於符號使用上有稍微偷懶， $y$ 與 $\epsilon$ 在這用於表徵 $N$ 維的向量。在前式的表徵下，LS 估計準則可以寫為

$$\begin{aligned}
\mathcal{D}(w) & =  \frac{1}{N} \sum_{n=1}^N \epsilon_n^2 \\
  & = \frac{1}{N} \epsilon^T \epsilon\\
 & =  \frac{1}{N} (y-Xw)^T(y-Xw).
\end{aligned}$$

透過對 $w$ 的每一個成分做偏微分，即計算 $\mathcal{D}(w)$ 的梯度（gradient），我們可以得到刻畫 LS 解的一階條件

$$\begin{aligned}
\nabla \mathcal{D}(\widehat{w}) &=
\begin{pmatrix}
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_0}  \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_1} \\
   \vdots \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_P}
 \end{pmatrix} \\
&=-\frac{2}{N} X^T(y-X\widehat{w})=0.
\end{aligned}$$

此純量函數對向量（scalar function by vector）的微分的計算，可按照定義對 $b$ 與各個 $w_p$ 進行微分後，再利用矩陣乘法之特性獲得。除此之外，亦可以參考矩陣微分（matrix calculus）中，對於[純量對向量微分之規則](https://en.wikipedia.org/wiki/Matrix_calculus#Scalar-by-vector_identities)。$-\frac{2}{N} X^T(y-X\widehat{w})=0$ 意味著 $\widehat{w}$ 需滿足以下的等式

$$
X^T X\widehat{w}=X^Ty.
$$

因此，若 $X^T X$ 存在反矩陣，則迴歸係數的 LS 估計值可寫為

$$
\widehat{w} = (X^T X)^{-1} X^Ty
$$

確保 $X^T X$ 存在反矩陣的數學條件為其各直行向量（column vector）間並未存在線性相依（linear dependence）的狀況。所謂向量間有線性相依指的是某個向量，可以寫為其它向量的線性組合。一般來說，當樣本數大於變項數（$N > P$），各變項間變異數皆大於0，且皆存在其獨特的訊息時，$X^T X$ 為可逆的。$(X^T X)^{-1}$ 的計算，可採用[高斯消去法](https://en.wikipedia.org/wiki/Gaussian_elimination)（Gaussian elimination）或是[QR分解](https://en.wikipedia.org/wiki/QR_decomposition)（QR decomposition），其計算複雜度皆為 $O(P^3)$。


