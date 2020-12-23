線性迴歸
================================

**線性迴歸**（linear regression）可說是統計建模的基礎，其試圖建立一線性函數（linear function），以描述兩變項 $x$ 與 $y$ 之間的關係。這裡，$x=(x_1,x_2,..., x_J)$為一 $J$ 維之向量，其常被稱**獨變項**（independent variable）、**共變量**（covariate），或是**特徵**（feature），而 $y$ 則為一純量（scalar），其常被稱作**依變項**（dependent variable）或是**反應變項**（response variable）。

在此主題中，我們將會學習以下的重點：

1. 使用線性函數刻畫兩變項間的關係。

2. 使用最小平方法（least squares method）來建立參數之估計準則。

3. 使用最適條件（optimality condition）來刻畫最小平方估計值（estimate）。

4. 使用線性代數來解決線性迴歸之問題。

5. 利用最小平方法估計準則來刻畫模型之適配度。


## 線性迴歸與最小平方法
### 線性迴歸模型
廣義來說，迴歸分析試圖使用一 $J$ 維之向量 $x$，對於 $y$ 進行預測，其假設 $x$ 與 $y$ 存在以下的關係

$$y=f(x)+\epsilon$$

這裡，$f(x)$ 表示一**迴歸函數**（regression function），其描述了 $x$ 與 $y$ 系統性的關係，而 $\epsilon$ 則表示一隨機誤差，其平均數為0，變異數為 $\sigma_{\epsilon}^2$，即 $\epsilon \sim (0,\sigma_{\epsilon}^2)$。

線性迴歸模型假設 $f(x)$ 為一線性函數（linear function），即

$$f(x) = w_0 + w_1 x_1 + w_2 x_2 + ... + w_J x_J$$

這裡，$w_j$ 為 $x_j$ 所對應之迴歸係數（regression coefficient），其亦被稱作權重（weight），反映 $x_j$ 每變動一個單位時，預期 $y$ 跟著變動的量，$w_0$ 則稱作截距（intercept），亦稱作偏誤（bias），其反映當 $x_1,x_2,..,x_J$皆為0時，我們預期 $y$ 的數值。利用連加的符號，$f(x)$可簡單的寫為

$$f(x) = w_0 + \sum_{j=1}^J w_j x_j$$

無論是迴歸係數 $w_j$ 或是截距 $w_0$，由於其刻畫了 $f(x)$ 的形狀，故其皆被稱作**模型參數**（model parameter），文獻中常簡單以一 $J+1$ 維之向量 $w = (w_0, w_1, ..., w_J)$ 來表示模型中的所有參數，透過此向量的形式表徵，我們可以將 $f(x)$ 寫為以下更簡單的形式

$$
f(x) = x^T w
$$

這裡，$x^T$ 表示 $x=(1,x_1,x_2,...,x_J)$ 此 $J + 1$ 維向量之轉置（transpose）。讀者需特別注意的是，在進入本章節時，$x = (x_1,x_2,...,x_J)$ 用於表示一 $J$ 維之共變量向量，但從此段落後，我們將 $x$ 重新定義為 $x = (1, x)$，以對模型做更為簡潔的表達。

線性迴歸分析的主要目的乃透過一樣本資料，獲得對於 $w$ 之估計 $\widehat{w}$，一方面對於斜率與截距進行推論，二方面則是使用 $\widehat{y} = \widehat{f}(x) = x^T \widehat{w}$ 對 $y$ 進行預測。


### 最小平方估計法
線性迴歸分析主要採用**最小平方法**（least squares method，簡稱 LS 法）以對模型參數進行估計。在建立 LS 估計準則之前，我們先定義平方損失函數（squared loss function）來度量 $y$ 與 $f(x)$ 之間的差異：

$$
L\left[ y, f(x) \right] = \left[ y - f(x) \right]^2
$$

在此損失函數之下，若 $y$ 與 $f(x)$ 越不一致，則 $L$ 所量出的差異性越大。因此，$L\left[ y, f(x) \right]$ 可以用來度量在特定的 $w$ 之下，資料 $y$ 與模型 $f(x) =x_n^T w$ 的適配性。

令 $\{(x_n, y_n) \}_{n=1}^N$ 表示一隨機樣本，這裡，$(x_n, y_n)$ 表示第 $n$ 位個體於 $x$ 與 $y$ 之觀測值，則 LS 法之適配函數（fitting function），或稱差異函數（discrepancy function），被定義為個別資料損失之平均，即

$$\begin{aligned}
\mathcal{D}(w)
= &\frac{1}{N} \sum_{n=1}^N L\left[ y_n, f(x_n) \right]
\end{aligned}$$

由於線性迴歸模型假設 $y_n=x_n^T w+\epsilon$，因此，概念上，第 $n$ 筆觀測值對應之殘差可寫為 $\epsilon_n = y_n - x_n^T w$，而 LS 估計準則可簡單地寫為

$$
\begin{aligned}
\mathcal{D}(w)
= &\frac{1}{N} \sum_{n=1}^N \epsilon_n^2
\end{aligned}
$$

LS估計法的目標在於找到一估計值（estimate） $\widehat{w} = (\widehat{w}_0, \widehat{w}_1, ..., \widehat{w}_J)$，其可最小化 LS 估計準則，意即，$\widehat{w}$ 能讓樣本資料中所有殘差的平方和達到最小。


## 適配函數之優化
### 優化之最適條件
根據定義，$w$ 的 LS 估計式 $\widehat{w}$ 必須最小化 LS 估計準則，利用數學符號來表示的話，$\widehat{w}$ 需滿足

$$
\mathcal{D}(\widehat{w}) = \min_{w} \mathcal{D}(w)
$$

而當最小化 $\mathcal{D}(w)$ 的 $w$ 存在唯一性（uniqueness）時，則我們可以透過以下的表達式來說明 $\widehat{w}$ 之意涵：

$$
\widehat{w} = \text{argmin}_{w} \mathcal{D}(w)
$$

意即，$\widehat{w}$ 是那一個唯一最小化 $\mathcal{D}(w)$ 的數值。

若 $\widehat{w}$ 為 $\mathcal{D}(w)$ 局部極小元（local minimizer），即對於 $\widehat{w}$ 周邊的 $w$ 來說（$w \neq \widehat{w}$），$\widehat{w}$ 滿足 $\mathcal{D}(\widehat{w}) \leq \mathcal{D}(w)$，則根據優化理論（optimization theory），當 $\mathcal{D}(w)$ 為可微分之函數時，$\widehat{w}$ 為局部極小元之必要條件（necessary condition）為

$$
\begin{aligned}
\nabla \mathcal{D}(\widehat{w}) =
\begin{pmatrix}
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_0}  \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_1} \\
   \vdots \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_J}
 \end{pmatrix}
 =\begin{pmatrix}
  0  \\
  0 \\
   \vdots \\
  0
 \end{pmatrix}
\end{aligned}
$$

意即，$\mathcal{D}(w)$ 的梯度（gradient），在 $\widehat{w}$ 的數值上必須等於0，此條件被稱作**一階最適條件**（first-order optimality condition）。

**二階最適條件**（second-order optimality condition）則進一步說明，當 $\widehat{w}$ 為局部極小元時，$\mathcal{D}(\widehat{w})$ 的二次微分矩陣於 $\widehat{w}$ 處需進一步滿足半正定矩陣（positive semidefinite matrix）之條件。$\mathcal{D}(\widehat{w})$ 的二次微分矩陣為一 $(J +1) \times (J + 1)$ 之矩陣，其被定義為

$$
\nabla^2 \mathcal{D}(\widehat{w}) =
\begin{pmatrix}
  \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_0 \partial w_0} &
    \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_0 \partial w_1} &
    \cdots & \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_0 \partial w_P} \\
  \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_1 \partial w_0} &
    \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_1 \partial w_1} &
    \cdots & \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_1 \partial w_J} \\
   \vdots & \vdots & \ddots & \vdots \\
  \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_P \partial w_0} &
    \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_P \partial w_1} &
    \cdots & \frac{\partial^2 \mathcal{D}(\widehat{w})}{\partial w_J \partial w_J}
 \end{pmatrix}
$$
在文獻中，二階微分矩陣亦被稱作**黑塞矩陣**（hessian matrix）。若 $\nabla^2 \mathcal{D}(\widehat{w})$為半正定矩陣，則意味著對於所有不為0的 $J+1$ 維向量 $v$，我們有$v^T\nabla^2 \mathcal{D}(\widehat{w}) v \geq 0$，這表示在 $\widehat{w}$ 附近，考慮任何方向之向量 $v$，其切線之速率皆展現維持水平或是遞增之狀況。

我們在此將可微分函數最小化有關之性質，以定理之方式呈現（見Nocedal & Wright, 1999之第二章）：


```{admonition} **定理：一階必要條件**
:class: note
若 $\widehat{w}$ 為 $\mathcal{D}(w)$ 之局部極小元，且 $\mathcal{D}(w)$ 在 $\widehat{w}$ 周邊之開集合（open neighborhood）為連續可微（continuously differentiable），則 $\nabla \mathcal{D}(\widehat{w}) =0$。
```


```{admonition} **定理：二階必要條件**
:class: note
若 $\widehat{w}$ 為 $\mathcal{D}(w)$ 之局部極小元，且 $\mathcal{D}(w)$ 在 $\widehat{w}$ 周邊之開集合為二次連續可微，則 $\nabla \mathcal{D}(\widehat{w}) =0$，且 $\nabla^2 \mathcal{D}(\widehat{w})$ 為半正定矩陣。
```


另外，若 $\nabla \mathcal{D}(\widehat{w}) =0$，再進一步搭配 $\nabla^2 \mathcal{D}(\widehat{w})$為正定矩陣（positive definite），即對於所有不為0的 $P+1$ 維向量 $v$ 我們有$v^T\nabla^2 \mathcal{D}(\widehat{w}) v > 0$，則我們可推論 $\widehat{w}$ 為嚴格的局部極小元（strictly local minimizer），意即，對於 $\widehat{w}$ 周邊的 $w$ 來說（$w \neq \widehat{w}$），$\widehat{w}$ 滿足 $\mathcal{D}(\widehat{w}) < \mathcal{D}(w)$。前述之充分條件，以定理的形式表達即為（見Nocedal & Wright, 1999之第二章）：

```{admonition} **定理：二階充分條件**
:class: note
若 $\mathcal{D}(w)$ 在 $\widehat{w}$ 周邊之開集合為二次連續可微，且滿足 $\nabla \mathcal{D}(\widehat{w}) =0$ 與 $\nabla^2 \mathcal{D}(\widehat{w})$ 為正定矩陣，則 $\widehat{w}$ 為 $\mathcal{D}$ 之嚴格局部極小元。
```

因此，根據前述之充分條件，當要對 $\mathcal{D}(w)$ 進行最小化，求得 $\widehat{w}$時，其步驟為：

1. 計算 $\mathcal{D}(w)$ 之梯度，即 $\nabla \mathcal{D}(w)$。
2. 獲得 $\nabla \mathcal{D}(\widehat{w})=0$ 此聯立方程組之解。
3. 檢驗 $\nabla^2 \mathcal{D}(\widehat{w})$ 此矩陣是否為正定矩陣。


### 計算範例：簡單線性迴歸

在此，我們以**簡單線性迴歸**（simple linear regression）為例來說明整個求解的過程。簡單線性迴歸僅考慮單一的共變量，即 $P=1$ 的情境，因此，對於任一觀測值 $(y_n, x_n)$，其模型表達式為：

$$
y_n = w_0 + w_1 x_n + \epsilon_n,
$$

這裡，$x_n$僅為一純量，而簡單線性迴歸的 LS 估計準則為：

$$
\mathcal{D}(w) = \frac{1}{N}\sum_{n=1}^N (y_n - w_0 - w_1 x_n)^2
$$

於前述估計準則分別對 $w_0$ 與 $w_1$ 進行偏微分，我們可得

$$
\begin{aligned}
\frac{\partial  \mathcal{D}(w)} {\partial w_0}
&= \frac{\partial} {\partial w_0}  \frac{1}{N} \sum_{n=1}^N (y_n - w_0 - w_1 x_n)^2 \\
&=   \frac{1}{N}\sum_{n=1}^N \frac{\partial} {\partial w_0} (y_n - w_0 - w_1 x_n)^2 \ \ \text{(by linear rule)} \\
&=   \frac{1}{N}\sum_{n=1}^N 2 (y_n - w_0 - w_1 x_n) \frac{\partial} {\partial w_0} (y_n - w_0 - w_1 x_n) \ \ \text{(by chain rule)} \\
&=   \frac{1}{N}\sum_{n=1}^N - 2 (y_n - w_0 - w_1 x_n).
\end{aligned}
$$

與

$$
\begin{aligned}
\frac{\partial  \mathcal{D}(w)} {\partial w_1}
&= \frac{\partial} {\partial w_1}  \frac{1}{N} \sum_{n=1}^N (y_n - w_0 - w_1 x_n)^2 \\
&=   \frac{1}{N}\sum_{n=1}^N \frac{\partial} {\partial w_1} (y_n - w_0 - w_1 x_n)^2 \ \ \text{(by linear rule)} \\
&=   \frac{1}{N}\sum_{n=1}^N 2 (y_n - w_0 - w_1 x_n) \frac{\partial} {\partial w_1} (y_n - w_0 - w_1 x_n) \ \ \text{(by chain rule)} \\
&=   \frac{1}{N}\sum_{n=1}^N - 2 x_n (y_n - w_0 - w_1 x_n).
\end{aligned}
$$

因此，在簡單線性迴歸的架構下，LS 估計準則的一階條件為

$$
\nabla \mathcal{D}(\widehat{w}) =
\begin{pmatrix}
 -\frac{2}{N}\sum_{n=1}^N (y_n - \widehat{w}_0 - \widehat{w}_1 x_n) \\
 -\frac{2}{N}\sum_{n=1}^N x_n (y_n - \widehat{w}_0 - \widehat{w}_1 x_n)
\end{pmatrix}
=
\begin{pmatrix}
0 \\
0
\end{pmatrix}.
$$

而模型參數的 LS 估計式，即為以下二元一次方程組的解：

$$
\begin{cases}
\sum_{n=1}^N y_n - N\widehat{w}_0 - \left( \sum_{n=1}^N x_n \right ) \widehat{w}_1  &=& 0 & \cdots (a) \\
 \sum_{n=1}^N x_n y_n - \left( \sum_{n=1}^N x_n \right) \widehat{w}_0  - \left( \sum_{n=1}^N x_n^2 \right)\widehat{w}_1  &=& 0 & \cdots  (b)
\end{cases}
$$

根據 $(a)$，我們可得 $m_Y  -  m_X \widehat{w}_1  = \widehat{w}_0$，這裡，$m_Y= \frac{1}{N}\sum_{n=1}^N y_n$ 與 $m_X= \frac{1}{N} \sum_{n=1}^N x_n$。將前述關係式帶入 $(b)$，即可得

$$
 \sum_{n=1}^N x_n y_n - \sum_{n=1}^N x_n  \left( m_Y  -  m_X \widehat{w}_1 \right)  -\sum_{n=1}^N x_n^2 \widehat{w}_1  = 0.
$$

透過整理，可進一步得到

$$
\sum_{n=1}^N x_n y_n -  \sum_{n=1}^N x_n  m_Y   = \left( \sum_{n=1}^N x_n^2  -  \sum_{n=1}^N x_n    m_X \right)\widehat{w}_1.
$$

因此，$\widehat{w}_1$ 的表達式為

$$
\begin{aligned}
 \widehat{w}_1 &= \frac{\sum_{n=1}^N x_n y_n - \sum_{n=1}^N x_n  m_Y}{\sum_{n=1}^N x_n^2  -  \sum_{n=1}^N x_n    m_X } \\
 &= \frac{ \frac{1}{N}\sum_{n=1}^N x_n y_n - \frac{1}{N}\sum_{n=1}^N x_n  m_Y}{\frac{1}{N} \sum_{n=1}^N x_n^2  -  \frac{1}{N} \sum_{n=1}^N x_n    m_X } \\
  &= \frac{ s_{YX}}{s_{X}^2 }. \\
 \end{aligned}
$$

這裡，$s_{YX}$ 與 $s_X^2$ 分別表示 $Y$ 與 $X$ 的共變數，以及 $X$ 的變異數。而 $\widehat{w}_0$ 即可透過 $\widehat{w}_0=m_Y  -  m_X \widehat{w}_1$ 獲得。

至於 $\mathcal{D}(\widehat{w})$ 之黑塞矩陣是否為正定矩陣，我們將在下一小節回答。

儘管在此範例中，我們僅考慮簡單線性迴歸的情況，不過，再稍微加工計算一下，我可獲得一般情況下，LS 估計準則的一階導數與二階導數：

$$
\begin{aligned}
 \frac{\partial  \mathcal{D}(w)}{ \partial w_j} & = -\frac{2}{N}\sum_{n=1}^N x_{nj} (y_n -x_n^T w),\\
  \frac{\partial^2  \mathcal{D}(w)}{ \partial w_j \partial w_{j'}} & = \frac{2}{N}\sum_{n=1}^N x_{nj} x_{nj'}.
\end{aligned}
$$

## 線性代數與線性迴歸

### 線性迴歸之矩陣表徵
線性迴歸的問題可以精簡地使用矩陣與向量的方式來表徵

$$
y = Xw + \epsilon,
$$

這裡，$y$、$X$、以及 $\epsilon$ 內部具體之元素為

$$\underbrace{\begin{pmatrix}
  y_{1} \\
  y_{2} \\
  \vdots \\
  y_{N}
 \end{pmatrix}}_{N \times 1}=
\underbrace{\begin{pmatrix}
  1 & x_{11} & \cdots & x_{1J} \\
  1 & x_{21} & \cdots & x_{2J} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  1 & x_{N1} & \cdots & x_{NJ}
 \end{pmatrix}}_{N \times (J+1)}
\underbrace{\begin{pmatrix}
  w_0 \\
  w_{1} \\
  \vdots \\
  w_{J}
 \end{pmatrix}}_{(J+1) \times 1}+
 \mathop{\begin{pmatrix}
  \epsilon_{1} \\
  \epsilon_{2} \\
  \vdots \\
  \epsilon_{N}
 \end{pmatrix}}_{N \times 1}.$$

在這邊，我們於符號使用上有稍微偷懶，原先 $y$ 與 $\epsilon$ 用於表徵純量，但在這邊，$y$ 與 $\epsilon$ 則皆用於表徵一 $N$ 維的向量，其包含了每筆觀測值之 $y_n$ 與 $\epsilon_n$。

在前述的矩陣向量表徵下，LS 估計準則可以寫為

$$\begin{aligned}
\mathcal{D}(w) & =  \frac{1}{N} \sum_{n=1}^N \epsilon_n^2 \\
  & = \frac{1}{N} \epsilon^T \epsilon\\
 & =  \frac{1}{N} (y-Xw)^T(y-Xw).
\end{aligned}$$

在下一小節，我們可以看到利用此矩陣向量表徵，可以將迴歸係數之 LS 估計，轉為一線性代數計算反矩陣之問題。

### 最適條件之矩陣表徵
透過對 $w$ 的每一個成分做偏微分，即計算 $\mathcal{D}(w)$ 之梯度，我們可以得到刻畫 LS 解的一階條件

$$\begin{aligned}
\nabla \mathcal{D}(\widehat{w}) &=
\begin{pmatrix}
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_0}  \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_1} \\
   \vdots \\
  \frac{\partial \mathcal{D}(\widehat{w})}{\partial w_J}
 \end{pmatrix} \\
&=-\frac{2}{N} X^T(y-X\widehat{w})=0.
\end{aligned}$$

此純量函數對向量（scalar function by vector）的微分的計算，可按照定義對各個 $w_j$ 進行微分後，再利用矩陣乘法之特性獲得。除此之外，亦可以參考矩陣微分（matrix calculus）中，對於[純量對向量微分之規則](https://en.wikipedia.org/wiki/Matrix_calculus#Scalar-by-vector_identities)，或是參考 Magnus 與 Neudecker（2019）之 [專書](https://www.amazon.com/-/zh_TW/gp/product/B07PS6Y6W6/ref=dbs_a_def_rwt_hsch_vapi_tkin_p1_i0)，以及 Pedersen 與 Pedersen（2012）的 [The Matrix Cookbook](http://www2.imm.dtu.dk/pubdb/pubs/3274-full.html) 。

前述一階條件 $-\frac{2}{N} X^T(y-X\widehat{w})=0$ 意味著 $\widehat{w}$ 需滿足以下的等式

$$
\underbrace{X^T X}_{(J+1) \times (J+1)} \underbrace{\widehat{w}}_{(J+1) \times 1}=\underbrace{X^T}_{(J+1) \times N} \underbrace{y}_{N \times 1}.
$$

因此，若 $X^T X$ 存在反矩陣，則迴歸係數的 LS 估計值可寫為

$$
\widehat{w} = (X^T X)^{-1} X^Ty
$$

確保 $X^T X$ 存在反矩陣的數學條件為其各直行向量（column vector）間並未存在線性相依（linear dependence）的狀況。所謂向量間有線性相依指的是某個向量，可以寫為其它向量的線性組合。一般來說，當樣本數大於變項數（$N > J$），各變項間變異數皆大於0，且皆存在其獨特的訊息時，$X^T X$ 為可逆的。$(X^T X)^{-1}$ 的計算，可採用[高斯消去法](https://en.wikipedia.org/wiki/Gaussian_elimination)（Gaussian elimination）或是[QR分解](https://en.wikipedia.org/wiki/QR_decomposition)（QR decomposition），其計算複雜度皆為 $O(P^3)$。


在前述矩陣表達式之下，LS 估計準則的黑塞矩陣為

$$
\nabla^2 \mathcal{D}(\widehat{w}) = \frac{2}{N} X^T X
$$

在此矩陣表達式之下，若要說明 $\widehat{w} = (X^T X)^{-1} X^Ty$ 為局部極小元，僅需說明 $\nabla^2 \mathcal{D}(\widehat{w})$ 為正定矩陣。給定一 $(J+1)$ 維之向量 $v$，考慮以下的二次式（quadratic form）

$$
v^T \nabla^2 \mathcal{D}(\widehat{w}) v = \frac{2}{N} v^T X^T X v
$$

根據 $(AB)^T = B^T A^T$ 此轉置的規則，$v^T X^T = (Xv)^T = u^T$，因此，我們有

$$
v^T \nabla^2 \mathcal{D}(\widehat{w}) v = \frac{2}{N} u^T u = \frac{2}{N} ||u||
$$

由於對於非0的 $u$ 來說，$||u||>0$，只要我們確保 $u = Xv$ 不為 0，則 $v^T \nabla^2 \mathcal{D}(\widehat{w}) v$ 一定得大於 0。事實上，如果 $X$ 的各直行為線性獨立，且 $v \neq 0$時，則$u = Xv \neq 0$ 必須成立，否則，會得到 $X$ 的各直行並非線性獨立的結論。故此，$\nabla^2 \mathcal{D}(\widehat{w})$ 為正定矩陣，而 $\widehat{w}$ 為嚴格的局部極小元，事實上，其亦為整體嚴格的極小元（global strict minimizer）。


## 模型適配度

線性迴歸模型使用一線性函數 $\widehat{f}(x) = x^T \widehat{w}$ 來解釋 $x$ 與 $y$ 之間的關係，然而，究竟 $\widehat{f}(x)$ 是否作出了好的刻畫，如屬一尚待評估之問題。

在統計上，評估模型用於解釋資料的適切性（appropriateness），甚至正確性（correctness），常透過**模型適配度**（goodness-of-fit）指標來進行刻畫。最簡單的適配度指標，即為估計準則在給定參數估計量下之數值，以線性迴歸搭配 LS 準則為例，該指標即為所謂的**均方誤**（mean squared error，簡稱 MSE）

$$
\mathcal{D}(\widehat{w}) = \frac{1}{N}\sum_{n=1}^N (y_n - \widehat{y}_n)^2
$$

這裡，$\widehat{y} = f(\widehat{w})= x_n^T \widehat{w}$ 表示對 $y_n$ 之預測值。當 MSE 為 0 時，則表示該模型於該資料上有完美之適配。然而，MSE 會受到變項尺度之影響而改變，且其數值意涵較難直接解讀，因此，在資料分析實務上，較少被用來評估單一模型之適配度，大多用於模型間之比較。

在統計實務上，最常用來評估線性迴歸模型（非線性的亦可）之指標應為**決定係數**（coefficient of determination），或稱作 **$R^2$**，該指標被定義為

$$
\begin{aligned}
R^2 &= \frac{\mathcal{D}(\widetilde{w}) - \mathcal{D}(\widehat{w})}{\mathcal{D}(\widetilde{w})} \\
&=  \frac{\sum_{n=1}^N (y_n - m_Y)^2 - \sum_{n=1}^N (y_n - \widehat{y}_n)^2 }{\sum_{n=1}^N (y_n - m_Y)^2}
\end{aligned}
$$

這裡，$\widetilde{w}$ 指的是在虛無模型（null model）下之估計量，即限制所有變項迴歸係數皆為 0，因此，$\widetilde{w}$僅對截距項進行估計，其數值為 $y$ 的樣本平均數 $m_Y$。決定係數之所以被稱作 $R^2$ 的原因在於，其數值等同於 $y$ 與 $\widehat{y}$ 相關之平方。

$R^2$ 以虛無模型所對應之 MSE 為出發點，計算 $\widehat{y}=f(\widehat{w})=x_n^T \widehat{w}$ 此模型可降低多少百分比之誤差。在單一樣本資料下，$R^2$ 介於 0 到 1 之間，當 $R^2=0$ 時，表示模型相較於虛無模型並沒有增加額外的預測力，而當 $R^2=1$ 時，則表示模型在樣本上達到完美的預測。


不過，在這邊要特別注意的是，$R^2$ 很靠近 1 未必代表所建構的 $\widehat{f}(w)$ 是所謂「正確的」（correct），$\widehat{f}(w)$ 可能有過度適配（over-fitting）之現象，意即，其僅在手邊的樣本資料獲得好的適配，但卻無法在來自相同母群的資料做出好的預測。另外，即使 $R^2$ 數值不大，也不代表模型一定是錯的。倘若 $y$ 與 $x$ 的真實關係為：

$$
y = f^0(x) + \epsilon
$$

這裡，$f^0(x)$ 表示了真實的迴歸函數。在此建構下，沒有任何的 $\widehat{f}(x)$ 可以表現得比 $f^0(x)$ 來得好，但即使是 $f^0(x)$，其仍因爲殘差 $\epsilon$ 的存在，並沒有辦法做出完美的預測。

總結來說，若要評估一迴歸模型的正確性，應該是要看 $\widehat{f}(x)$ 與 $f^0(x)$ 的差異性。由於 $f^0(x)$ 乃未知的函數，故前述比較在統計實務上是不可能的任務。因此，研究者採用模型選擇（model selection）之策略，透過建立多個迴歸函數，利用模型選擇指標（model selection criteria），如赤池訊息指標（Akaike information criterion，簡稱AIC）、貝氏訊息指標（Bayesian information criterion，簡稱BIC），或是交叉檢驗（cross-validation，簡稱CV），評估這些模型之相對表現，藉此說明某一個模型相較之下較為正確。不過，此取向，我們不在此做詳細的說明，有興趣之讀者可以參考 Konishi 與 Kitagawa（2007）之[專書](https://www.amazon.com/Information-Criteria-Statistical-Modeling-Statistics/dp/0387718869)。

## 延伸閱讀

1. Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2019). *Applied Linear Statistical Models：Applied Linear Regression Models* (5th ed.). New York: McGraw-Hill.
2. Rencher, A. C., & Schaalje, G. B. (2008). *Linear models in statistics*. Hoboken: Wiley & Sons.
3. Magnus, J. R., & Neudecker, H. (2019). *Matrix differential calculus with applications in statistics and econometrics*. Chichester: Wiley.
4. Petersen, K. B. & Pedersen, M. S. (2008). *The Matrix Cookbook*. Technical University of Denmark
5. Nocedal, J., & Wright, S. J. (1999). *Numerical optimization*. New York: Springer.
6. Konishi, S., & Kitagawa, G. (2008). *Information criteria and statistical modeling*. New York: Springer.
