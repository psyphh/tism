邏輯斯迴歸
================================

邏輯斯迴歸（logistic regression）與線性迴歸相似，都是透過一線性函數 $f(x)$ 以描述兩變項 $x$ 與 $y$ 之間的關係，但不同之處在於邏輯斯迴歸考慮的 $y$ 為類別變項，其類別數為2，此外，$f(x)$ 與 $y$ 之間的關係，需再透過一邏輯斯（logistic）函數進行轉換。邏輯斯迴歸可說是統計領域最基本之二元分類（binary classification）方法，其可視為線性迴歸於分類問題上之拓展。


在此主題中，我們將會學習以下的重點：

1. 使用邏輯斯轉換（logistic transformation）來刻畫類別變數之隨機行為。

2. 使用最大概似法（maximum likelihood method，簡稱ML法），對邏輯斯迴歸模型參數進行估計。

3. 利用數值優化（numerical optimization）的技術，對函數逐步地進行優化以求解。

4. 利用適配度指標以評估模型與資料之適配度。




## 二元分類與邏輯斯迴歸

### 二元線性分類器
廣義來說，二元分類之問題關注的是如何使用一 $P$ 維之向量 $x$，對於二元變項 $y$ 進行預測，這裡，$y$的數值只能為0或1，即$y \in \{0,1\}$，$y=1$ 表示某關注的事件發生，而 $y=0$則表示該事件並未發生。在二元分類的問題下，研究者常試圖刻畫在給定 $x$ 之下，$y$ 的條件機率（conditional probability），即：

$$\mathbb{P}(y|x)=\frac{\mathbb{P}(y,x)}{\mathbb{P}(x)},$$

這裡，$\mathbb{P}(y,x)$ 表示同時考慮 $x$ 與 $y$ 的聯合機率（joint probability），而 $\mathbb{P}(x)$ 則為僅考慮 $x$ 之邊際機率（marginal probability）。在此講義中，我們將簡單的使用 $\pi(x)$ 來表示在給定 $x$ 之下，$y=1$ 之條件機率，即

$$\pi(x) =\mathbb{P}(y=1|x).$$

由於 $y=1$ 與 $y=0$ 為互補之事件（complement events），兩者對應機率之加總需為 1，因此，$1-\pi(x)$ 可用於表示給定 $x$ 之下 $y=0$ 的機率，即

$$1-\pi(x) =\mathbb{P}(y=0|x).$$

令 $f(x)=w_0 + \sum_{p=1}^P w_p x_p$表示一線性函數，$w_0$與$w_p$分別表示截距（偏誤）與迴歸係數（權重）。若我們將 $x$ 與 $w$ 重新定義為 $x = (1, x_1,...,x_P)$ 與 $w = (w_0, w_1,...,w_P)$，則此線性函數可簡單寫為 $f(x)=x^Tw$（見線性迴歸之章節）。

線性分類器（classifier）可視為對二元變項最簡單的分類模型，其基本想法為當 $f(x)$ 越大時，模型所對應之 $\mathbb{P}(y=1|x)$ 應越大，反之，當 $f(x)$ 越小時，$\mathbb{P}(y=1|x)$ 應越小。然而，由於 $f(x)$ 的數值未必介於 0 到 1 之間，因此，從建模合理性的角度來看，$f(x)$ 不宜直接作為 $\pi(x)$ 此條件機率使用。


### 邏輯斯迴歸
邏輯斯迴歸試圖使用一邏輯斯轉換（logistic transformation）來刻畫 $\pi(x)$ 與 $f(x)=x^Tw$ 之關聯性，其將 $\pi(x)=\mathbb{P}(y=1|x)$ 刻畫為

$$\pi(x) = \frac{\exp{ \left( x^Tw \right) }}{1+\exp{ \left( x^Tw \right) }},$$

另一方面，$1-\pi(x) =\mathbb{P}(y=0|x)$ 即為

$$
\begin{aligned}
1-\pi(x) = \frac{1}{1+\exp{ \left( x^Tw \right) }}.
\end{aligned}
$$

透過邏輯斯迴歸模型的結構，我們可以觀察到以下兩件事情：

1. $\pi(x)$ 與 $1 - \pi(x)$ 兩者之數值皆介於0到1之間，符合機率的公理（axiom）。

2. 當 $x^Tw$ 數值大時，$\pi(x)$ 的數值將很靠近1，意味著獲得 $y=1$ 的可能性很大，反之，$1 - \pi(x)$ 的數值則較大，獲得 $y=0$ 的可能性較高。

在迴歸係數的解讀方面，$w_p$ 越大，表示 $x_p$ 對於觀察到 $y=1$ 此事件有較正向之影響，反之，則對觀察到 $y=0$ 此事件有較正面之影響。然而，$w_p$ 對於 $y$ 之具體效果不容易解讀，一般來說，需透過比較給定 $x$ 下的對數勝率（log-odds），才能夠進行解讀：

$$\begin{aligned}
\log \left[ \frac{\pi(x)}{1-\pi(x)} \right]
=& \log \left[ \frac{\frac{\exp{ \left( x^Tw \right) }}{1+\exp{ \left( x^Tw \right) }}}{\frac{1}{1+\exp{ \left( x^Tw \right) }}} \right] \\
=& \log \left[ \exp \left( x^Tw \right) \right] \\
=& x^Tw 
\end{aligned}$$

因此，$w_p$ 可解讀為當 $x_p$ 每變動一個單位時，預期對數勝率跟著變動的單位。不過，實務上對數勝率之數值大小應如何理解仍不是一簡單的工作。


邏輯斯迴歸的目的在於透過一樣本資料，獲得對迴歸係數 $w$ 之估計 $\widehat{w}$，一方面對於 $x$ 與 $y$ 間的關係進行推論與解釋，二方面則是利用 $\widehat{y} =\widehat{\pi}(x) = \frac{\exp{ \left( x^T\widehat{w} \right) }}{1+\exp{ \left( x^T \widehat{w} \right) }}$ 此機率預測值對 $y$ 進行預測。


## 最大概似估計法

### 交叉熵與最大概似估計準則
在邏輯斯回歸下，我們使用 $\pi(x)$ 此條件機率值，對於 $y \in \{0,1\}$ 進行預測。為了度量 $y$ 與 $\pi(x)$ 之差異，邏輯斯廻歸了使用交叉熵（cross-entropy）此損失函數（loss function）

$$
L\left [ y, \pi(x) \right] = -  
y \log \pi(x)  - (1-y) \log\left[ 1- \pi(x) \right].
$$

為了瞭解交叉熵的數值如何反應 $y$ 與 $\pi(x)$之間的差異，我們可以參考以下的表格：


| 預測值\實際值 |    $y = 0$ |    $y=1$ |
|----|:--:|:--:|
| $\pi(x) \approx 0$ | $-\log\left[ 1- \pi(x) \right] \approx 0$ | $- \log \pi(x) \approx \infty$ |
| $\pi(x) \approx 1$ | $-\log\left[ 1- \pi(x) \right] \approx \infty$ | $- \log \pi(x) \approx 0$ |

透過此表格可觀察到：
+ 當 $y = 1$ 時，我們僅需考慮 $- \log \pi(x)$ 
之數值，此時，若 $\pi(x)$ 相當靠近 1 時，表示模型進行的分類是與資料匹配的，則 $- \log \pi(x)$ 的數值會相當靠近零，反之，若 $\pi(x)$ 靠近 0，則 $- \log \pi(x)$ 會趨近於一相當大的數值。

+ 當 $y=0$ 時，我們僅需考慮 $-\log\left[ 1- \pi(x) \right]$之數值，當 $\pi(x)$ 靠近 0 時，交叉熵的數值會靠近 0，反之，則會靠近一很大之數值。


給定一組隨機樣本 $\{(y_n, x_n)\}_{n=1}^N$，邏輯斯迴歸之最大概似（maximum likelihood，簡稱 ML）之適配函數（fitting function），被定義為每筆資料點所對應之交叉熵之平均：

$$
\begin{aligned}
\mathcal{D}(w) 
= \frac{1}{N} \sum_{n=1}^N L\left [ y_n, \pi(x_n) \right]. 
\end{aligned}
$$

透過尋找此 ML 估計準則之極小元 $\widehat{w}$，我們可獲得迴歸係數之 ML 估計值 。




### 最大概似估計準則之梯度

為了求得 ML 估計準則之極小元 $\widehat{w}$，我們須計算 $\mathcal{D}(w)$ 之梯度。再開始計算梯度前，我們可以先做一些觀察以簡化眼前的問題。首先，由於 ML 估計準則，是個別交叉熵的平均，則根據微分的線性性質，我們僅需考慮對個別交叉熵微分之結果，之後再將其進行平均即可。再來，對於個別的交叉熵，我們可對其表達式簡化為以下之形式：

$$
\begin{aligned}
L\left[ y, \pi(x) \right]=&-y \log \pi(x)  - (1-y) \log\left[ 1- \pi(x) \right] \\
=& -y \log \left[ \frac{\exp{ \left( x^Tw \right) }}{1+\exp{ \left( x^Tw \right) }} \right]  - (1-y) \log\left[ \frac{1}{1+\exp{ \left( x^Tw \right) }}\right] \\
=& -y x^T w +  \log\left[ 1+\exp{ \left( x^Tw \right) }\right].
\end{aligned}
$$

這邊主要是使用到了 $\log(a/b) = \log(a)-\log(b)$ 此性質。最後，對於個別交叉熵的一階偏微分結果可寫為

$$
\begin{aligned}
\frac{\partial L\left[ y, \pi(x) \right]}{\partial w_j}
=&  -y \frac{\partial}{\partial w_j}  x^T w +  \frac{\partial}{\partial w_j} \log\left[ 1+\exp{ \left( x^Tw \right) }\right] \\
=&  -y x_j + x_j\frac{\exp{ \left( x^Tw \right) }}{ 1+\exp{ \left( x^Tw \right) }}.
\end{aligned}
$$

根據前面三點觀察，我們可得對 ML 估計準則之一階導數

$$
\begin{aligned}
\frac{\partial \mathcal{D}(w)}{\partial w_j}
=& \frac{1}{N} \sum_{n=1}^N
\left \{
-y_n x_{nj} + x_{nj}\frac{\exp{ \left( x_n^Tw \right) }}{ 1+\exp{ \left( x_n^Tw \right) }}
\right \}.
\end{aligned}
$$

值得注意的是，前述的一階導數可以被寫為

$$
\begin{aligned}
\frac{\partial \mathcal{D}(w)}{\partial w_j}
= \frac{1}{N} \sum_{n=1}^N
-x_{nj} \left[ y_n - \pi(x_n) \right].
\end{aligned}
$$

前式與線性迴歸 LS 估計準則之一階導數有其在結構上的相似性。

令 $y$、$X$、以及 $\pi$ 分別表示以下之向量矩陣：

$$
y =
\underbrace{\begin{pmatrix}
y_1\\
y_2 \\
\vdots \\
y_N
\end{pmatrix}}_{N \times 1},
X =
\underbrace{\begin{pmatrix}
x_1^T\\
x_2^T \\
\vdots \\
x_N^T
\end{pmatrix}}_{N \times (P+1)},
\pi = \pi(X) =
\underbrace{\begin{pmatrix}
\pi(x_1)\\
\pi(x_2) \\
\vdots \\
\pi(x_N)
\end{pmatrix}}_{N \times 1}.
$$


前述估計準則之一階導數可以寫為矩陣之形式

$$
\begin{aligned}
\frac{\partial \mathcal{D}(w)}{\partial w} 
= - X^T y + X^T \pi.
\end{aligned}
$$

而根據一階必要條件，ML 估計式 $\widehat{w}$ 須符合以下之等式

$$
\begin{aligned}
 X^T y = X^T \widehat{\pi}.
\end{aligned}
$$

這裡，$\widehat{\pi} = \widehat{\pi}(X) = (\widehat{\pi}(x_1), \widehat{\pi}(x_2),..., \widehat{\pi}(x_N))$ 表示在 $w = \widehat{w}$ 之下，每筆觀測值對 $y$ 機率預測所組成之向量。儘管前述等式的結構相當簡單，然而，由於其並非 $\widehat{w}$ 之線性函數，故我們無法僅使用線性代數的技術來求解，取而代之的是，我們須使用數值優化的技巧。


## 數值優化技術與求解

### 線搜尋與梯度下降法
**數值優化**（numerical optimization）乃一系列用於計算目標函數 $\mathcal{D}(w)$ 極小元 $\widehat{w}$ 之技術。在此章節中，我們主要關注**線搜尋**（line search）這一類之技術。

令 $\widehat{w}^{(t)}$ 表示在第 $t$ 步驟下所得之參數估計，則線搜尋試圖使用以下之形式進行更新，以獲得第 $t+1$ 步驟下之參數估計：

$$
\widehat{w}^{(t+1)} = \widehat{w}^{(t)} + s \times \underbrace{d}_{(P+1) \times 1},
$$

這裡，$d$ 為一向量，其表示所欲更新的方向，而 $s$ 則唯一純量，表示更新步伐的大小。不同的線搜尋方法，使用不同的方向 $d$ 與步伐 $s$ 進行更新。然而，一般來說，更新方向 $d$ 多具有以下之形式

$$
d = - \underbrace{B^{-1}}_{(P+1) \times (P+1)} \nabla D(\widehat{w}^{(t)} ),
$$

這裡，$B$ 表示一對稱且可逆之矩陣。然而，為何線搜尋的方向會與目標函數的梯度取負號有關呢？考慮 $w=\widehat{w}^{(t)}$ 此位置，沿著一標準化方向 $d$ （即 $||d||=1$）的方向導數為：

$$
\langle \nabla D(\widehat{w}^{(t)} ), d \rangle = d^T  \nabla D(\widehat{w}^{(t)} ).
$$

我們希望可以找到一方向，其所對應之方向導數斜率為最陡的。根據 Cauchy–Schwarz 不等式，我們知道

$$
|d^T  \nabla D(\widehat{w}^{(t)} )|  \leq ||d|| ||\nabla D(\widehat{w}^{(t)} )||
$$

而等號成立之條件為 $d = \pm \tfrac{1}{||\nabla D(\widehat{w}^{(t)} )||} \nabla D(\widehat{w}^{(t)} )$。因此，當我將 $d$ 設為梯度取負號時，其所對應的斜率為最陡的。

根據「梯度取負號為最陡之方向」此特性，最簡單的線搜尋方法為**梯度下降法**（gradient descent method），亦稱做**最陡下降法**（steepest descent method），該算則可摘要為：


```{admonition} **算則：梯度下降法**
:class: note
1. 設定參數之起始值 $\widehat{w}^{(0)}$ 與一更新步伐 $s$。
2. 當未滿足收斂標準時，計算最陡方向 $d = -\nabla D(\widehat{w}^{(t)} )$，並更新參數估計 $\widehat{w}^{(t + 1)} = \widehat{w}^{(t + 1)} + s \times d$。
```


一般來說，數值優化常使用的收斂標準包括：
+ 直接判定梯度是否小於某很小之數值 $\epsilon$，即 $||\nabla D(\widehat{w}^{(t+1)} )|| < \epsilon$。 
+ 判定參數估計值之改變是否小於某很小之數值 $\epsilon$，即$||D(\widehat{w}^{(t+1)} ) - D(\widehat{w}^{(t)} )|| < \epsilon$。此標準較為間接，主要用於無法直接評估梯度數值的情境。
+ 判斷 $t$ 是否已大於等於可接受之最大迭代次數 $T$，即 $t \geq T$。不過要注意的是，滿足此標準不代表已找到適切的解，此標準只是用來避免過多的迭代次數。


### 牛頓法與BFGS法
若進一步考慮 $d = - B^{-1}\nabla D(\widehat{w}^{(t)} )$ 此一形式，則是否能夠找到一方向，其能夠更有效率地找到目標函數的極小元呢？根據泰勒之定理（Taylor's theorem），$\mathcal{D}(w)$ 於 $w$ 之附近，可以被以下之二次函數逼近

$$
\mathcal{D}(w + d) \approx \mathcal{D}(w) + d^T \nabla \mathcal{D}(w) + \frac{1}{2} d^T \nabla^2 \mathcal{D}(w) d.
$$

令 $f(d) = \mathcal{D}(w) + d^T \nabla \mathcal{D}(w) + \frac{1}{2} d^T \nabla^2 \mathcal{D}(w) d$，若我們試圖找到一可以最小化 $f(d)$ 之 $d$，則該 $d$ 需滿足的條件

$$
\nabla f(d) =  \nabla \mathcal{D}(w) + \nabla^2 \mathcal{D}(w) d = 0.
$$

因此，更新方向應為

$$
d = -  \nabla^2 \mathcal{D}(w)^{-1} \nabla \mathcal{D}(w).
$$

意即，將 $B$ 設為估計準則之黑塞矩陣 $\nabla^2 \mathcal{D}(w)$。

進一步考慮到黑塞矩陣之訊息，我們可以得到一較為進階的線搜尋法，其稱作**牛頓法**（Newton's method），該算則可摘要為：

```{admonition} **算則：牛頓法**
:class: note
1. 設定參數之起始值 $\widehat{w}^{(0)}$ 與一更新步伐 $s$。
2. 當未滿足收斂標準時，計算 $d = - \nabla^2 \mathcal{D}(\widehat{w}^{(t)}) \nabla D(\widehat{w}^{(t)} )$，並更新參數估計 $\widehat{w}^{(t + 1)} = \widehat{w}^{(t)} + s \times d$。
```



牛頓法比梯度下降法有較佳的收斂性，意即，可使用較少之迭代次數就找到極小元。然而，其缺點在於黑塞矩陣的計算，並非對所有的估計準則來說都是容易的，此外，當參數個數較多時，計算黑塞矩陣的成本提高，甚至在大型模型下有可能會引發記憶體不足的問題。


為了解決前述牛頓法之缺點，在實務上，大多會採用所謂的**準牛頓法**（quasi-Newton's method），其使用某種方式獲得對於黑色矩陣 $\nabla^2 \mathcal{D}(\widehat{w}^{(t)})$ 之逼近。其中，最為有名的是**BFGS法**，此名稱乃根據其提出者Broyden、Fletcher、Goldfarb、Shanno命名。


令 $u^{(t)} = \widehat{w}^{(t)} - \widehat{w}^{(t-1)}$ 與 $v^{(t)}=\nabla D(\widehat{w}^{(t)} ) - \nabla D(\widehat{w}^{(t-1)} )$，根據泰勒之定理，我們有

$$
v^{(t)} \approx \nabla^2 \mathcal{D}(\widehat{w}^{(t)}) u^{(t)}.
$$

前述關係式亦描述了以下之關係

$$
\nabla^2 \mathcal{D}(\widehat{w}^{(t)}) ^{-1} v^{(t)} \approx  u^{(t)}.
$$

BFGS 法利用此式之特性，試圖獲得對於黑塞矩陣反矩陣之逼近。令 $H^{(t-1)}$ 表示在 $t-1$ 步驟時對於黑塞矩陣反矩陣之逼近，BFGS 法考慮的問題為：

$$
\begin{aligned}
&\text{minimize}_{H} ||H - H^{(t-1)}|| \\
& \text{subject to } H = H^T, Bv^{(t)} = u^{(t)} .
\end{aligned}
$$

透過解此優化問題，可得到 $H^{(t)}$ 的更新公式為：

$$
H^{(t)} = (I - \rho^{(t)} u^{(t)} {v^{(t)}}^T )H^{(t-1)} (I - \rho^{(t)} v^{(t)} {u^{(t)}}^T ) + \rho^{(t)} u^{(t)} {u^{(t)}}^T ,
$$

這裡，$\rho^{(t)} = \frac{1}{ {v^{(t)}}^T u^{(t)}}$。在此，我們可觀察到在BFGS法之下，我們直接利用了一簡單的公式，獲得了對於黑塞矩陣的反矩陣，並且該簡單的公式僅利用到了參數估計與梯度在迭代過程中的差值。BFGS 可以摘要為

```{admonition} **算則：BFGS法**
:class: note
1. 設定參數之起始值 $\widehat{w}^{(0)}$ 、一更新步伐 $s$、以及$H^{(0)}$（實務上多採用$H^{(0)}=I$）。
2. 當未滿足收斂標準時，計算 $d = - H^{(t)} \nabla D(\widehat{w}^{(t)} )$，並更新參數估計 $\widehat{w}^{(t + 1)} = \widehat{w}^{(t)} + s \times d$。
```


### 更新步伐之選取

除了更新方向 $d$，更新步伐 $s$ 的挑選，亦在線搜尋方法中扮演重要的角色。當 $s$ 過大時，可能會無法找到局部極小元，而當 $s$ 過小時，則可能會導致收斂過慢的問題。因此，要如何找到一適切的更新步伐大小，不是一件容易的工作。


在實務上，$s$ 常在優化過程中，以動態的方式進行調整。理想上，在給定更新方向 $d$ 之後，我們可透過解決以下的問題，獲得一最佳的更新步伐：

$$
\text{minimize}_{s} \mathcal{D}(\widehat{w} + s \times d).
$$

然而，這種找最適更新步伐的方法，等同於要去解一個優化問題，除非目標函數的結構很單純，否則，前述的作法是相當不經濟的。因此，在實務上，更新步伐的調整多採用以下兩種策略：

+ 根據某定好之規則，令$s$ 之數值隨迭代次數 $t$ 遞減。舉例來說，我們可增加一滿足 $0<\gamma<1$ 之參數 $\gamma$，接著，每當 $t$ 滿足某些條件時（如 $t$ 為10的倍數時），重新定義新的更新步伐為 $s \leftarrow \gamma s$。

+ 利用**回朔線搜尋**（backtracking line search）尋找 $s$。此方法之執行如下：

```{admonition} **算則：回朔線搜尋**
:class: note
1. 給定 $s_0>0$、$\gamma \in (0, 1)$、以及 $c \in (0, 1)$。
2. 在每次迭代 $t$ 開始時，將 $s$ 設為 $s_0$，接著，當 $\mathcal{D}(\widehat{w}^{(t)} + s \times d) < \mathcal{D}(\widehat{w}^{(t)}) + s \times c \times d^T \nabla \mathcal{D}(\widehat{w}^{(t)})$ 此條件未被滿足時，根據 $s \leftarrow \gamma \times s$ 此公式持續調整 $s$ 之數值。
```


## 模型適配度

邏輯斯迴歸利用 $\widehat{\pi}(x) = \frac{\exp{ \left( x^T\widehat{w} \right) }}{1+\exp{ \left( x^T \widehat{w} \right) }}$ 對 $y$ 進行預測，然而，$\widehat{\pi}(x)$ 究竟是否能夠良好地刻畫 $x$ 與 $y$ 間的關係，則需透過適配度指標來評估。

首先，ML 適配函數 $\mathcal{D}(\widehat{w})$ 即可用於評估模型之表現，但其數值大小不易解讀，故實務上難以使用。

有鑒於 $R^2$ 於線性迴歸問題上的受歡迎，在邏輯斯迴歸的脈絡下，亦有不少所謂的虛擬 $R^2$ （pseudo $R^2$）被提出。如 McFadden（1973）將虛擬 $R^2$ 定義為

$$
\begin{aligned}
R_{McFadden}^2 &= \frac{\mathcal{D}(\widetilde{w}) - \mathcal{D}(\widehat{w})}{\mathcal{D}(\widetilde{w})},
\end{aligned}
$$

這裡，$\widetilde{w}$ 指的是在虛無模型下之估計量，即限制所有變項迴歸係數皆為 0，僅對截距項進行估計。另外，Efron（1978）則是將虛擬 $R^2$ 定義為

$$
\begin{aligned}
R_{Efron}^2 &=  \frac{\sum_{n=1}^N(y_n - m_Y)^2 - \sum_{n=1}^N(y_n - \widehat{\pi}(x_n))^2}{\sum_{n=1}^N(y_n - m_Y)^2}.
\end{aligned}
$$

從 $R_{McFadden}^2$ 與 $R_{Efron}^2$ 兩者之公式來看，前者乃透過 ML 估計準則之數值來定義 $R^2$，後者則是根據 LS 估計準則搭配機率預測值 $\widehat{\pi}(x)$ 來定義。但不管採用何種定義，兩者皆透過與一虛無模型比較建立，數值越靠近 0 表示模型與虛無模型表現類似，而數值越靠近 1 則表示模型與資料之適配程度越好。

除了前述的指標外，若我們將 $\widehat{\pi}(x)$ 此機率預測值改為類別預測，則可使用分類正確率來評估模型適配度。常使用的類別預測為

$$
\widehat{y}^{c}=
\begin{cases}
1 &\text{if } \widehat{\pi}(x) \geq 0.5, \\
0 &\text{if } \widehat{\pi}(x) < 0.5,
\end{cases}
$$
意即，當 $\widehat{\pi}(x) \geq 0.5$ 時，我們傾向認為其所對應之 $y$ 應為 1，反之，則認為其所對應之 $y$ 應為0。因此，我們可以將分類正確率定義為樣本資料中，$y_n$ 與 $\widehat{y}_n^c$ 相等之比例，即

$$
Accuracy = \frac{1}{N} \sum_{n=1}^N 1\{y_n = \widehat{y}_n^c \},
$$

這裡，$1\{\cdot \}$ 表示一指示函數（indicator function），其用於「指示」大括號內之事件是否為真。當 $E$ 事件為真時，$1\{E \} = 1$，反之，$1\{E \} = 0$。


## 實作範例與練習

### 虛無模型之截距估計
我們在此展示如何使用梯度下降法，對虛無模型下之截距項進行估計。由於此模型中並未加入任何的共變量，因此，其 ML 適配函數與一階導數為

$$
\begin{aligned}
\mathcal{D}(w_0) & = \frac{1}{N} \sum_{n=1}^N \left \{ -y_n w_0 +  \log\left[ 1+\exp{ \left( w_0 \right) }\right] \right\} \\
& =  - w_0 m_Y +  \log\left[ 1+\exp{ \left( w_0 \right) }\right],
\end{aligned}
$$

與

$$
\begin{aligned}
\frac{\partial \mathcal{D}(w_0)}{\partial w_0} &=
\frac{1}{N} \sum_{n=1}^N
\left \{
-y_n + \frac{\exp{ \left( w_0 \right) }}{ 1+\exp{ \left( w_0 \right) }}
\right \}\\
& = -m_Y+ \frac{\exp{ \left( w_0 \right) }}{ 1+\exp{ \left( w_0 \right) }},
\end{aligned}
$$
這裡，$m_Y$ 表示 $Y$ 此變項之樣本平均數，在二元變項下，$m_Y$ 等同於樣本資料中等於 1 的比例。倘若手邊的樣本資料中，$y_n=1$ 的比例為80%，則 $m_Y=.8$，我們可以使用 `python` 撰寫函數，來計算在不同的 $w_0$之下，所對應到的 ML 適配函數與其一階導數之數值：

import torch
def cal_loss(bias):
    if type(bias) is not torch.Tensor:
        bias = torch.tensor(bias, dtype = torch.float64)
    loss = -0.8 * bias + torch.log(1 + bias.exp())
    return loss

def cal_grad(bias):
    if type(bias) is not torch.Tensor:
        bias = torch.tensor(bias, dtype = torch.float64)
    grad = -0.8 + bias.exp() / (1 + bias.exp())
    return grad


接下來，我們撰寫一執行梯度下降法之函數，以尋找一可最小化 ML 適配函數之 $\widehat{w}_0$。

def gradient_descent(bias, step_size, iter_max):
    for t in range(1, iter_max + 1):
        direction = - cal_grad(bias)
        bias += step_size * direction
        loss = cal_loss(bias)
        print("iter {:2.0f}, loss = {:2.3f}, grad = {:2.3f}, bias = {:2.3f}".format(
            t, loss.item(), -direction.item(), bias.item()))

我們將參數起始值 $\widehat{w}_0^{(0)}$設為0，步伐大小 $s$ 設為 2，觀察迭代 10 次後，$\widehat{w}_0^{(10)}$ 之數值為何

bias = 0
step_size = 2.
iter_max = 10
gradient_descent(bias, step_size, iter_max)

我們可以看到梯度下降法，逐步地更新截距項，使得 ML 適配函數數值下降，且一階導數越來越靠近 0。事實上，此優化問題的極小元存在閉合形式解（closed-form solution），根據一階最適條件，$\frac{\exp{ \left( \widehat{w}_0 \right) }}{ 1+\exp{ \left( \widehat{w}_0 \right) }} = m_Y$，因此，$\widehat{w}_0 = \log \left[\frac{m_Y}{1-m_Y} \right] = \log \left[\frac{.8}{1-.8} \right] \approx  1.386$，我們可以看到梯度下降找到的解，與此閉合形式解幾無差異。

接下來，我們開始觀察在不同更新步伐大小下之行為表現：

print("\nbais = {:2.3f}, step size = {:2.2f}".format(0, 20.))
gradient_descent(0, 20., 10)
print("\nbais = {:2.3f}, step size = {:2.2f}".format(0, .1))
gradient_descent(0, .2, 10)
print("\nbais = {:2.3f}, step size = {:2.2f}".format(1.5, 2.))
gradient_descent(1.5, 2., 10)

### 練習
請利用以下之程式碼，產生邏輯斯迴歸之資料，並撰寫一函數執行梯度下降法，計算在此資料下之模型參數估計，並觀察其在不同起始值與更新步伐下之行為表現。

# set seed
torch.manual_seed(246437)

# write a function to generate data
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

## 延伸閱讀

1. Agresti, A. (2012). *Categorical data analysis (3rd edition)*. New York: Wiley.
2. Efron, B. (1978). Regression and ANOVA with zero-one data: Measures of residual variation. *Journal of the American Statistical Association, 73*(361), 113–121.
3. McFadden, D. (1973). Conditional logit analysis of qualitative choice behavior. In *Frontiers in Econometrics* (Edited by P. Zarembka), 105-42. Academic Press, New York.
4. Nocedal, J., & Wright, S. J. (1999). *Numerical optimization*. New York: Springer.



