潛在變項建模
================================



## 真實分數模型與信度

### 模型架構

在心理計量的測量議題上，最基本的模型大概就是所謂的真實分數模型（true score model）了。令 $X$ 表示個體於測量上的觀察分數（observed score），真實分數模型假設 $X$ 可以拆解為真實分數（true score）與測量誤差分數（measurement error score）之加總，即

$$
X = T + E
$$

這裡，$E$ 表示 $X$ 對應之真實分數，而 $E$ 則為 $X$ 所對應之誤差。觀測分數表示個體於特定測驗分數（test）或是試題（item）上的分數，其為研究者可以直接觀察到的部分，而真實分數則表示觀測分數背後系統性之部分，而誤差分數則為觀測分數非系統性之部分。以智力的測量為例，觀測分數可以是個體於某智力面向上的測驗分數（如詞彙測驗），而真實分數則為個體於該智力面向上經過無數嘗試後之平均分數（個體之真實詞彙分數），誤差分數則表示個體於測量當下，受內在歷程或是外在環境影響所導致之誤差（如遺忘、測試焦慮、環境太熱等）。


在真實分數模型中，觀察分數、真實分數、以及誤差分數三者皆為隨機的量，其假設：

1. 真實分數 $T$ 與誤差分數 $E$ 兩者統計獨立。
2. 真實分數 $T$ 的期望值為 $\mu_{T}$，變異數為 $\sigma^2_{T}$，即 $T_i \sim (\mu_{T}, \sigma^2_{T})$。
3. 誤差分數 $E$ 的期望值為0，變異數為 $\sigma^2_{E}$，即 $E \sim (0, \sigma^2_{E})$。

透過前述的假設，我們可以將觀測分數的變異數拆解為真實分數之變異數加上誤差分數之變異數，即

$$
\sigma_X^2= \sigma^2_{T} + \sigma^2_{E}
$$

### 信度之定義與估計
在真實分數模型之下，觀察分數的信度係數（reliability coefficient）可以定義為真實分數變異佔觀測分數總變異之百分比，即

$$
\rho_{XX} = \frac{\sigma_{T}^2}{\sigma_{E}^2}
$$

前述之定義，等價於觀測分數與真實分數相關之平方，即

$$
\rho_{XX} = \rho_{XT}^2
$$

其等價於使用真實分數預測觀測分數之決定係數（coefficient of determination）。

在實際處理測驗資料時，我們僅能夠直接觀測到觀察分數，無法獲得真實分數與誤差分數，因此，我們無法直接計算信度係數之數值。為了解決前述之困難，心理計量研究者引入了平行測驗（parallel test）之概念。考慮兩觀測分數 $X$ 與 $X'$，其皆可寫為真實分數與誤差分數之加總，即

$$
\begin{aligned}
X &= T + E \\
X' &= T' + E'
\end{aligned}
$$

同時，$T$ 與 $E$，以及 $T'$ 與 $E'$ 皆滿足真實分數模型之假設。而當前述分數進一步滿足以下條件時，我們說 $X$ 與 $X'$ 為平行測驗

1. 兩觀測分數對應之真實分數相等，即$T = T'$。
2. 兩觀測分數對應之誤差變異數相等，即 $\sigma_{E}^2$ 與 $\sigma_{E'}^2$。
3. $T$ 與 $E'$ 統計獨立，$T'$ 與 $E$ 亦為統計獨立。

在平行測驗的假設下，我們可以得到以下的結果

$$
\rho_{XX'} = \frac{\sigma_{T}^2}{\sigma_{E}^2}
$$

意即，$X$ 與 $X'$ 的相關係數，即為 $X$ 之信度（或是 $X'$ 之信度）。



## 單一因素模型

### 模型架構
在心理計量領域中，個體的能力、特質、與情感之水準，常透過潛在變項（latent variable）來表示。而在眾多的潛在變項模型中，因素分析（factor analysis）可說是最具代表性之模型，在此小節中，我們將介紹單一因素模型（single factor model），其透過引入單一的因素來解釋觀察分數間的共變。

令 $x_i$ 表示個體於第 $i$ 個測驗（或是試題）的觀測分數（$i=1,2,...,I$），單因素模型假設觀測分數可以拆解為因素分數與測量誤差的組合：

$$
x_i = \nu_i + \lambda_i \eta + \epsilon_i
$$

這裡，$\eta$ 表示潛在因素，其對 $x_i$ 之效果 $\lambda_i$ 被稱作因素負荷量（factor loading），其反映 $\eta$ 每變動一單位，預期 $x_i$ 跟著變動的量，$\nu_i$ 為截距項，其反映當 $\eta = 0$時，$x_i$ 的預期數值，而 $\epsilon_i$ 則為試題 $i$ 所對應之測量誤差。

在單一因素模型下，$\eta$ 與 $\epsilon$ 為隨機的量，我們對其分配進行假設

1. $\eta$ 與 $\epsilon$ 統計獨立，而 $\epsilon_i$ 與 $\epsilon_{j}$（$i \neq j$） 亦為統計獨立。
2. $\eta$ 之期望值為0，變異數為 $1$，即 $\eta \sim (0, 1)$。
3. $\epsilon$ 之期望值為0，變異數為 $\psi_i^2$，即 $\eta \sim (0, \psi_i^2)$


前述的單一因素模型，可以改寫為

$$
\begin{aligned}
x_i &= (\nu_i + \lambda_i \eta) + \epsilon_i \\
&= \tau_i  + \epsilon_i
\end{aligned}
$$

這裡，$\tau_i$ 表示 $x_i$ 所對應之真實分數，因此，$\tau_i$ 可對應於真實分數模型的 $T$，而 $\epsilon$ 可對應於真實分數模型之 $E$。

在單一因素模型下，我們可以進一步細分以下四種測驗的關係

1. 當 $x_1, x_2, ...,x_I$ 滿足 $\nu_i = \nu$，$\lambda_i = \lambda$，以及 $\psi_i^2 = \psi^2$ 時，我們說 $x_1, x_2, ...,x_I$ 為平行測驗（parallel tests）。
2. 當 $x_1, x_2, ...,x_I$ 滿足 $\nu_i = \nu$ 與 $\lambda_i = \lambda$，我們說 $x_1, x_2, ...,x_I$ 為 $\tau$ 相等測驗（$\tau$-equivalent tests）。
3. 當 $x_1, x_2, ...,x_I$ 滿足 $\lambda_i = \lambda$，，我們說 $x_1, x_2, ...,x_I$ 為本質 $\tau$ 相等（essentially $\tau$-equivalent tests）。
4. 當 $x_1, x_2, ...,x_I$ 滿足單一因素模型時，我們說 $x_1, x_2, ...,x_I$ 為本同源（congeneric tests）。

### 共變異數結構
在單因素模型的架構下，我們可以推導出觀察變項 $x_1, x_2, ...,x_I$ 之間的模型隱含的平均數與共變異數結構（model-implied mean and covariance structures）：

1. $\mu_i = \mathbb{E}(x_i) = \nu_i$。
2. $\sigma_i^2 = \mathbb{V}\text{ar}(x_i) = \lambda_i^2 +  \psi_i^2$。
3. $\sigma^2_{ij}= \mathbb{C}\text{ov}(x_i, x_j) = \lambda_i  \lambda_j +  \psi_i^2$。

這邊，模型隱含的平均數與變異數結構意味著，我們可以將 $x_1, x_2, ...,x_I$ 的平均數與共變數，寫為模型的參數，包括 $\nu_i$、$\lambda_i$、以及 $\psi_i^2$。

舉例來說，當 $I = 4$ 時，前述的共變異數結構可以寫為

$$
\begin{pmatrix}
  \sigma_{1}^2 &  &  &  \\
  \sigma_{21} & \sigma_{2}^2 &  &  \\
  \sigma_{31}  & \sigma_{32}  & \sigma_{3}^2 &   \\
  \sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma_{4}^2
 \end{pmatrix}
 =
\begin{pmatrix}
  \lambda_{1}^2 +\psi_1^2 &  &  &  \\
  \lambda_{2}\lambda_{1} & \lambda_{2}^2 +\psi_2^2 &  &  \\
  \lambda_{3}\lambda_{1}  & \lambda_{3}\lambda_{2}  & \lambda_{3}^2 +\psi_3^2 &   \\
  \lambda_{4}\lambda_{1} & \lambda_{4}\lambda_{2} & \lambda_{4}\lambda_{3} & \lambda_{4}^2 +\psi_4^2
 \end{pmatrix}.
$$

這裡，我們可以看到 $x_1, x_2, ...,x_I$ 的共變異數矩陣中，總共有 $4 \times 5 /2 = 10$ 個獨特的元素，而模型隱含的共變異數矩陣中只有 8 個參數，因此，單一因素模型提供了一個比較簡單的結構，來解釋共變異數矩陣。


進一步，當 $x_1, x_2, ...,x_I$ 為平行測驗時，則前述的共變異數結構可以簡化為

$$
\begin{pmatrix}
  \sigma_{1}^2 &  &  &  \\
  \sigma_{21} & \sigma_{2}^2 &  &  \\
  \sigma_{31}  & \sigma_{32}  & \sigma_{3}^2 &   \\
  \sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma_{4}^2
 \end{pmatrix}
 =
\begin{pmatrix}
  \lambda^2 +\psi^2 &  &  &  \\
  \lambda^2 & \lambda^2 +\psi^2 &  &  \\
  \lambda^2  & \lambda^2  & \lambda^2 +\psi^2 &   \\
  \lambda^2 & \lambda^2 & \lambda^2 & \lambda^2 +\psi^2
 \end{pmatrix}.
$$

此時，我們可以看到在模型隱含的共變異數結構中，只有兩個參數，因此，在平行測驗的假設下，其模型隱含的共變異數矩陣結構相當的簡單。

### 單一因素模型之最大概似法估計



## 信度估計之一般性架構

### $\omega$ 係數


### $\alpha$ 係數
