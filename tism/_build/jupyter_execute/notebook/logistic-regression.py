邏輯斯迴歸
================================

邏輯斯迴歸（logistic regression）與線性迴歸相似，都是透過一線性函數 $f(x)$ 以描述兩變項 $x$ 與 $y$ 之間的關係，但不同之處在於邏輯斯迴歸考慮的 $y$ 為類別變項，其類別數為2，此外，$f(x)$ 與 $y$ 之間的關係，需再透過一邏輯斯（logistic）函數進行轉換。邏輯斯迴歸可說是統計領域最基本之二元分類（binary classification）方法，其可視為線性迴歸於分類問題上之拓展。


在此主題中，我們將會學習以下的重點：

1. 使用伯努利分配（bernoulli distribution）刻畫類別變數之隨機行為。

2. 使用最大概似法（maximum likelihood method，簡稱ML法），對邏輯斯迴歸模型參數進行估計。

3. 利用數值優化（numerical optimization）的技術，對函數逐步地進行優化以求解。


## 邏輯斯迴歸模型
廣義來說，二元分類之問題關注的是如何使用一 $P$ 維之向量 $x$，對於二元變項 $y$ 進行預測，這裡，$y$的數值只能為0或1，即$y \in \{0,1\}$，$y=1$ 表示觀測值屬於某一類，而 $y=0$則表示觀測值屬於另外一類。在二元分類的問題下，研究者常試圖刻畫在給定 $x$ 之下，$y$ 的條件機率（conditional probability），即：

$$\mathbb{P}(y|x)=\frac{\mathbb{P}(y,x)}{\mathbb{P}(x)}$$

這裡，$\mathbb{P}(y,x)$ 表示同時考慮 $x$ 與 $y$ 的聯合機率（joint probability），而 $\mathbb{P}(x)$ 則為僅考慮 $x$ 之邊際機率（marginal probability）。在此講義中，我們將簡單的使用$\pi_1(x)$與$\pi_0(x)$來表示在給定 $x$ 之下，$y=1$ 與 $y=0$ 之條件機率，即

$$\pi_1(x) =\mathbb{P}(y=1|x)$$

與

$$\pi_0(x) =\mathbb{P}(y=0|x)$$

令$f(x)=\beta_0 + \sum_{p=1}^P \beta_p x_p$表示一線性函數，$\beta_p$與$\beta_0$分別表示迴歸係數與截距，邏輯斯迴歸試圖使用一邏輯斯函數來刻畫$f(x)$與$\pi_1(x)$的關聯性，即

$$\pi_1(x) = \frac{\exp{ \left[ f(x) \right] }}{1+\exp{ \left[ f(x) \right] }}$$

由於 $\pi_1(x)$ 與 $\pi_0(x)$ 兩者的和須為1，因此，$\pi_0(x)$ 則可寫為

$$\pi_1(x) = \frac{1}{1+\exp{ \left[ f(x) \right] }}$$

透過邏輯斯迴歸模型的結構，我們可以觀察到以下兩件事情：

1. $\pi_1(x)$ 與 $\pi_0(x)$ 兩者之數值皆介於0到1之間，符合機率的公理（axiom）。

2. 當 $f(x)$ 數值大時，$\pi_1(x)$ 的數值將很靠近1，意味著獲得 $y=1$ 的可能性很大，反之，$\pi_0(x)$ 的數值則較大，獲得 $y=0$ 的可能性較高。

在迴歸係數的解讀方面，$\beta_p$ 越大，表示 $x_p$ 對於獲得 $y=1$ 有較大的影響，反之，則對 $y=0$ 有較大的影響，然而，$\beta_p$ 影響的具體效果則不容易解讀，一般來說，需透過比較給定 $x$ 下的對數勝率（log-odds）進行解讀：

$$\begin{aligned}
\log \left[ \frac{\pi_1(x)}{\pi_0(x)} \right]
=& \log \left\{ \frac{\frac{\exp{ \left[ f(x) \right] }}{1+\exp{ \left[ f(x) \right] }}}{\frac{1}{1+\exp{ \left[ f(x) \right] }}} \right\} \\
=& \log \left\{ \exp \left[ f(x) \right] \right\} \\
=& f(x)  \\
=& \beta_0 + \sum_{p=1}^P \beta_p x_p
\end{aligned}$$

因此，$\beta_p$ 可解讀為當 $x_p$ 每變動一個單位時，預期對數勝率跟著變動的單位。


## 最大概似估計法

## 數值優化技術與求解