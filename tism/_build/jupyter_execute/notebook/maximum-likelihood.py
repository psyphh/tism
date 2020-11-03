最大概似法
================================

在前一章節，我們提到所有隨機的現象，儘管我們無法事先得知其實現值為何，但我們可以透過其分配函數來了解行為表現。而許多的統計建模問題，我們試圖對變項的行為表現提出一參數化的機率模型，接著，收集樣本資料以對模型之參數進行估計。這裡，參數（parameter）指的是決定機率分佈形狀之量，像是二項式分配的成功機率、常態分配的平均數與變異數，皆為模型之參數。在此章節中，若一隨機變數 $X$ 的PMF/PDF仰賴某參數 $\theta$ 的話，我們將其 PMF/PDF 寫為 $f_X(x|\theta)$。

估計模型參數為統計的核心議題，在諸多參數估計方法中，最大概似（maximum likelihood，簡稱ML）法可說是最為重要的一種方法，在本章節，我們將學習如何進行最大概似法進行參數估計。



## 最大概似估計

### 估計的基本概念
在初等統計學當中，同學們應該已具有基本的估計概念，舉例來說，樣本平均數 $m$ 乃作為母群體平均數 $\mu$ 之估計，樣本變異數 $s^2$ 則是作為母群體變異數 $\sigma^2$ 之估計。我們可以看見，無論是 $m$ 或是 $s^2$，其都是樣本資料的函數，因此，很寬鬆的來說，任何樣本資料的函數都被稱作為估計式（estimator，可以理解為公式），而在特定資料數值下所算出來的結果，則被稱作估計量或估計值（estimate，可以理解為數值）。

在此章節中，我們使用 $\theta$ 來表示參數，而 $\widehat{\theta}$ 則用於表示對 $\theta$ 的估計式或估計值（適情況而定。）

一個估計式是不是夠好，是需要進一步說明的。舉例來說，$m$ 或是 $s^2$ 分別是 $\mu$ 與 $\sigma^2$ 的不偏估計式（unbiased estimator），這意味著 $\mathbb{E}(m) = \mu$ 與 $\mathbb{E}(s^2) = \sigma^2$。因此，當 $\widehat{\theta}$ 滿足以下條件時，我們說 $\widehat{\theta}$ 為 $\theta$ 的不偏估計式：

$$
\mathbb{E}(\widehat{\theta}) = \theta
$$

然而，不偏性（unbiasedness）只是評估估計式好壞的其中一標準而已，事實上，隨便挑選資料中的一筆出來，其也能作為平均數的不偏估計（$\mathbb{E}(X_n) = \mu$），因此，不偏性在實務上並非是個重要的標準，相較之下，統計學家更常使用均方誤（mean squared error，簡稱MSE）來評估一估計式的好壞：

$$
MSE(\widehat{\theta}) = \mathbb{E}(|\widehat{\theta} - \theta|^2)
$$

一般來說，MSE越小則意味該估計式的品質越好，不過在解讀 MSE 的意涵時要特別小心，事實上，MSE 包含了變異與偏誤兩個成分

$$
\begin{aligned}
MSE(\widehat{\theta}) &= \mathbb{E}(|\widehat{\theta} - \theta|^2) \\
&=\mathbb{E}\left\{ | [\widehat{\theta}- \mathbb{E}(\widehat{\theta})] + [\mathbb{E}(\widehat{\theta})  - \theta]|^2 \right\} \\
&=\mathbb{E}\left\{ [\widehat{\theta}- \mathbb{E}(\widehat{\theta})]^2 + 2 [\widehat{\theta}- \mathbb{E}(\widehat{\theta})][\mathbb{E}(\widehat{\theta})  - \theta] + [\mathbb{E}(\widehat{\theta})  - \theta]^2 \right\}\\
&=\mathbb{E}\left\{ [\widehat{\theta}- \mathbb{E}(\widehat{\theta})]^2 \right\} +
\mathbb{E}\left\{  2 [\widehat{\theta}- \mathbb{E}(\widehat{\theta})][\mathbb{E}(\widehat{\theta})  - \theta]\right\}+
\mathbb{E}\left\{ [\mathbb{E}(\widehat{\theta})  - \theta]^2 \right\} \\
& = \mathbb{V}\text{ar} (\widehat{\theta}) + [\mathbb{E}(\widehat{\theta})  - \theta]^2
\end{aligned}
$$

因此，在評估一估計式的表現時，建議可以將變異與偏誤兩部分分開來評估。

### 概似函數與最大概似估計式
令 $X_1, X_2,...,X_N$ 表示一來自 $f_X(x|\theta)$ 此分配之隨機樣本（random sample），這意味著：

1. $X_n$ 與 $X_{n'}$ 彼此相互獨立（$n \neq n'$）；
2. 對於每筆觀測值 $X_n$ 來說，其分配皆為 $f_X(x|\theta)$，此分配的特性由模型參數 $\theta$ 所控制。

若此樣本的實現值為為 $X_1=x_1,X_2=x_2,...,X_N=x_N$，則奠基於$f_X(x|\theta)$ 此分配家族與前述資料之概似函數（likelihood function）為

$$
\mathcal{L}(\theta; x_1,x_2,...,x_N)= \prod_{n=1}^N f_X(x_n|\theta)
$$

此概似函數反映了在當下的參數數值 $\theta$ 之下，我們觀察到 $X_1=x_1,X_2=x_2,...,X_N=x_N$ 這筆資料的可能性，因此，一種合理的估計策略為，試圖找到一參數估計式 $\widehat{\theta}$ 使得觀察到此樣本資料之可能性達到最大，意即

$$
\mathcal{L}(\widehat{\theta}; x_1,x_2,...,x_N) = \max_{\theta} \mathcal{L}(\theta; x_1,x_2,...,x_N)
$$

而此 $\widehat{\theta}$ 則被稱作最大概似估計式（maximum likelihood estimator，簡稱MLE）。


由於 $f_X(x|\theta)$ 數值常小於 1，因此，$\prod_{n=1}^N f_X(x_n|\theta)$ 會令概似函數的數值變得很靠近 0，此外，連續相乘在數學上處理難度較高，因此，在實務上研究者主要處理對數概似函數，即

$$
\begin{aligned}
\ell(\theta) &=\log \mathcal{L}(\theta; x_1,x_2,...,x_N) \\
&= \sum_{n=1}^N \log f_X(x_n|\theta)
\end{aligned}
$$


## 最大概似估計範例

### 伯努利分配成功機率之估計

令 $X_1, X_2,...,X_N$ 表示一來自 $f_X(x|\pi) = \pi^x (1-\pi)^{(1-x)}$ 之隨機樣本，則其對應之對數概似函數為

$$
\begin{aligned}
\ell(\pi) & = \sum_{n=1}^N \log f_X(x_n|\pi) \\
& = \sum_{n=1}^N \log\left[ \pi^{x_n} (1-\pi)^{(1-x_n)} \right] \\
& = \sum_{n=1}^N \left[ x_n \log\pi + (1-x_n)\log(1-\pi) \right] \\
\end{aligned}
$$

若要求得該概似函數的極大元（maximizer），等同於求得該函數取負號的極小元（minimizer），我們可計算其一階導數

$$
\begin{aligned}
-\frac{d}{d \pi} \ell(\pi)
&=  \sum_{n=1}^N \frac{d}{d \pi} \left[ -x_n \log\pi - (1-x_n)\log(1-\pi) \right] \\
&=  \sum_{n=1}^N  \left[ -x_n \frac{d}{d \pi} \log\pi - (1-x_n) \frac{d}{d \pi} \log(1-\pi) \right] \\
&=  \sum_{n=1}^N  \left[ -x_n \frac{1}{\pi} - (1-x_n)\frac{1}{1-\pi}  \frac{d}{d \pi} \log(1-\pi) \right] \\
&=  \sum_{n=1}^N  \left[  \frac{-x_n}{\pi} + \frac{1-x_n}{1-\pi}   \right] \\
&=  \frac{-1}{\pi}\sum_{n=1}^N x_n  + \frac{1}{1-\pi} \sum_{n=1}^N (1-x_n)    \\
\end{aligned}
$$

因此，$\widehat{\pi}$ 須滿足

$$
\begin{aligned}
 (1-\widehat{\pi})\sum_{n=1}^N x_n  = \widehat{\pi} (N - \sum_{n=1}^N x_n) \iff
 \sum_{n=1}^N x_n  =  \widehat{\pi} N  \\
\end{aligned}
$$

我們可得 $\widehat{\pi} = \frac{1}{N} \sum_{n=1}^N x_n$，意即，$\widehat{\pi}$ 為樣本比率（sample proportion）。

若我們進一步計算 $-\ell(\widehat{\pi})$ 的二階微分，則可得

$$
\begin{aligned}
-\frac{d^2}{d \pi^2} \ell(\widehat{\pi})
&=  \frac{d}{d \pi}\frac{-1}{\widehat{\pi}}\sum_{n=1}^N x_n  + \frac{d}{d \pi} \frac{1}{1-\widehat{\pi}} \sum_{n=1}^N (1-x_n)    \\
&=  \frac{N}{\widehat{\pi}}  + \frac{N}{1-\widehat{\pi}}    \\
&=  \frac{N(1-\widehat{\pi}) + N \widehat{\pi}}{\widehat{\pi}(1-\widehat{\pi})}  \\
&=  \frac{N}{\widehat{\pi}(1-\widehat{\pi})} \geq 0 \\
\end{aligned}
$$

由於此二階導數在 $\pi = \widehat{\pi}$ 大於 0，因此，$\widehat{\pi}$ 確實最小化了$-\ell(\widehat{\pi})$ 此取負號的對數概似函數。

### 常態分配之平均數與變異數估計

令 $X_1, X_2,...,X_N$ 表示一來自 $f_X(x|\mu, \sigma^2)  = \frac{1}{\sqrt{2 \pi} \sigma_X} e^{-(x - \mu)^2/2 \sigma^2}$ 之隨機樣本，則其對應之對數概似函數為


$$
\begin{aligned}
\ell(\mu, \sigma^2) & = \sum_{n=1}^N \log f_X(x_n|\mu, \sigma^2) \\
& = \sum_{n=1}^N \log\left[ \frac{1}{\sqrt{2 \pi} \sigma} e^{-(x_n - \mu)^2/2 \sigma^2} \right] \\
& = \sum_{n=1}^N \left[ -\frac{1}{2}\log{2 \pi} -\frac{1}{2}\log{\sigma^2} -\frac{(x_n - \mu)^2}{2 \sigma^2} \right] \\
& =  -\frac{N}{2}\log{2 \pi} - \frac{N}{2}\log{\sigma^2} - \frac{1}{2 \sigma^2} \sum_{n=1}^N (x_n - \mu)^2  \\
\end{aligned}
$$

為了計算其MLE，我們分別對 $\mu$ 與 $\sigma^2$ 進行偏微分

$$
\begin{aligned}
-\frac{\partial}{\partial \mu }\ell(\mu, \sigma^2)
& = \frac{1}{2 \sigma^2} \sum_{n=1}^N  \frac{\partial}{\partial \mu } (x_n - \mu)^2  \\
& =  \frac{1}{2 \sigma^2} \sum_{n=1}^N 2(x_n - \mu)\frac{\partial}{\partial \mu } (x_n - \mu) \\
& =  -\frac{1}{ \sigma^2} \sum_{n=1}^N (x_n - \mu) \\
\end{aligned}
$$

$$
\begin{aligned}
-\frac{\partial}{\partial \sigma^2 }\ell(\mu, \sigma^2)
& =  \frac{\partial}{\partial \sigma^2 } \frac{N}{2}\log{\sigma^2} + \frac{\partial}{\partial \sigma^2 } \frac{1}{2 \sigma^2} \sum_{n=1}^N (x_n - \mu)^2  \\
& =  \frac{N}{2\sigma^2} + \frac{-1}{2 \sigma^4} \sum_{n=1}^N (x_n - \mu)^2  \\
\end{aligned}
$$
因此，我們有

$$
\begin{aligned}
 -\frac{1}{\sigma^2} \sum_{n=1}^N (x_n - \widehat{\mu}) = 0
 \iff
\widehat{\mu} = \frac{1}{N}\sum_{n=1}^Nx_n
\end{aligned}
$$


$$
\begin{aligned}
 \frac{N}{2\widehat{\sigma}^2} = \frac{1}{2 \widehat{\sigma}^4} \sum_{n=1}^N (x_n - \widehat{\mu})^2
 \iff
 \widehat{\sigma}^2 = \frac{1}{N} \sum_{n=1}^N (x_n - \widehat{\mu})^2  \\
\end{aligned}
$$

而此負對數概似函數的二階導數為

$$
\begin{aligned}
-\frac{\partial^2}{\partial \mu^2 }\ell(\widehat{\mu}, \sigma^2)
& =  -\frac{1}{\widehat{\sigma}^2}\frac{\partial}{\partial \widehat{\mu} } \sum_{n=1}^N (x_n - \widehat{\mu}) \\
& =  \frac{N}{\widehat{\sigma}^2}
\end{aligned}
$$

$$
\begin{aligned}
-\frac{\partial^2}{\partial \sigma^4 }\ell(\widehat{\mu}, \widehat{\sigma}^2)
& =  \frac{\partial}{\partial \sigma^2 }\frac{N}{2\widehat{\sigma}^2} + \frac{\partial}{\partial \sigma^2 }\frac{-1}{2 \widehat{\sigma}^4} \sum_{n=1}^N (x_n - \widehat{\mu})^2  \\
& = \frac{-N}{2\widehat{\sigma}^4} + \frac{1}{\widehat{\sigma}^6} \sum_{n=1}^N (x_n - \widehat{\mu})^2  \\
& =  \frac{N}{2\widehat{\sigma}^4}  \\
\end{aligned}
$$

$$
\begin{aligned}
-\frac{\partial^2}{\partial \mu \partial \sigma^2}\ell(\widehat{\mu}, \widehat{\sigma}^2)
& =  \frac{\partial}{\partial \widehat{\sigma}^2}-\frac{1}{ \sigma^2} \sum_{n=1}^N (x_n - \widehat{\mu}) \\
& = \frac{1}{ \widehat{\sigma}} \sum_{n=1}^N (x_n - \widehat{\mu}) \\
& = 0
\end{aligned}
$$

因此，負對數概似函數的黑塞矩陣為

$$
\begin{aligned}
-\nabla^2 \ell(\widehat{\mu}, \widehat{\sigma}^2)
&= -
\begin{pmatrix}
\frac{\partial^2}{\partial \mu^4}\ell(\widehat{\mu}, \widehat{\sigma}^2)  & \frac{\partial^2}{\partial \mu \partial \sigma^2}\ell(\widehat{\mu}, \widehat{\sigma}^2) \\
\frac{\partial^2}{\partial \sigma^2 \partial \mu }\ell(\widehat{\mu}, \widehat{\sigma}^2) &  \frac{\partial^2}{\partial \sigma^4}\ell(\widehat{\mu}, \widehat{\sigma}^2)
\end{pmatrix}\\
&=
\begin{pmatrix}
 \frac{N}{\widehat{\sigma}^2} & 0 \\
 0 &  \frac{N}{2\widehat{\sigma}^4}
\end{pmatrix}
\end{aligned}
$$


### 線性迴歸之最大概似估計
在線性廻歸模型下，我們使用 $x$ 對於 $y$ 進行預測。假設在給定 $x$ 之下，$y$ 的條件分佈為

$$
y|x \sim \text{Normal}(w_0 + \sum_{p=1}^P w_p x_p, \sigma_{\epsilon}^2)
$$
則給定一樣本資料 $\{(y_n,x_n)\}_{n=1}^N$，我們可以使用前述之條件分佈來建立參數之概似函數

$$
\begin{aligned}
\ell(w, \sigma_{\epsilon}^2)
&= \sum_{n=1}^N \log f_{Y|X}(y_n| x_n, w, \sigma_{\epsilon}^2)\\
&= \sum_{n=1}^N \log\left[ \frac{1}{\sqrt{2 \pi} \sigma_{\epsilon}} e^{-\frac{1}{2 \sigma_{\epsilon}^2}\left (y_n - w_0 - \sum_{p=1}^P w_p x_{np} \right)^2} \right] \\
&= \sum_{n=1}^N \left[-\frac{1}{2}\log{2 \pi} -\frac{1}{2}\log{\sigma_{\epsilon}} - {\frac{1}{2 \sigma_{\epsilon}^2}\left (y_n - w_0 - \sum_{p=1}^P w_p x_{np} \right)^2} \right] \\
&=  -\frac{N}{2}\log{2 \pi} -\frac{N}{2}\log{\sigma_{\epsilon}} - \frac{1}{2 \sigma_{\epsilon}^2} \sum_{n=1}^N \left (y_n - w_0 - \sum_{p=1}^P w_p x_{np} \right)^2 \\
\end{aligned}
$$

事實上，先前所使用之最小平方法適配函數，其可以寫為對負號對數概述函數標準化後之結果

$$
\mathcal{D}(w) \propto - \frac{1}{N} \ell(w, \sigma_{\epsilon}^2)
$$



## 最大概似估計式之理論性質

在滿足以下的條件時，MLE具有一致性、大樣本常態性、與不變性等性質

1. $X_1,X_2,...,X_n$ 為來自 $f_X(x|\theta)$ 之獨立且相同分配之樣本。
2. 參數為可辨識（identifiable），意即，當 $\theta \neq \theta'$ 時，$f_X(x|\theta) \neq f_X(x|\theta')$。
3. $f_X(x|\theta)$ 此函數具有相同的支撐（support，意即，$x$ 可數值範圍與 $\theta$ 無關）。
4. 參數真實數值 $\theta^*$ 位於參數空間之內點（interior point）。
5. $f_X(x|\theta)$ 為三次連續可微分，$\int f_X(x|\theta)dx$ 為三次可微。
6. 對於任一的 $\theta^*$，存在一$\delta>0$與函數 $M(x)$，使得對於所有的 $x$ 與 $\theta \in [\theta^* - \delta, \theta^* + \delta]$，滿足

$$
|\frac{\partial^3}{\partial \theta^3} \log f_X(x|\theta)| \leq M(x)
$$
且 $\mathbb{E}(M(X)) < \infty$。

### 一致性

MLE具有一致性（consistency），意即，當樣本數夠大時，$\widehat{\theta}$ 會跟參數的真實數值 $\theta^*$ 很靠近。若用數學概念來表示的話，即給定任何的 $\epsilon >0$，我們有

$$
\lim_{N \to \infty } \mathbb{P}(|\widehat{\theta} - \theta^*| > \epsilon) = 0
$$

這邊須要特別注意的是，$\widehat{\theta}$ 事實上是樣本數的函數。前述的數學條件，可以簡單地寫為 $\widehat{\theta} \to_{P} \theta^*$，表示 $\widehat{\theta}$ 於機率收斂於 $\theta^*$（$\widehat{\theta}$ converges to $\theta^*$ in probability）。


### 大樣本常態性

MLE具有大樣本常態性（consistency），意即，當樣本數夠大時，$\widehat{\theta}$ 會呈現多元常態分配，其平均數與共變異數矩陣為 $\theta^*$ 與 $\mathcal{I}(\theta^*)^{-1}$，意即

$$
\widehat{\theta} \sim \text{Normal}\left[\theta^*, \frac{1}{N} \mathcal{I}(\theta^*)^{-1} \right]
$$

這裡，$\mathcal{I}(\theta^*)= \mathbb{E}\left[-\nabla^2 \frac{1}{N} \ell(\widehat{\theta}) \right]$，其被稱作費雪期望訊息矩陣（Fisher's expected information matrix）。

此外，由於 $\frac{1}{N} \mathcal{I}(\theta^*)^{-1}$ 此共變異數矩陣為理論上變異性最小的，其達到了所謂的Cramer-Rao下界，因此，MLE具有所謂的有效性（efficiency），意即，其大樣本時的MSE，是所有估計式中最小的。

### 不變性
如果 $\widehat{\theta}$ 為 $\theta^*$ 之MLE，令 $g$ 表示一函數，其將 $\theta$ 轉換為 $\vartheta$，即$\vartheta = g(\theta)$，則$g(\widehat{\theta})$ 即為 $g(\theta^*)$ 之MLE。
