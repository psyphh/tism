先備數學知識
================================
## 向量
### 何謂向量？
在進行統計建模時，一筆觀測值（observation）常透過**向量**（vector）來表徵。令 $x$ 表示一 $N$ 維之直行向量（column vector），則其包含了 $x_1,x_2,...,x_N$ 共 $N$ 個元素（element），按順序由上至下排列而成，即

$$
x =
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{pmatrix}
$$

在此講義中，我們通常使用大寫的英文字母作為一組數列的元素個數，而該字母的小寫則作為索引使用，因此，$x$ 的元素個素為 $N$，我們用$x_n$ 來表示 $x$ 的第 $n$ 個元素。

我們可以將 $x$ 進行**轉置**（transpose），將其轉為一橫列向量（row vector），即

$$
x^T =
\begin{pmatrix}
x_1 & x_2 & \cdots & x_N
\end{pmatrix}
$$

透過轉置，原本由上而下的排列，改成有左至右的排列。不過，請記得在文獻中向量一詞，大多指稱的是直行向量。

有時為了節省呈現空間，會將一直行向量寫為 $x = (x_1, x_2,...,x_N)$，利用逗點來分隔不同的上下欄位。注意，這跟橫列向量利用空格將元素左右分隔的做法是不同的。

一個向量**長度**（length），可透過其**範數**（norm）來獲得。範數有許多種定義方式，不過，最常見的為 $L_2$ 範數，即

$$
\begin{aligned}
||x|| &= \sqrt{x_1^2 + x_2^2 +...+x_N^2}\\
& = \sqrt{ \sum_{n=1}^N x_n^2}
\end{aligned}
$$

此式被稱作 $L_2$ 範數的原因在於，其使用了二次方來處理每一項。從公式中可以看出來，唯有當 $x$ 的所有成分皆為0時，其範數才會是0。

當一向量的範數為1時，我們會說其為標準化的向量（normalized vector），給定任何長度不為0的向量 $x$，我們可以透過以下公式對其進行標準化

$$
x^* = \frac{1}{||x||} x =
\begin{pmatrix}
\frac{1}{||x||} x_1 \\
\frac{1}{||x||} x_2 \\
\vdots \\
\frac{1}{||x||} x_N
\end{pmatrix}
$$
這邊牽涉到純量對向量的乘法，請見下一小節。

### 向量的運算
令 $x$ 與 $y$ 表示兩 $N$ 維之向量，則我們可以將向量的加法與減法分別定義為

$$
\begin{aligned}
x  + y=
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{pmatrix}
+
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{pmatrix}
=
\begin{pmatrix}
x_1 + y_1 \\
x_2 + y_2 \\
\vdots \\
x_N + y_N
\end{pmatrix}
\end{aligned}
$$

與

$$
\begin{aligned}
x  - y=
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{pmatrix}
-
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{pmatrix}
=
\begin{pmatrix}
x_1 - y_1 \\
x_2 - y_2 \\
\vdots \\
x_N - y_N
\end{pmatrix}
\end{aligned}
$$

即所謂元素對元素的（element to element）加法與減法。

依循相同的邏輯，我們可定義所謂元素對元素的乘法與除法，不過，在線性代數（linear algebra）此一學門中，甚少直接使用此類的運算子。相較之下，純量對向量之乘法則較常使用。令 $\alpha$ 表示一純量（scalar），純量對向量之乘法定義為

$$
\begin{aligned}
\alpha x =
\alpha
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{pmatrix}
=
\begin{pmatrix}
\alpha x_1 \\
\alpha x_2 \\
\vdots \\
\alpha x_N
\end{pmatrix}
\end{aligned}
$$

### 距離、內積、與餘弦值
在度量兩向量是否相似時，最基本的做法是計算兩者之間的**距離**（distance），即

$$
\begin{aligned}
d(x, y) &= ||x - y|| \\
&=
\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_N - y_N)^2} \\
&=
\sqrt{\sum_{n=1}^N (x_n - y_n)^2 }
\end{aligned}
$$

前述根據 $L_2$ 範數計算的距離，亦稱做 $L_2$ 距離，或是歐幾里德距離（Euclidean distance）。從公式中可以看出來，$x$ 和 $y$ 唯有在所有元素都相等的前提之下，其距離才會等於0。當 $x$ 與 $y$ 的內積為0時，我們會說 $x$ 與 $y$ 為垂直（orthogonal），表示兩向量在 $N$ 為空間中，呈現90度的夾角，兩垂直的向量常被解讀為其具有獨立未重疊的訊息。

另外一種度量兩向量是否相似的做法是，計算其**內積**（inner product），即

$$
\begin{aligned}
\langle x,y \rangle &=
x_1 y_1 + x_2 y_2 + ... + x_N y_N \\
& = \sum_{n=1}^N x_n y_n
\end{aligned}
$$

當 $x$ 與 $y$ 個元素間存在同時大同時小的關係時，兩者的內積會很大，若存在一個大另一個小的關係時，則內積會很小（指的是存在負號的很小），若未存在前述的組型時，則內積會靠近0。

有時，$x$ 與 $y$ 的內積會簡單的寫為 $x^T y$，這與下一小節會提到的矩陣乘法有關。

然而，內積並未考慮到 $x$ 和 $y$ 自身的長度，其數值大小較難直接做解釋。故此，令$\theta$ 表示兩向量之夾角，則其**餘弦值**（cosine）之計算，乃將兩向量之內積除上各自的長度，即

$$
\text{cos}(\theta) = \frac{\langle x,y \rangle}{||x|| ||y||}
$$

當兩向量夾角的餘弦值，可透過以下的方式解釋：
+ $\text{cos}(\theta)$ 靠近1時，表示兩向量在同一方向上，相似性高。
+ $\text{cos}(\theta)$ 靠近-1時，表示兩向量在相反方向上，相似性亦高，但存在相反的關係。
+ $\text{cos}(\theta)$ 靠近0時，表示兩向量靠近垂直的關係，相似性低。

不過，餘弦值大多在向量內部數值皆為正的情境下作為相似性指標（即所謂的第一象限），此時，$0 \leq\text{cos}(\theta) \leq 1$，不需考慮 $-1 \leq\text{cos}(\theta) \leq 0$ 的情況。


## 矩陣

### 何謂矩陣
若把 $M$ 個 $N$ 維的橫列向量由上至下排列，則可形成一尺寸為 $M \times N$ 之**矩陣**（matrix）。這裡，$M$ 為矩陣的橫列（row）個數，$N$ 則為矩陣的直行（column）個數。


舉例來說，令 $a_1, a_2,...,a_M$ 皆表示 $N$ 維之向量，則我們可以將其排為一矩陣 $A$

$$
A =
\begin{pmatrix}
  a_{1}^T \\
   a_{2}^T \\
    \vdots \\
     a_{M}^T \\
 \end{pmatrix}

$$
注意，在這邊，我們有過度使用符號的狀況，在前一小節，$a_n$用於表示向量 $a$ 的第 $n$ 個元素，但在這邊，$a_m$ 被用於表示第 $m$ 個 $N$ 維之向量，讀者應嘗試理解以具備獨立判斷的能力。

若我們將前述的矩陣 $A$ 每一個元素寫開來，則可以表示為

$$
A =
 \begin{pmatrix}
  a_{1,1} & a_{1,2} & \cdots & a_{1,N} \\
  a_{2,1} & a_{2,2} & \cdots & a_{2,N} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{M,1} & a_{M,2} & \cdots & a_{M,N}
 \end{pmatrix}
$$

這裡，$A$ 的第 $(m,n)$ 個元素，我們使用 $a_{m,n}$（或$a_{mn}$）來表示。

給定一尺寸為 $M \times N$之矩陣 $A$，其第 $(m,n)$ 個元素為 $a_{mn}$，則 $A$ 的**轉置**（transpose）被定義為

$$
A^T =
 \begin{pmatrix}
  a_{11} & a_{21} & \cdots & a_{N1} \\
  a_{12} & a_{22} & \cdots & a_{N2} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{1M} & a_{2M} & \cdots & a_{NM}
 \end{pmatrix}
$$
因此，$A^T$ 可視為將 $A$ 的直行與橫列訊息互換的結果，而 $A^T$ 的尺寸則轉為了 $N \times M$。

舉例來說，若矩陣 $A =  \begin{pmatrix}
  1 & 2 & 3  \\
  4 & 5 & 6
 \end{pmatrix}$為一$2 \times 3$的矩陣，則其轉置為一 $3 \times 2$ 之矩陣 $A^T =  \begin{pmatrix}
  1 & 4 \\
  2 & 5 \\
  3 & 6
 \end{pmatrix}$


在矩陣的世界中，有幾種特別的矩陣讀者需要特別認識：

1. 當 $A$ 的尺寸為 $M \times M$時，則 $A$ 被稱作**方陣**（square matrix）。

2. 當 $A$ 為方陣，且進一步滿足 $A = A^T$ 時，則其稱作**對稱矩陣**（symmetric matrix）。

3. 當 $A$ 為方陣，且其只有對角線左下角（可包含對角線）之元素不為0時，其稱作**下三角矩陣**（lower triangular matrix），反之，若只有對角線右上角之元素不為0時，則其稱作**上三角矩陣**（upper triangular matrix）。

4. 當 $A$ 為方陣，且其只有對角線元素不為0時，則其稱作**對角矩陣**（diagonal matrix）。

5. 當 $A$ 為方陣，且其僅有對角線元素數值為1，其它為0時，則其稱作**單位矩陣**（identity matrix）。其在矩陣的世界中，扮演像是純量1的角色。

這幾個特別的矩陣，在計算矩陣乘法、反矩陣、或是拆解時，可能會有一些好的特性讓計算變得比較容易。

### 矩陣的運算
令 $A$ 與 $B$ 皆表示 $M \times N$ 之矩陣，其內部元素分別為 $a_{mn}$ 與 $b_{mn}$，則矩陣加減法，即 $A \pm B$，被定義元素對元素的加減法：

$$
A  + B=
 \begin{pmatrix}
  a_{11} \pm b_{11} & a_{12}\pm  b_{12} & \cdots & a_{1N}\pm  b_{1N} \\
  a_{21} \pm  b_{21} & a_{22}\pm  b_{22} & \cdots & a_{2N}\pm  b_{2N} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{M1} \pm  b_{M1}& a_{M2}\pm  b_{M2} & \cdots & a_{MN}\pm  b_{MN}
 \end{pmatrix}
$$

由於此為元素對元素的運算，因此，$A$ 和 $B$ 兩者的尺寸必須一樣。

既然可以計算元素對元素的加減法，自然也可以同樣定義元素對元素的乘除法，不過，線性代數的領域依然很少直接用到此運算子。

**矩陣乘法**（matrix multiplication）在統計建模中則是相當的關鍵。令 $A$ 表示一 $M \times N$ 之矩陣，$x$ 表示一 $N$ 維之向量，或視為一 $N \times 1$ 的矩陣，則矩陣對向量的乘法被定義為

$$
\begin{aligned}
Ax &=  \underbrace{\begin{pmatrix}
  a_{11} & a_{12} & \cdots & a_{1N} \\
  a_{21} & a_{22} & \cdots & a_{2N} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{M1} & a_{M2} & \cdots & a_{MN}
 \end{pmatrix}}_{M \times N}
\underbrace{\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{pmatrix}}_{N \times 1} \\
&=
\underbrace{\begin{pmatrix}
  a_{11} x_1 + a_{12} x_2 + ... + a_{1N} x_N \\
  a_{21} x_1 + a_{22} x_2 + ... + a_{2N} x_N \\
  \vdots \\
  a_{M1} x_1 + a_{M2} x_2 + ... + a_{MN} x_N
 \end{pmatrix}}_{M \times 1}\\
&=
\begin{pmatrix}
  \sum_{n=1}^N a_{1n} x_n \\
  \sum_{n=1}^N a_{2n} x_n\\
  \vdots \\
  \sum_{n=1}^N a_{Mn} x_n
 \end{pmatrix}
\end{aligned}
$$

在此，我們可以觀察到兩件事情

1. $x$ 的維度，必須跟 $A$ 的直行個數相等，此矩陣向量之乘法才能夠被合法定義。
2. $A$ 和 $x$ 相乘後的結果乃一 $M$ 維之向量，其亦可理解為 $M \times 1$ 之矩陣。
3. $Ax$ 的第 $m$ 個元素，可以視為 $A$ 的第 $m$ 個橫列所形成之向量，對 $x$ 做內積的結果。


因此，若我們將 $A$ 寫為 $A =
\begin{pmatrix}
  a_1^T \\
  a_2^T  \\
  \vdots \\
  a_{M}^T
 \end{pmatrix}$，這裡，$a_m^T$ 表示由 $A$ 的第 $m$ 個橫列形成的橫列向量，則 $Ax$ 可以表示為

 $$
 Ax
 = \begin{pmatrix}
   a_1^Tx  \\
   a_2^Tx  \\
  \vdots \\
   a_M^Tx
 \end{pmatrix}
 = \begin{pmatrix}
  \langle a_1,x \rangle \\
  \langle a_2,x \rangle \\
  \vdots \\
  \langle a_M,x \rangle
 \end{pmatrix}
 $$

我們可延伸前述的概念，來定義矩陣對矩陣的乘法。令 $A$ 與 $B$ 分別表示 $M \times N$ 與 $N \times P$ 之矩陣，其內部元素分別為 $a_{mn}$ 與 $b_{np}$，則 $A$ 與 $B$ 相乘被定義為

$$
\begin{aligned}
AB &=
 \underbrace{\begin{pmatrix}
  a_{11} & a_{12} & \cdots & a_{1N} \\
  a_{21} & a_{22} & \cdots & a_{2N} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{M1} & a_{M2} & \cdots & a_{MN}
 \end{pmatrix}}_{M \times N}
  \underbrace{\begin{pmatrix}
  b_{11} & b_{12} & \cdots & b_{1P} \\
  b_{21} & b_{22} & \cdots & b_{2P} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  b_{N1} & b_{N2} & \cdots & b_{NP}
 \end{pmatrix}}_{N \times P} \\
  &=
  \underbrace{\begin{pmatrix}
  \sum_{n=1}^N a_{1n} b_{n1} & \sum_{n=1}^N a_{1n} b_{n2} & \cdots & \sum_{n=1}^N a_{1n} b_{nP} \\
  \sum_{n=1}^N a_{2n} b_{n1} & \sum_{n=1}^N a_{2n} b_{n2} & \cdots & \sum_{n=1}^N a_{2n} b_{nP} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  \sum_{n=1}^N a_{Nn} b_{n1} & \sum_{n=1}^N a_{Nn} b_{n2} & \cdots & \sum_{n=1}^N a_{Nn} b_{nP}
 \end{pmatrix}}_{M \times P}
 \end{aligned}
$$

其相乘之結果為一 $N \times P$ 之矩陣。

前述的公式，不太容易看出來矩陣乘法的結構，因此，我們將 $A$ 與 $B$ 重新寫成以下的結構

$$
A =
\begin{pmatrix}
  a_1^T \\
  a_2^T  \\
  \vdots \\
  a_{M}^T
 \end{pmatrix},
 B =
\begin{pmatrix}
  b_1 & b_2  & \cdots & b_{P}
 \end{pmatrix}
$$

這裡，$a_m^T$ 表示由 $A$ 的第 $m$ 個橫列形成的橫列向量，$b_p$ 表示由 $B$ 的第 $p$ 個直行形成的直行向量，則 $AB$ 可寫為

$$
AB =
  \begin{pmatrix}
  a_1^T b_1 & a_1^T b_2 & \cdots & a_1^T b_P \\
  a_2^T b_1 & a_2^T b_2 & \cdots & a_2^T b_P \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_M^T b_1 & a_M^T b_2 & \cdots & a_M^T b_P \\
 \end{pmatrix}
$$


### 反矩陣
當 $A$ 為 $N \times N$之方陣，且其行或列並未包含累贅的（redundant）訊息時，則 $A$ 存在**反矩陣**（inverse）。意即，存在一 $N \times N$ 矩陣 $A^{-1}$，其滿足

$$
A^{-1} A = A A^{-1} = I
$$

這裡 $I$ 為 $N \times N$ 單位矩陣。當反矩陣存在時，其為獨特的（unique），意思是，反矩陣只會有一個。

反矩陣的計算常與 $N$ 元一次方程組的求解有關。考慮以下的聯立方程組

$$
\begin{cases}
a_{11} x_1 + a_{12} x_2 + ... + a_{1N} x_N = b_1 \\
a_{21} x_1 + a_{22} x_2 + ... + a_{2N} x_N = b_2 \\
\vdots \\
a_{N1} x_1 + a_{N2} x_2 + ... + a_{NN} x_N = b_N \\
\end{cases}
$$

在此方程組中，有$x_1, x_2,...,x_N$ 共 $N$ 個未知數，並有 $N$ 條式子。令$A$表示由 $a_{mn}$組成的 $N \times N$ 矩陣（$1 \leq m,n\leq N$），$x$ 與 $b$ 則表示由 $x_n$ 與 $b_n$ 形成的 $N$ 維向量。前述方程組可以用矩陣向量乘法的形式來表徵

$$
Ax=b
$$

因此，若可以獲得 $A^{-1}$，則根據 $A^{-1}Ax=A^{-1}b$，方程組的解即為 $x=A^{-1}b$。

反矩陣的計算主要仰賴電腦程式，採用[**高斯消去法**](https://en.wikipedia.org/wiki/Gaussian_elimination)（Gaussian elimination）或是[**QR分解**](https://en.wikipedia.org/wiki/QR_decomposition)（QR decomposition），這兩種方法的計算複雜度皆為 $O(N^3)$，意即，其牽涉到大約 $K \times N^3$ 這麼多步驟的計算，這裡，$K$ 表示一跟 $N$ 無關的常數，因此，當 $N$ 很大時，此反矩陣的計算可能會很耗時。

而讀者需要特別注意的是，究竟 $A$ 是否真的存在反矩陣，$A$ 存在反矩陣的條件為，其 $N$ 個直行（或橫列）所形成的向量，並不存在線性相依（linear dependent）的狀況。令 $a_1, a_2, ..., a_N$ 表示 $A$ 的 $N$ 個直行所對應之向量，若其中存在某 $a_i$，其可以寫為 $a_i = \sum_{n \neq i} w_n a_n$ 的話，則表示有線性相依的狀況。這裡，$w_n$ 為純量，用於表示一權重係數。事實上，線性相依的問題意味著在該聯立方程組的系統中，存在了多餘訊息，各位聰明的讀者，應該都知道此時存在無窮多組解，故不存在 $A^{-1}$。

### 分解
在矩陣的世界中，存在許多將矩陣拆成幾個比較簡單矩陣乘積的**分解**（decomposition），我們在這邊做個簡單的介紹（這邊只考慮實數矩陣，不考慮虛數）。

a. $LU$ 分解

當 $A$ 為 $N \times N$ 可逆方陣時，則其存在 **$LU$ 分解**：

$$
A = LU
$$

+ $L$表示一 $N \times N$ 下三角矩陣。
+ $U$表示一 $N \times N$ 上三角矩陣。


b. $QR$ 分解

當 $A$ 為 $N \times N$ 可逆方陣時，則其存在 **$QR$ 分解**：

$$
A = QR
$$

+ $Q$ 表示一 $N \times N$ 的**垂直矩陣**（orthogonal matrix）（垂直矩陣滿足 $Q Q^T = Q^T Q = I$）。
+ $R$則表示一 $N \times N$ 上三角矩陣。


c. Cholesky 分解

當 $A$ 為 $N \times N$ 正定對稱矩陣（positive definite symmetric matrix）時（在此，我們先不對「正定」一詞多做解釋，其在矩陣中扮演類似正數的概念），則其存在 **Cholesky 分解**：

$$
A = LL^T
$$

+ $L$表示一 $N \times N$ 下三角矩陣，$L^T$ 表示 $L$ 的轉置


d. 特徵值分解

當 $A$ 為 $N \times N$ 對稱矩陣時，其存在以下之**特徵值分解**（eigendecomposition）

$$
A = Q \Lambda Q ^T
$$

+ $\Lambda$ 表示一 $N \times N$ 對角矩陣，其對角線元素為 $A$ 的特徵值。
+ $Q$ 表示一 $N \times N$ 的垂直矩陣，其第 $n$ 個直行表示 $\Lambda$ 第 $n$ 個元素所對應之特徵向量。


e. 奇異值分解

對於任何的 $M \times N$矩陣 $A$，其存在**奇異值分解**（singular value decomposition）：

$$
A = U \Sigma V^T
$$

+ $\Sigma$ 表示一 $M \times N$ 的長方對角矩陣（rectangular diagonal matrix），其對角線元素為 $A$ 的**奇異值**（singular value）
+ $U$ 表示一 $M \times M$ 的垂直矩陣。
+ $V$ 表示一 $N \times N$ 的垂直矩陣。

## 微分
**微分**（differentiation）乃微積分（calculus）此科目中，用於瞭解函數局部訊息的重要技術。在統計領域，此技術常用於逼近目標函數，以及求得參數解有關。

若讀者對於微分的概念很不熟悉的話，可以參考台灣大學開放式課程的[影片](http://ocw.aca.ntu.edu.tw/ntu-ocw/ocw/cou/103S121)，特別是單元5跟6，以及28和29的內容。


### 何謂微分？


給定一函數 $f(x)$，其將一實數變量 $x$，從其**定義域**（domain）透過轉換（transformation）送至其**值域**（range）。若 $f(x)$ 在 $x=x^*$ 此位置為**可微分**（differentiable）的話，我們使用 $f'(x^*)$ 來表示此微分值，其計算方式為

$$
f'(x^*) = \lim_{\Delta x \to 0} \frac{f(x^* + \Delta x) - f(x^*)}{\Delta x}
$$

由於 $\frac{f(x^* + \Delta x) - f(x^*)}{\Delta x}$ 可視為對函數 $f$ 於 $x^*$ 位置斜率之逼近，透過使用極限（limit）的運算將 $\Delta x$ 趨近於0，事實上，$f'(x^*)$ 表徵的就是 $f$ 於 $x^*$ 位置切線之斜率。

實務上，我們常使用以下的符號來表示對函數 $f$ 的 $x$ 進行微分

$$
f'(x) = \frac{\text{d} f(x)}{\text{d} x}
$$

注意，在這邊 $f'(x)$ 與 $f'(x^*)$ 表徵的事情略有不同。$f'(x)$ 表徵的是函數 $f$ 的**一階導數**（first-order derivative），其為一函數，帶入不同的 $x$ 則 $f'(x)$ 會輸出不同的數值來，而 $f'(x^*)$ 強調的是，該一階導數於 $x = x^*$ 之數值。一般來說，比較精確的寫法是

$$
f'(x^*) = \frac{\text{d} f(x)}{\text{d} x} \bigg|_{x = x^*}
$$

下表呈現了一些常見函數的一階導數

|函數名稱|函數形式|一階導數|
|-------|------|-------|
|常數函數|$f(x)=c$|$f'(x)=0$|
|單位函數|$f(x)=x$ | $f'(x)=1$|
|多項式函數|$f(x)=x^K$ | $f'(x)=K x^{K-1}$ |
|  對數函數   |$f(x)=\log(x)$ |$f'(x)=\frac{1}{x}$ |
|  指數函數   |$f(x)=\exp(x)$ |$f'(x)=\exp(x)$ |

在獲得了 $f(x)$ 的一階導數 $f'(x)$ 後，我們也可以將 $f'(x)$ 視為一新的函數再去計算其導數，此時，我們等同於計算 $f(x)$ 的二階導數

$$
f''(x) = \frac{\text{d}^2 f(x)}{\text{d} x^2} = \frac{\text{d} f'(x)}{\text{d} x}
$$


### 微分的規則

前一小節的表格，呈現的都是一些較為簡單形式的函數，面對實務的問題，研究者常須處理較為複雜的函數，這時，我們會需要一些微分的規則。令 $f(x)$ 與 $g(x)$ 表示兩定義於 $x$ 此變數之函數，假設 $f(x)$ 與 $g(x)$ 皆為可微分之函數，則我們有以下的規則

+ **線性規則**：$\frac{\text{d} }{\text{d}x} ( a f(x) + b g(x)) = a f'(x) + b g'(x)$

+ **乘法規則**：$\frac{\text{d} }{\text{d}x} (f(x) g(x)) = f'(x) g(x) + f(x) g'(x)$

+ **除法規則**：$\frac{\text{d} }{\text{d}x} (f(x)/g(x)) = \frac{f'(x) g(x) - f(x) g'(x)}{[g(x)]^2}$（$g(x)$ 不可為0）

除此之外，**連鎖規則**（chain rule）亦為一相當重要的手法。令 $g(x)$ 表示一定義於 $x$ 變數的函數，其輸出為 $y$，而 $f(y)$ 表示一定義於變數 $y$ 之函數，其輸出為 $z$。考慮一組成函數（composition function）$h$，其建構方式為$h =f \circ g$，意思是，$h(x) = f(g(x))$。根據連鎖規則，$h$ 的一階導數可透過以下的公式計算

$$
h'(x) = f'(g(x))  g'(x)
$$


### 多變數函數的微分
在統計的問題中，研究者常須處理多變數的實函數，意即，一函數$f$ 其定義於 $x_1,x_2,...,x_N$，並透過一轉換將此 $N$ 個變數送到一實數空間。在面對此類函數時，我們需要利用**偏微分**（partial differentiation）的技術，計算 $f(x)=f(x_1,x_2,...,x_N)$ 於各 $x_n$ 方向上的**偏導數**（partial derivative），其定義為

$$
\frac{\partial  f(x)}{\partial x_n}  = \lim_{\Delta x_n \to 0} \frac{f(x_1,...,x_n + \Delta x_n,...,x_N) - f(x_1,...,x_n,...,x_N)}{\Delta x_n}
$$

在實作上，計算 $\frac{\partial f(x)}{\partial x_n}$時，僅須將 $f(x)$ 視為 $x_n$ 的函數，其它的變數視為常數即可。

在獲得 $f(x)$ 對於每個 $x_n$ 的偏導數後，我們可以將這些導數收集起來排成一 $N$ 維之向量，即

$$
\nabla f(x) =
\begin{pmatrix}
\frac{\partial  f(x)}{\partial x_1} \\
\frac{\partial  f(x)}{\partial x_2} \\
\vdots \\
\frac{\partial  f(x)}{\partial x_N} \\
\end{pmatrix}
$$

我們將 $\nabla f(x)$ 稱作**梯度**（gradient），其在了解多變數函數時，提供了相當重要的訊息：

1. 梯度的第 $n$ 個成分，表徵了該函數於 $x_n$ 方向上切線的斜率訊息。
2. 給定一具體的方向 $d$，則該函數於方向 $d$ 的方向導數（directional derivative）可以寫為 $\langle \nabla f(x),d/||d|| \rangle$。




