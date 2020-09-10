先備數學知識
================================
## 向量
### 何謂向量？
在進行統計建模時，一筆觀測值（observation）常透過向量（vector）來表徵。令 $x$ 表示一 $M$ 維之直行向量（column vector），則其包含了 $x_1,x_2,...,x_M$ 共 $M$ 個元素（element），按順序由上至下排列而成，即

$$
x =
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_M
\end{pmatrix}
$$

在此講義中，我們將使用 $x_m$ 來表示 $x$ 的第 $m$ 個元素。

我們可以將 $x$ 轉置（transpose），將其轉為一橫列向量，即

$$
x^T =
\begin{pmatrix}
x_1 & x_2 & \cdots & x_M
\end{pmatrix}
$$

透過轉置，原本由上而下的排列，改成有左至右的排列。不過，請記得在文獻中向量一詞，大多指稱的是直行向量。


一個向量長度（length），可透過其範數（norm）來獲得。範數有許多種定義方式，不過，最常見的為 $L_2$ 範數，即

$$
\begin{aligned}
||x|| &= \sqrt{x_1^2 + x_2^2 +...+x_M^2}\\
& = \sqrt{ \sum_{m=1}^M x_m^2}
\end{aligned}
$$

唯有當 $x$ 的所有成分皆為0時，其範數才會是0。

當一向量的範數為1時，我們會說其為標準化的（normalized）向量，給定任何長度不為0的向量 $x$，我們可以透過以下公式對其進行標準化

$$
x^* = \frac{1}{||x||} x =
\begin{pmatrix}
\frac{1}{||x||} x_1 \\
\frac{1}{||x||} x_2 \\
\vdots \\
\frac{1}{||x||} x_M
\end{pmatrix}
$$
這邊牽涉到純量對向量的乘法，請見下一小節。

### 向量的運算
令 $x$ 與 $y$ 表示兩 $M$ 維之向量，則我們可以將向量的加減法分別定義為

$$
\begin{aligned}
x  + y=
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_M
\end{pmatrix}
+
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_M
\end{pmatrix}
=
\begin{pmatrix}
x_1 + y_1 \\
x_2 + y_2 \\
\vdots \\
x_M + y_M
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
x_M
\end{pmatrix}
-
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_M
\end{pmatrix}
=
\begin{pmatrix}
x_1 - y_1 \\
x_2 - y_2 \\
\vdots \\
x_M - y_M
\end{pmatrix}
\end{aligned}
$$

即所謂元素對元素的（element to element）加減法。依循相同的邏輯，我們可定義所謂元素對元素的乘除法，不過，在線性代數（linear algebra）此一學門中，甚少直接使用此類的運算子。相較之下，令 $\alpha$ 表示一純量（scalar），則我們可定義純量對向量之乘法

$$
\begin{aligned}
\alpha x =
\alpha
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_M
\end{pmatrix}
=
\begin{pmatrix}
\alpha x_1 \\
\alpha x_2 \\
\vdots \\
\alpha x_M
\end{pmatrix}
\end{aligned}
$$

### 距離、內積、與餘弦值
在度量兩向量是否相似時，最基本的做法是計算兩者之間的距離，即

$$
\begin{aligned}
d(x, y) &= ||x - y|| \\
&=
\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_M - y_M)^2} \\
&=
\sqrt{\sum_{m=1}^M (x_m - y_m)^2 }
\end{aligned}
$$

前述根據 $L_2$ 範數計算的距離，亦稱做 $L_2$ 距離，或是歐幾里德距離（Euclidean distance）。從公式中可以看出來，$x$ 和 $y$ 唯有在所有元素都相等的前提之下，其距離才會等於0。當 $x$ 與 $y$ 的內積為0時，我們會說 $x$ 與 $y$ 為垂直（orthogonal），表示兩向量在 $M$ 為空間中，呈現90度的夾角，兩垂直的向量常被解讀為其具有獨立未重疊的訊息。

另外一種度量兩向量是否相似的做法是，計算其內積（inner product），即

$$
\begin{aligned}
\langle x,y \rangle &=
x_1 y_1 + x_2 y_2 + ... + x_M y_M \\
& = \sum_{m=1}^M x_m y_m
\end{aligned}
$$

當 $x$ 與 $y$ 個元素間存在同時大同時小的關係時，兩者的內積會很大，若存在一個大另一個小的關係時，則內積會很小（指的是存在負號的很小），若未存在前述的組型時，則內積會靠近0。

有時，$x$ 與 $y$ 的內積會簡單的寫為 $x^T y$，這與下一小節會提到的矩陣乘法有關。

然而，內積並未考慮到 $x$ 和 $y$ 自身的長度，其數值大小較難直接做解釋。故此，令$\theta$ 表示兩向量之夾角，則其餘弦值（cosine）之計算，乃將兩向量之內積除上各自的長度，即

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
若把 $N$ 個 $M$ 維的向量由左至右排列，則可形成一尺寸為 $M \times N$ 之矩陣（matrix）。這裡，$M$ 為矩陣的橫列（row）個數，$N$ 則為矩陣的直行（column）個數。


舉例來說，令 $a_1, a_2,...,a_N$ 皆表示 $M$ 維之向量，則我們可以將其排為一矩陣 $A$

$$
A =
\underbrace{ \begin{pmatrix}
  a_{1} & a_{2} & \cdots & a_{N} \\
 \end{pmatrix}}_{N \ \text{Vectors}}

$$
注意，在這邊，我們有過度使用符號的狀況，在前一小節，$a_n$用於表示向量 $a$ 的第 $n$ 個元素，但在這邊，$a_n$ 被用於表示第 $n$ 個 $M$ 維之向量，讀者應嘗試理解後具備獨立判斷的能力。

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

給定一尺寸為 $M \times N$之矩陣 $A$，其第 $(m,n)$ 個元素為 $a_{m,n}$，則 $A$ 的轉置（transpose）被定義為

$$
A^T =
 \begin{pmatrix}
  a_{1,1} & a_{2,1} & \cdots & a_{N,1} \\
  a_{1,2} & a_{2,2} & \cdots & a_{N,2} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{1,M} & a_{2,M} & \cdots & a_{N,M}
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

當 $A$ 的尺寸為 $M \times M$時，則 $A$ 被稱作方陣（square matrix）。而一個方陣又進一步滿足 $A = A^T$ 時，則其稱作對稱矩陣（symmetric matrix）。不管是方陣或是對稱矩陣，其都具有一些好的特性，我們將會在後續的小節進行說明。

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

矩陣乘法（matrix multiplication）在統計建模中則是相當的關鍵。令 $A$ 表示一 $M \times N$ 之矩陣，$x$ 表示一 $N$ 維之向量（注意尺寸跟維度），則矩陣對向量的乘法被定義為

$$
\begin{aligned}
Ax &=  \begin{pmatrix}
  a_{11} & a_{12} & \cdots & a_{1N} \\
  a_{21} & a_{22} & \cdots & a_{2N} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  a_{M1} & a_{M2} & \cdots & a_{MN}
 \end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_N
\end{pmatrix} \\
&=
\begin{pmatrix}
  a_{11} x_1 + a_{12} x_2 + ... + a_{1N} x_N \\
  a_{21} x_1 + a_{22} x_2 + ... + a_{2N} x_N \\
  \vdots \\
  a_{M1} x_1 + a_{M2} x_2 + ... + a_{MN} x_N
 \end{pmatrix}\\
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


## 反矩陣


