Lab: 張量與線性代數
================

此 lab 中，我們將會透過 `torch` 此套件，學習以下的主題。

1. 認識 `torch` 的張量（tensor）之基礎。

2. 了解如何對 `torch` 張量進行操弄。

3. 使用 `torch` 進行線性代數之運算。

4. 應用前述之知識，建立一可進行線性迴歸分析之類型（class）。

`torch`之安裝與基礎教學，可參考 [PyTorch官方網頁](https://pytorch.org/get-started/locally)。在安裝完成後，可透過以下的指令載入

import torch

## 張量之基礎

### 張量之輸入
`torch` 最基本的物件是張量（tensor），其與 `numpy` 的陣列（array）相當的類似。產生一個張量最基本的方法為，將所欲形成張量的資料（其可為 `python` 的 `list` 或是 `numpy` 的 `ndarray`），置於`torch.tensor`函數中


a = torch.tensor(data = [[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]])
type(a)

透過 `type()`，可看見其屬於 `torch.Tensor` 此一類型（class），若欲了解 `a` 的樣貌，我們可使用 `print` 指令來列印其主要的內容

print(a)

透過對 `a` 列印的結果，我們可觀察到：

+ `a` 內部的資料數值（value）為 `[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`。

除此之外，`a` 還有兩個重要的屬性並未顯示在列印的結果中：

+ `a` 的尺寸（size）為 `(3, 4)`，表示 `a` 為一 $3 \times 4$ 之張量。在進行運算時，張量間的形狀需滿足某些條件，如相同，或是滿足某種廣播（broadcasting）的規則。
+ `a` 的資料類型（data type）為 `int64`，表示64位元的整數。在進行運算時，張量間的類型須相同。

稍後，我們會討論如何獲得 `torch` 張量的尺寸與資料類型。


### 張量之數值
若要獲得張量的資料數值（value），可透過 `.numpy()`獲得，其回傳該張量對應之 `numpy` 陣列

print("data of tensor is: \n", a.numpy())

 `numpy` 陣列是 `python` 進行科學運算時，幾乎都會仰賴的資料格式。`torch` 內建了多種函數，以協助產生具有特別數值結構之張量：

print("tensor with all elements being ones \n",
      torch.ones(size = (4, 1)).numpy())
print("tensor with all elements being zeros \n",
      torch.zeros(size = (2, 3)).numpy())
print("identity-like tensor \n",
      torch.eye(n = 3, m = 5).numpy())
print("diagonal matrix \n",
      torch.diag(input = torch.tensor([1, 2, 3, 4])).numpy())

`torch` 亦可指定分配產生隨機的資料

print("tensor with random elements from uniform(0, 1) \n",
      torch.rand(size = (2, 6)).numpy())
print("tensor with uninitialized data \n",
      torch.empty(size = (2, 6)).numpy())

### 張量之形狀

張量之形狀與形狀維度之數量，可透過張量物件的 `.size()`（或 `.shape`） 與 `dim` 方法來獲得

print("size of tensor is", a.size())
print("size of tensor is", a.shape)
print("dim of tensor is", a.dim())

如果要對張量的形狀進行改變的話，可透過 `.view()` 此方法獲得

print("tensor with shape (4, 3): \n",
      a.view(size = (4, 3)).numpy())
print("tensor with shape (2, 2, 3): \n",
      a.view(size = (2, 2, 3)).numpy())
print("tensor with shape (12, 1): \n",
      a.view(size = (12, 1)).numpy())
print("tensor with shape (12, 1) by (-1, 1): \n",
      a.view(size = (-1, 1)).numpy())
print("tensor with shape (12,): \n",
      a.view(size = (12, )).numpy())


注意，`(12, 1)` 與 `(12,)` 兩種形狀是不一樣的，前者為2d的張量，後者為1d的張量。在進行張量操弄時，若將兩者混淆，很可能會帶來錯誤的計算結果。另外，-1表示該面向對應之尺寸，由其它面向決定。


### 張量之資料類型
張量的資料類型，可透過 `.dtype` 方法獲得

print("data type of tensor is", a.dtype)

若是要調整資料類型的話，則可透過 `.type()` 此方法：

print(a.type(torch.float64))

`torch` 內建多種資料類型，包含整數類型（如 `torch.int32` 與 `torch.int64`）與浮點數類型（如 `torch.float32` 與 `torch.float64`），完整的資料類型請見 [torch.Tensor文件](https://pytorch.org/docs/stable/tensors.html)。

在進行張量的數學運算時，請務必確認張量間的資料類型都是一致的，而 `torch` 常用之資料類型為 `torch.float32` 與 `torch.float64`，前者所需的記憶體較小，但運算結果的數值誤差較大。


## 張量之操弄

### 張量之切片

若要擷取一張量特定的行（row）或列（column）的話，則可透過切片（slicing）的功能獲得。`torch` 張量的切片方式，與 `numpy` 類似，皆使用中括號 `[]`，再搭配所欲擷取資料行列的索引（index）獲得。

print("extract 1st row: \n",
      a[0, :].numpy())
print("extract 1st and 2nd rows: \n",
      a[:2, :].numpy())
print("extract 2nd column: \n",
      a[:, 1].numpy())
print("extract 2nd and 3rd columns: \n",
      a[:, 1:3].numpy())

進行切片時，有幾項重點需要注意。

+ 各面向之索引從0開始。
+ 負號表示從結尾數回來，如 `-1` 表示最後一個位置。
+ `:`表示該面向所有元素皆挑選。
+ `start:stop` 表示從 `start` 開始挑選到 `stop-1`。
+ `start:stop:step` 表示從 `start` 開始到 `stop-1`，間隔 `step` 挑選。

### 張量之串接
多個張量在維度可對應之前提下，可透過 `torch.cat` 串接

print("vertical concatenation \n",
      torch.cat([a, a], dim = 0).numpy())
print("horizontal concatenation \n",
      torch.cat([a, a], dim = 1).numpy())

## 張量之運算
考慮以下 `a` 與 `b` 兩張量

a = torch.tensor(data = [[1, 2], [3, 4], [5, 6]],
                dtype = torch.float64)
b = torch.tensor(data = [[1, 2], [1, 2], [1, 2]],
                dtype = torch.float64)
print("tensor a is \n", a.numpy())
print("tensor b is \n", b.numpy())


我們將使用 `a` 與 `b` 來展示如何使用 `torch` 進行張量間的計算。

### 張量元素對元素之運算
透過 `torch` 的數學函數，可進行張量元素對元素的四則運算

print("element-wise add \n",
      torch.add(a, b))
print("element-wise subtract \n",
      torch.sub(a, b))
print("element-wise multiply \n",
      torch.mul(a, b))
print("element-wise divide \n",
      torch.div(a, b))

前述採用的函數，皆可取代為其所對應之運算子計算

print("element-wise add \n", a + b)
print("element-wise subtract \n", a - b)
print("element-wise multiply \n", a * b)
print("element-wise divide \n", a / b)

若需要進行絕對值、對數、指數等較為進階之數學運算，可以至 [troch官方文件](https://pytorch.org/docs/stable/torch.html#math-operations) 此模組中尋找對應的數學函數。

### 張量線性代數之運算
除了簡單的四則運算外，當張量的 `dim` 為2時，`torch` 提供了進行線性代數（linear algebra）相關的函數，如

+ 矩陣轉置（matrix transpose）

a_t = torch.transpose(input=a, dim0=0, dim1=1)
print("transpose of a is \n",
      a_t.numpy())

+ 矩陣乘法（matrix multiplication）

c = a_t @ a
print("c = a_t @ a is \n",
      c.numpy())

+ 反矩陣（matrix inverse）

c_inv = torch.inverse(input = c)
print("inverse of c is \n",
      c_inv.numpy()) # c @ c_inv should be identity matrix
print("check for inverse (left) \n",
      (c_inv @ c).numpy())
print("check for inverse (right) \n",
      (c @ c_inv).numpy())

+ Cholesky 拆解（Cholesky decomposition）

c_chol = torch.cholesky(input = c)
print("Cholesky factor of c is \n",
      c_chol.numpy())
print("check for Cholesky decomposition \n",
      (c_chol @ torch.transpose(c_chol, 0 , 1)).numpy())

+ 特徵拆解（eigen-decomposition）

e, v = torch.symeig(input = c, eigenvectors=True)
print("eigenvalue of c is \n",
      e.numpy())
print("eigenvector of c is \n",
      v.numpy())
print("check for eigen-decomposition \n",
      (v @ torch.diag(e) @
       torch.transpose(v, 0 , 1)).numpy())


+ 奇異值拆解（singular value decomposition）

u, s, v = torch.svd(input = a)
print("singular value of a is \n",
      s.numpy())
print("left singular vector of a is \n",
      u.numpy())
print("right singular vector of a is \n",
      v.numpy())
print("check for singular value decomposition \n",
      (u @ torch.diag(s) @
       torch.transpose(v, 0, 1)).numpy())


### 對張量之數值進行摘要
`torch` 提供了一些化約（reduction）的函數，對張量內的數值進行摘要

print("calculate mean \n",
      torch.mean(input = a).numpy())
print("calculate standard deviation \n",
      torch.std(input = a).numpy())
print("calculate max \n",
      torch.max(input = a).numpy())
print("calculate min \n",
      torch.min(input = a).numpy())

我們亦可對張量的各面向，進行前述的摘要。以平均數為例：

print("calculate mean for each column \n",
      torch.mean(input = a, dim=0).numpy())
print("calculate mean for each row \n",
      torch.mean(input = a, dim=1).numpy())

其它的化約函數，可以參考[官方文件](https://pytorch.org/docs/stable/torch.html#reduction-ops)。

## 實徵範例

### 產生線性迴歸資料

在開始之前，我們先設定一種子，以讓後續的亂數生成都能夠獲得相同的結果（不過，這裡的 `manual_seed` 僅適用於CPU，若使用GPU，請改為 `torch.cuda.manual_seed`）。

torch.manual_seed(48)


# define a function to generate x and y
def generate_data(n_sample, weight,
                  bias = 0,
                  std_residual = 1,
                  mean_feature = 0,
                  std_feature = 1,
                  dtype = torch.float64):
    weight = torch.tensor(weight, dtype = dtype)
    n_feature = weight.shape[0]
    x = torch.normal(mean = mean_feature,
                     std = std_feature,
                     size = (n_sample, n_feature),
                     dtype = dtype)
    e = torch.normal(mean = 0,
                     std = std_residual,
                     size = (n_sample, 1),
                     dtype = dtype)
    weight = weight.view(size = (-1, 1))
    y = bias + x @ weight + e
    return x, y

# run generate_data
x, y = generate_data(n_sample = 10,
                     weight = [-5, 3, 0],
                     bias = 5,
                     std_residual = 1,
                     mean_feature = 10,
                     std_feature = 3,
                     dtype = torch.float64)
print("feature matrix x is \n", x.numpy())
print("response vector y is \n", y.numpy())



### 計算模型參數

# define a function to calculate model parameter
def calculate_parameter(x, y, dtype = torch.float64):
    if x.dtype is not dtype:
        x = x.type(dtype = dtype)
    if y.dtype is not dtype:
        y = y.type(dtype = dtype)
    u = torch.ones(size = (x.size()[0], 1), dtype = dtype)
    x_design = torch.cat([u, x], dim = 1)
    parameter = torch.inverse(
        torch.transpose(x_design, dim0=0, dim1=1) @ x_design) @ \
                torch.transpose(x_design, dim0=0, dim1=1) @ y
    bias = parameter[0, 0]
    weight = parameter[1:, 0]
    return bias, weight

# run calculate_parameter
x, y = generate_data(n_sample = 1000,
                     weight = [-5, 3, 0],
                     bias = 5,
                     std_residual = 1,
                     mean_feature = 10,
                     std_feature = 3,
                     dtype = torch.float64)
bias, weight = calculate_parameter(x, y)
print("bias estimate is \n", bias.numpy())
print("weight estimate is \n", weight.numpy())


### 建立一進行迴歸分析之物件

# define a class to fit linear regression
class LinearRegression():
    def __init__(self, dtype = torch.float64):
        self.dtype = dtype
        self.bias = None
        self.weight = None
    def fit(self, x, y):
        if x.dtype is not self.dtype:
            x = x.type(dtype = self.dtype)
        if y.dtype is not self.dtype:
            y = y.type(dtype = self.dtype)
        u = torch.ones(size = (x.size()[0], 1), dtype = self.dtype)
        x_design = torch.cat([u, x], dim = 1)
        parameter = torch.inverse(
            torch.transpose(x_design, dim0=0, dim1=1) @ x_design) @ \
                    torch.transpose(x_design, dim0=0, dim1=1) @ y
        self.bias = parameter[0, 0]
        self.weight = parameter[1:, 0]
        return self

model_lr = LinearRegression()
model_lr.fit(x, y)
print("bias estimate is \n", model_lr.bias.numpy())
print("weight estimate is \n", model_lr.weight.numpy())

