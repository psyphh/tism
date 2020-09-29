Lab: 張量與線性代數
================

此 lab 中，我們將會透過 `torch` 此套件，學習以下的主題。

1. 認識 `torch` 的張量（tensor）之基礎。

2. 了解如何對 `torch` 張量進行操弄。

3. 使用 `torch` 進行線性代數之運算。

4. 應用前述之知識，建立一可進行線性迴歸分析之類型（class）。

`torch`之安裝與基礎教學，可參考 [PyTorch官方網頁](https://pytorch.org/get-started/locally)，如果讀者使用的是Google的Colab服務，則不需要另外安裝 `torch`。在安裝完 `torch` 後，可透過以下的指令載入

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

+ `a` 的尺寸（size）為 `(3, 4)`，表示 `a` 為一 $3 \times 4$ 之 2d 張量。在進行運算時，張量間的形狀需滿足某些條件，如相同，或是滿足某種廣播（broadcasting）的規則。
+ `a` 的資料類型（data type）為 `int64`，表示64位元的整數。在進行運算時，張量間的類型須相同。

稍後，我們會討論如何獲得 `torch` 張量的尺寸與資料類型。


### 張量之數值
若要擷取 `torch` 張量的資料數值（value），則可透過 `.numpy()`獲得，其回傳該張量對應之 `numpy` 陣列

print("data of tensor is: \n", a.numpy())

`numpy` 陣列是 `python` 進行科學運算時，幾乎都會仰賴的資料格式。因此，`.numpy()` 此指令主要用於不同套件間的資料交換，或是希望列印出來的結果比較簡單使用。

在形成張量時，要記得張量的變數名稱，僅為其表徵資料的一個標籤，而相同的資料，可以有許多不同的標籤指稱其。舉例來說，考慮一下的程式碼

c = torch.tensor([[1, 1, 1], [2, 2, 2]])
d = c
print("tensor c is: \n",
      c.numpy())
print("tensor d is: \n",
      d.numpy())


沒有意外的，`c` 和 `d` 內部的數值是一樣的。然而，若我們利用 `.fill_()` 方法，將 `c` 的內部全部填入 0 的話，則我們可以看到不僅是 `c`，`d` 內部的資料亦改變了。

c.fill_(0)
print("tensor c is: \n",
      c.numpy())
print("tensor d is: \n",
      d.numpy())

在此，讀者需特別注意的是，`torch` 中若有方法的尾端是底線 `_` 的話，則意味著該方法會取代原有物件中的資料，如 `c` 此張量對應的資料直接被取代掉了，不需要另外寫 `c = c.fill_(0)`。

前述的設計，主要是為了避免資料的拷貝，以減少記憶體的使用。如果說希望 `d` 表徵的資料為 `c` 對應資料的拷貝的話，可以使用 `.clone()`此方法：

d = c.clone()


如此，就不會出現更動 `c` 的資料，`d` 也跟著動的狀況發生。


`torch` 內建了多種函數，以協助產生具有特別數值結構之張量：

print("tensor with all elements being ones \n",
      torch.ones(size = (4, 1)).numpy())
print("tensor with all elements being zeros \n",
      torch.zeros(size = (2, 3)).numpy())
print("identity-like tensor \n",
      torch.eye(n = 3, m = 5).numpy())
print("diagonal matrix \n",
      torch.diag(input = torch.tensor([1, 2, 3, 4])).numpy())

`torch` 亦可隨機地產生資料，或是直接使用尚未起始化的資料：

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

`.view()` 所回傳的張量，即為原本張量之資料在不同尺寸下對應之張量，`torch` 並未對資料進行拷貝（copy）。考慮以下的程式碼

c = torch.zeros(4, 2)
d = c.view((8, ))
print("tensor c is: \n",
      c.numpy())
print("tensor d is: \n",
      d.numpy())

如預期地，`c` 與 `d` 內部有著相同的數值，僅差在尺寸有所不同。接著，我們將 `c` 的內部全部填入1，我們會觀察到 `d` 的數值也跟著改變了：

c.fill_(1)
print("tensor c is: \n",
      c.numpy())
print("tensor d is: \n",
      d.numpy())

這顯示 `c` 與 `d` 背後有著共享的資料內容。


在0.4版之後，`.reshape()` 此方法亦可改變張量的尺寸，但其有可能會對資料進行拷貝，不過，當資料本身的排列不具有連續性時，僅 `.reshape()` 能夠使用。因此，在 `torch` 的[官方文件](https://pytorch.org/docs/master/tensors.html#torch.Tensor.view)中，建議使用 `.reshape()`此指令。

### 張量之資料類型
張量的資料類型，可透過 `.dtype` 方法獲得

print("data type of tensor is", a.dtype)

若是要調整資料類型的話，則可透過 `.type()` 此方法：

print(a.type(torch.float64))

`torch` 內建多種資料類型，包含整數類型（如 `torch.int32` 與 `torch.int64`）與浮點數類型（如 `torch.float32` 與 `torch.float64`），完整的資料類型請見 [`torch.Tensor`文件](https://pytorch.org/docs/stable/tensors.html)之 `dtype` 欄位。

在進行張量的數學運算時，請務必確認張量間的資料類型都是一致的，而 `torch` 常用之資料類型為 `torch.float32` 與 `torch.float64`，前者所需的記憶體較小，但運算結果的數值誤差較大。


除了利用 `torch.tensor()` 搭配 `dtype` 產生張量外，另一種產生張量的方法為，直接利用特定資料型態張量的建構式，再利用特定的方法填入數值。例如，以下的程式碼在 CPU 先產稱了一資料類型為 `torch.float64` 之張量，再利用均勻分配隨機生成資料填入：

c = torch.FloatTensor(2, 4).uniform_()


前述之張量建構風格，在後續討論到 GPU 計算時會很有幫助。不同資料類型之建構式，可同樣參考 [torch.Tensor文件](https://pytorch.org/docs/stable/tensors.html) 之 `CPU tensor` 欄位（若要在 GPU 上建構張量，則是參考 `GPU tensor` 之欄位）。


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
print("transpose of a is \n",
      a.t().numpy())

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


## 張量運算之進階議題

### 廣播
在先前討論到兩張量進行元素對元素的加減乘除時，有一個條件是兩張量的尺寸必須相等。然而，此條件並非必要的。考慮以下的範例：

a = torch.tensor([[1, 2, 3]],
                 dtype = torch.float64)
b = torch.tensor([2],
                 dtype = torch.float64)
print("a * b is \n", a * b)

我們可以看到，雖然 `a` 和 `b` 的尺寸並不相同，但 `torch` 仍然會根據某種規則，將 `b` 的數值分配給 `a` 進行元素對元素的運算，這樣的特性被稱作廣播（broadcasting）。廣播的優點在於，其一方面可以處理不同尺寸張量的運算，二方面則是提供了高效率的計算。

廣播的概念承襲自 `numpy` 套件，其詳細的運作機制可以參考 [Array Broadcasting in Numpy](https://numpy.org/doc/stable/user/theory.broadcasting.html) 一文。當以下條件滿足時，則 `torch` 可進行廣播：

+ 兩張量從尾端軸（trailing axes）往前算回來的尺寸相同，或是其中一張量之維度必須是 1。

以下之範例為尺寸 `(2, 3)` 與 `(3, )` 兩張量之乘法，由於兩張量從尾端軸算回來的維度都是 `(3, )`，故符合前述條件。

a = torch.tensor([[1, 2, 3], [4, 5, 6]],
                 dtype = torch.float64)
b = torch.tensor([0, 1, 2],
                 dtype = torch.float64)
print("tensor a is \n", a)
print("tensor b is \n", b)
print("tensor a * b is \n", a * b)

以下之範例為尺寸 `(2, 3)` 與 `(2, 1)` 兩張量之乘法，其從尾端軸算回來的維度為相等或是其中一張量等於1，亦符合廣播條件

a = torch.tensor([[1, 2, 3], [4, 5, 6]],
                 dtype = torch.float64)
b = torch.tensor([[0], [1]],
                 dtype = torch.float64)
print("tensor a is \n", a)
print("tensor b is \n", b)
print("tensor a * b is \n", a * b)

然而，若是尺寸為 `(2, 3)` 與 `(2, )` 之張量相乘的，則不符合廣播條件，會產生錯誤訊息。


### GPU運算
當執行程式碼之機器支援GPU運算時，`torch` 可利用GPU獲得高效能之運算。而機器是否具有GPU 支援，可透過 `torch.cuda.is_available()` 此指令進行判定。若使用 Google 的 Colab 服務，進行 GPU 運算需要點選 `Runtime\Change runtime type`，在 `Hardware accelerator` 處點選 `GPU`，才會轉移到具有 GPU 之機器進行計算。

為了瞭解 GPU 在計算上的幫助，我們撰寫了一個函數 `math_speed_test`，其內部生成一 `(n, n)` 之方陣 `a`，接著對其連續進行 `iter_max` 次的連加，並記錄完成整個工作所需的時間。

import time
def math_speed_test(n, iter_max, device):
    a = torch.zeros((n, n), dtype = torch.float64).to(device)
    start_time = time.time()
    for _ in range(iter_max):
        a += a
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(device.upper(), 'time =', elapsed_time)

`torch` 進行 GPU 運算的關鍵步驟在於，利用 `.to("cuda")` 方法將資料送到 GPU 上，而如果將指令改為 `.to("cpu")`，則意味著我們將使用一般的 CPU 進行運算。

接下來的程式碼，我們將 `iter_max` 設為 100，`n` 設為 `10`、`100`、以及`1000`，觀察計算時間的變化（注意，下述程式碼得在支援 GPU 之機器上執行，才可以看見 GPU 之計算時間，否則僅會呈現 CPU 的結果）：

iter_max = 100
for n in [10, 100, 1000]:
    print("n =", n)
    math_speed_test(n = n, iter_max = iter_max, device = "cpu")
    if torch.cuda.is_available():
        math_speed_test(n = n, iter_max = iter_max, device = "cuda")

大體上，我們可以觀察到在 `n` 數值為 `10` 或 `100` 時，GPU並沒有顯著的幫助，甚至可能表現得更差。然而，在 `n = 1000`，GPU 可大幅提升計算的速度。

除了數學運算外，隨機亂數之生成亦可透過 GPU 顯著的加速，可參考以下之程式碼：

n = 1000

%timeit torch.FloatTensor(n, n).uniform_()

if torch.cuda.is_available():
    %timeit torch.cuda.FloatTensor(n, n).uniform_()

在這裡，我們使用了 `IPython` 的魔術指令 `%timeit` ，其可用於評估該行程式碼之計算時間。如果要評估的是整個程式塊（code block）的時間的話，可以改為使用 `%%timeit` 置於程式塊的開頭。

另外，線性代數亦可獲得 GPU 之幫助

%timeit torch.FloatTensor(n, n).uniform_() @ torch.FloatTensor(n, n).uniform_()

if torch.cuda.is_available():
    %timeit torch.cuda.FloatTensor(n, n).uniform_() @ torch.cuda.FloatTensor(n, n).uniform_()



然而，部分線性代數的計算則未必可以獲得 GPU 的強力幫助，如計算反矩陣

%timeit torch.FloatTensor(n, n).uniform_().inverse()

if torch.cuda.is_available():
    %timeit torch.cuda.FloatTensor(n, n).uniform_().inverse()

一個數學運算是否能夠獲得 GPU 幫助的關鍵在於，該運算是否能夠被平行化。如果讀者曾經有使用過高斯消去法（Gaussian elimination）來解反矩陣的話，就知道高斯消去法是個得序列求解的方法，故 GPU 幫助不大。


在實務上，我們可以透過以下的程式碼來檢驗機器是否有可供使用的GPU，以動態決定 `device` 應為何：

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

## 實作範例

### 產生線性迴歸資料

在開始之前，我們先設定一種子，以讓後續的亂數生成都能夠獲得相同的結果（不過，這裡的 `torch.manual_seed` 僅適用於 CPU 之張量，若使用GPU，請改為 `torch.cuda.manual_seed`）。

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

## 作業

1. 請根據本章節的內容，進行一系列的實驗，以了解在哪些運算與張量尺寸上，GPU 計算才具有其優勢。

2. 請在 `LinearRegression` 此類型中，加入以下新的功能：
+ 新增一個選項，以決定 `x` 是否需要進行標準化（每個變項的平均數為0，變異數為1，不准使用其它的套件）
+ 新增一個選項，以決定是否使用 GPU 進行運算。
+ 新增一方法 `predict()`，其輸入為一新的 `x`，輸出為在該 `x` 下，`y` 的預測值。
+ 新增一方法 `gradient()`，其可獲得在當前係數估計下，最小平方估計準則的梯度數值。