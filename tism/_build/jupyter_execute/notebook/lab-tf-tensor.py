Lab I: 線性代數
================

此 lab 中，我們將會透過 `tensorflow` 此套件，學習以下的主題。

1. 認識 `tensorflow` 的張量（tensor）之基礎。

2. 了解如何對 `tensorflow` 張量進行操弄。

3. 使用 `tensorflow` 進行線性代數之運算。

4. 應用前述之知識，建立一可進行線性迴歸分析之類型（class）。

`tensorflow`之安裝與基礎教學，可參考 [tensorflow官方網頁](https://www.tensorflow.org)。在安裝完成後，可透過以下的指令載入

import tensorflow as tf

## 張量之基礎

### 張量之輸入
`tensorflow` 最基本的物件是張量（tensor），其與 `numpy` 的陣列（array）相當的類似。產生一個張量最基本的方法為，將所欲形成張量的資料（其可為 `python` 的 `list` 或是 `numpy` 的 `ndarray`），置於`tf.constant`函數中

a = tf.constant(value = [[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]])
type(a)

透過 `type()`，可看見其屬於 `EagerTensor` 此一類型（class），若欲了解 `a` 的樣貌，我們可使用 `print` 指令來列印其主要的內容

print(a)

透過對 `a` 列印的結果，我們可觀察到：

+ `a` 內部的資料數值（value）為 `[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`。
+ `a` 的形狀（shape）為 `(3, 4)`，表示 `a` 為一 $3 \times 4$ 之張量。在進行運算時，張量間的形狀需滿足某些條件，如相同，或是滿足某種廣播（broadcasting）的規則
+ `a` 的資料類型（data type）為 `int32`，表示32位元的整數。在進行運算時，張量間的類型須相同。一般來說，

### 張量之數值
若要獲得張量的資料數值（value），可透過 `.numpy()`獲得，其回傳該張量對應之 `numpy` 陣列

print("data of tensor is", a.numpy())

`tensorflow` 內建了多種函數，以協助產生具有特別數值結構之張量：

print("tensor with all elements being ones \n",
      tf.ones(shape = (4, 1)).numpy())
print("tensor with all elements being zeros \n",
      tf.zeros(shape = (2, 3)).numpy())
print("identity matrix like tensor \n",
      tf.eye(num_rows = 3, num_columns = 5).numpy())
print("diagonal matrix \n",
      tf.linalg.diag(diagonal = [1, 2, 3, 4]).numpy())

`tensorflow` 亦可指定分配產生隨機的資料

print("tensor with random elements from uniform(0, 1) \n",
      tf.random.uniform(shape = (2, 6),
                        minval = 0, maxval = 1,
                        seed = 48).numpy())
print("tensor with random elements from normal(0, 1) \n",
      tf.random.normal(shape = (3, 2),
                       mean = 0, stddev = 1,
                       seed = 48).numpy())

### 張量之形狀

張量之形狀與形狀之維度數量，可透過張量物件的 `.shape` 與 `ndim` 方法來獲得

print("shape of tensor is", a.shape)
print("ndim of tensor is", a.ndim)

如果要對張量的形狀進行改變的話，可透過 `tf.reshape()` 此函數

print("tensor with shape (4, 3): \n",
      tf.reshape(tensor = a, shape = (4, 3)).numpy())
print("tensor with shape (2, 2, 3): \n",
      tf.reshape(tensor = a, shape = (2, 2, 3)).numpy())
print("tensor with shape (12, 1): \n",
      tf.reshape(tensor = a, shape = (12, 1)).numpy())
print("tensor with shape (12, 1) by (-1, 1): \n",
      tf.reshape(tensor = a, shape = (-1, 1)).numpy())
print("tensor with shape (12,): \n",
      tf.reshape(tensor = a, shape = (12,)).numpy())


注意，`(12, 1)` 與 `(12,)` 兩種形狀是不一樣的，前者為2d的張量，後者為1d的張量。在進行張量操弄時，若將兩者混淆，很可能會帶來錯誤的計算結果。


### 張量之資料類型
張量的資料類型，可透過 `.dtype` 方法獲得

print("data type of tensor is", a.dtype)

若是要調整資料類型的話，則可透過 `tf.cast()`此函數

tf.cast(x = a, dtype = tf.float32)

`tensorflow` 內建多種資料類型，包含整數類型（如 `tf.int32` 與 `tf.int64`）與浮點數類型（如 `tf.float32` 與 `tf.float64`），完整的資料類型請見 [tf.dtypes.DType](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)。

在進行張量的數學運算時，請務必確認張量間的資料類型都是一致的，而 `tensorflow` 常用之資料類型為 `tf.float32` 與 `tf.float64`，前者所需的記憶體較小，但運算結果的數值誤差較大。


## 張量之操弄

### 張量之切片

若要擷取一張量特定的行（row）或列（column）的話，則可透過切片（slicing）的功能獲得。`tensorflow` 張量的切片方式，與 `numpy` 類似，皆使用中括號 `[]`，再搭配所欲擷取資料行列的索引（index）獲得。

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
多個張量在維度可對應之前提下，可透過 `tf.concat` 串接

print("vertical concatenation \n",
      tf.concat([a, a], axis = 0).numpy())
print("horizontal concatenation \n",
      tf.concat([a, a], axis = 1).numpy())

## 張量之運算
考慮以下 `a` 與 `b` 兩張量

a = tf.constant(value = [[1, 2], [3, 4], [5, 6]],
                dtype = tf.float64)
b = tf.constant(value = [[1, 2], [1, 2], [1, 2]],
                dtype = tf.float64)
print("tensor a is \n", a.numpy())
print("tensor b is \n", b.numpy())

我們將使用 `a` 與 `b` 來展示如何使用 `tensorflow` 進行張量間的計算。 

### 張量元素對元素之運算
透過 `tensorflow` 的數學函數，可進行張量元素對元素的四則運算

print("element-wise add \n",
      tf.add(a, b))
print("element-wise subtract \n",
      tf.subtract(a, b))
print("element-wise multiply \n",
      tf.multiply(a, b))
print("element-wise divide \n",
      tf.divide(a, b))

前述採用的函數，皆可取代為其所對應之運算子計算

print("element-wise add \n", a + b)
print("element-wise subtract \n", a - b)
print("element-wise multiply \n", a * b)
print("element-wise divide \n", a / b)

若需要進行絕對值、對數、指數等較為進階之數學運算，可以至 [tf.math](https://www.tensorflow.org/api_docs/python/tf/math) 此模組中尋找對應的數學函數。

### 張量線性代數之運算
除了簡單的四則運算外，當張量的 `ndim` 為2時，`tensorflow` 提供了進行線性代數（linear algebra）相關的函數，如

+ 矩陣轉置（matrix transpose）

a_transpose = tf.transpose(a)
print("transpose of a is \n",
      a_transpose.numpy())

+ 矩陣乘法（matrix multiplication）

# equivalent to tf.linalg.matmul(a, a_transpose)
c = a_transpose @ a
print("c = a_transpose @ a is \n",
      c.numpy())

+ 反矩陣（matrix inverse）

c_inv = tf.linalg.inv(input = c)
print("inverse of c is \n",
      c_inv.numpy()) # c @ c_inv should be identity matrix
print("check for inverse (left) \n",
      (c_inv @ c).numpy())
print("check for inverse (right) \n",
      (c @ c_inv).numpy())

+ Cholesky 拆解（Cholesky decomposition）

c_chol = tf.linalg.cholesky(input = c)
print("Cholesky factor of c is \n",
      c_chol.numpy())
print("check for Cholesky decomposition \n",
      (c_chol @ tf.transpose(c_chol)).numpy())


+ 特徵拆解（eigen-decomposition）

e, v = tf.linalg.eigh(tensor = c)
print("eigenvalue of c is \n",
      e.numpy())
print("eigenvector of c is \n",
      v.numpy())
print("check for eigen-decomposition \n",
      (v @ tf.linalg.diag(diagonal = e) @
       tf.transpose(v)).numpy())

+ 奇異值拆解（singular value decomposition）

s, u, v = tf.linalg.svd(tensor = a)
print("singular value of a is \n",
      s.numpy())
print("left singular vector of a is \n",
      u.numpy())
print("right singular vector of a is \n",
      v.numpy())
print("check for singular value decomposition \n",
      (u @ tf.linalg.diag(diagonal = s) @
       tf.transpose(v)).numpy())

### 對張量之數值進行摘要
`tf.math` 提供了一些化約（reduce）的函數，對張量內的數值進行摘要

print("calculate mean \n",
      tf.math.reduce_mean(input_tensor = a).numpy())
print("calculate standard deviation \n",
      tf.math.reduce_std(input_tensor = a).numpy())
print("calculate max \n",
      tf.math.reduce_max(input_tensor = a).numpy())
print("calculate min \n",
      tf.math.reduce_min(input_tensor = a).numpy())

我們亦可對張量的各面向，進行前述的摘要。以平均數為例：

print("calculate mean for each column \n",
      tf.math.reduce_mean(
          input_tensor = a, axis = 0).numpy())
print("calculate mean for each row \n",
      tf.math.reduce_mean(
          input_tensor = a, axis = 1).numpy())

## 實徵範例

### 產生線性迴歸資料

# define a function to generate x and y
def generate_data(n_sample, coef, intercept = 0, sd_residual = 1,
                  mean_feature = 0, sd_feature = 1,
                  dtype = tf.float64, seed = None):
    coef = tf.constant(coef, dtype = dtype)
    n_feature = coef.shape[0]
    x = tf.random.normal(shape = (n_sample, n_feature),
                         mean = mean_feature,
                         stddev = sd_feature,
                         seed = seed, dtype = dtype)
    e = tf.random.normal(shape = (n_sample, 1),
                         mean = 0,
                         stddev = sd_residual,
                         seed = seed, dtype = dtype)
    coef = tf.reshape(coef, shape = (-1, 1))
    y = intercept + x @ coef + e
    return x, y

# run generate_data
x, y = generate_data(
    n_sample = 10, coef = [-5, 3, 0],
    intercept = 5, sd_residual = 1,
    mean_feature = 10, sd_feature = 3,
    dtype = tf.float64, seed = 48)
print("feature matrix x is \n", x.numpy())
print("response vector y is \n", y.numpy())


### 計算模型參數

# define a function to calculate model parameter
def calculate_parameter(x, y, dtype = tf.float64):
    if x.dtype is not dtype:
        x = tf.cast(x, dtype = dtype)
    if y.dtype is not dtype:
        y = tf.cast(y, dtype = dtype)
    u = tf.ones(shape = (x.shape[0], 1), dtype = dtype)
    x_design = tf.concat([u, x], axis = 1)
    parameter = tf.linalg.inv(tf.transpose(x_design) @ x_design) @ \
                tf.transpose(x_design) @ y
    intercept = parameter[0, 0]
    coef = parameter[1:, 0]
    return intercept, coef

# run calculate_parameter
x, y = generate_data(
    n_sample = 1000, coef = [-5, 3, 0],
    intercept = 5, sd_residual = 1,
    mean_feature = 10, sd_feature = 3,
    dtype = tf.float64, seed = 48)
intercept, coef = calculate_parameter(x, y)
print("intercept estimate is \n", intercept.numpy())
print("coefficient estimate is \n", coef.numpy())

### 建立一進行迴歸分析之物件

# define a class to fit linear regression
class LinearRegression():
    def __init__(self, dtype = tf.float64):
        self.dtype = dtype
        self.intercept = None
        self.coef = None
    def fit(self, x, y):
        if x.dtype is not self.dtype:
            x = tf.cast(x, dtype = self.dtype)
        if y.dtype is not self.dtype:
            y = tf.cast(y, dtype = self.dtype)
        u = tf.ones(shape = (x.shape[0], 1), dtype = self.dtype)
        x_design = tf.concat([u, x], axis = 1)
        parameter = tf.linalg.inv(tf.transpose(x_design) @ x_design) @ \
                tf.transpose(x_design) @ y
        self.intercept = parameter[0, 0]
        self.coef = parameter[1:, 0]
        return self

linear_regression = LinearRegression()
linear_regression.fit(x, y)
print(linear_regression.intercept.numpy())
print(linear_regression.coef.numpy())