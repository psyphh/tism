
Lab: 數值微分與優化
================

在此 lab 中，我們將介紹

1. 如何使用 `tensorflow` 進行數值微分。

2. 如何使用 `tensorflow` 進行數值優化。

3. 利用前述知識，撰寫一採用梯度下降（gradient descent）獲得迴歸參數估計之類型。

import tensorflow as tf

## 數值微分

而在統計模型中，模型之參數常需透過一優化（optimization）方法獲得，而許多的優化方法皆仰賴目標函數（objective function）的一階導數（first-order derivative），或稱梯度（gradient），因此，如何獲得目標函數對於模型參數的梯度，即為一重要的工作。

### 變量

在 `tensorflow` 中，我們所欲進行微分之變量（variable）乃透過 `tf.Variable` 此類型來表徵，其可透過 `tf.Variable` 此建構式來建立：

x = tf.constant([1, 1, 1], dtype = tf.float64)
x = tf.Variable(x)
print(x)

這裏，我們建立了一尺寸為 $(3,)$ 的變量。乍看之下，變量與張量非常類似，兩者皆牽涉到資料、形狀、以及類型等面向，事實上，變量背後的資料結構的確是一張量，當我們進行運算時，使用的都是該張量資料，如

print("element-wise addition:", x + x)
print("element-wise multiplication:", x * x)

然而，變量容許我們在程式執行的過程中，不斷地對其狀態進行更新。比如說，我們可以使用 `.assign` 此方法對一變量的張量資料進行更新，其會重新使用儲存該張量的記憶體：

x.assign([1, 2, 3,])
print(x)

在 `tensorflow` 中，變量最重要的功能就是用來表徵模型的參數，其可透過不同的優化策略來更新其數值，因此，變量通常是可以被訓練的（trainable），我們可透過其 `.trainbale` 屬性來了解

print(x.trainable)


### 數值微分之基礎
在 `tensorflow` 中，自動微分乃透過 `tf.GradientTape` 來進行。首先，我們利用 `tf.GradientTape()` 所建立的環境，紀錄該變量的計算過程

with tf.GradientTape() as tape:
    y = 2 * x - 4
    z = tf.reduce_sum(y ** 2)
print("tensor y: \n", y)
print("tensor z: \n", z)

在此例子中，整個計算過程可寫為 $y_i = 2(x_{i} - 2)$，$z = \sum_{i=1}^3 y_i^2$。針對已紀錄之運算過程，想要獲得與該運算有關的梯度時，可以使用 `tf.GradientTape.gradient`此函數：

grad_x = tape.gradient(z, x)
print("dz/dx: ", grad_x)

這裡， `grad_x` 即為 $\frac{\partial z}{\partial x}$ 之計算結果，其張量尺寸與 `x` 相同，即 `(3,1)`。

需要特別注意的是，在執行完一次 `tf.GradientTape.gradient` 後，`tensorflow` 即會將該記錄器所使用的資源給釋放出來，故無法再次使用 `tf.GradientTape.gradient` 此指令。若想要多次執行 `tf.GradientTape.gradient`，可以在建立 `GradientTape` 時，使用 `persistent=True` 此指令：

with tf.GradientTape(persistent=True) as tape:
    y = 2 * x - 4
    z = tf.reduce_sum(y ** 2)

透過該指令，我們就能夠計算多個梯度的結果，如計算 $\frac{\partial z}{\partial x}$ 與 $\frac{\partial z}{\partial y}$ 兩者。

grad_x = tape.gradient(z, x)
grad_y = tape.gradient(z, y)
print("dz/dx: ", grad_x)
print("dz/dy: ", grad_y)

我們也可以在單一的 `gradient` 指令下同時計算 `x` 與 `y` 的梯度

grad = tape.gradient(z, {"x":x, "y":y})
for name, value in grad.items():
    print("dz/d" + name + ": ", value)


### 二階微分矩陣
`tensorflow` 也可以用於計算二次微分矩陣，即黑塞矩陣。其過程需使用到兩組 `GradeientTape`，其中一組用來計算一階導數，另外一組用來紀錄計算一階導數的過程，並計算二階導數，如以下範例

with tf.GradientTape() as tape2:
  with tf.GradientTape() as tape1:
    y = 2 * x - 4
    z = tf.reduce_sum(y ** 2)
  grad_x = tape1.jacobian(z, x)
hess = tape2.jacobian(grad_x, x)
print("d^2z/dx^2: ", hess.numpy())

然而，在此需要特別小心注意張量的尺寸會如何影響計算之過程。

## 數值優化

### 手動撰寫優化算則

前一小節所使用的範例，其計算過程可以寫為

$$
\begin{aligned}
z &= f(x)\\
 &= \sum_{i=1}^3\left[2(x_i-2)\right]^2 \\
 &= \sum_{i=1}^3 y_i^2
\end{aligned}
$$

若想要找到一 $\widehat{x}$，其使得 $f(\widehat{x})$ 達到最小值的話，由於 $z$ 為 $y_1, y_2, y_3$ 的平方和，因此，其會在 $\widehat{y} = (0, 0, 0)$的地方達到最小值，也意味著 $\widehat{x} = (2,2,2)$。

那麼，我們應該如何使用數值方法，對目標函數進行優化呢？令 $\theta$ 表示模型參數（其扮演範例中$x$的角色），$\mathcal{D}(\theta)$ 表示度量模型好壞的目標函數（其扮演$f(x)$的角色）。根據梯度下降（gradient descent）法，極小元（minimizer）$\widehat{\theta}$ 的更新規則為

$$
\widehat{\theta} \leftarrow \widehat{\theta} - s \times \frac{\partial \mathcal{D}(\widehat{\theta})}{\partial \theta}
$$

這裏，$s$ 表示一步伐大小（step size），或稱學習速率（learning rate）。一般來說，當 $\mathcal{D}$ 足夠圓滑（smooth），且 $s$ 的數值大小適切時，梯度下降法能夠找到一臨界點（critical points），其可能為 $\mathcal{D}$ 最小值的發生位置。

在開始進行梯度下降前，我們先定義一函數 `f` 使得 `z = f(x)`，並了解起始狀態時，`f(x)` 與 `x` 的數值為何：

def f(x):
    z = tf.reduce_sum((2 * x - 4) ** 2)
    return z

x = tf.Variable([1, 2, 3],
                dtype = tf.float64)
z = f(x)
print("f(x) = {:2.3f}, x = {}".format(z.numpy(), x.numpy()))

使用 `tensorflow` 進行梯度下降，需先計算在當下 `x` 數值下的梯度，接著，根據該梯度的訊息與設定的步伐大小對 `x` 進行更新，即

lr = .1
with tf.GradientTape() as tape:
    z = f(x)
grad_x = tape.gradient(z, x)
x.assign_sub(lr * grad_x)
print("f(x) = {:2.3f}, x = {}".format(
    z.numpy(), x.numpy()))

這裡，我們將學習速率 `lr` 設為0.1，而張量的 `.assign_sub()` 方法則是就地減去括號內的數值直接更新。透過 `f(x)` 的數值，可觀察到梯度下降的確導致 `z` 數值的下降，而 `x` 也與 $\widehat{x}=(2,2,2)$ 更加地靠近。

梯度下降的算則，需重複前述的程序多次，才可獲得一收斂的解。最簡單的方法，即使用 `for` 迴圈，重複更新$I$次：

x = tf.Variable([1, 2, 3],
                dtype = tf.float64)
for i in range(1, 21):
    with tf.GradientTape() as tape:
        z = f(x)
    grad_x = tape.gradient(z, x)
    x.assign_sub(lr * grad_x)
    print("f(x) = {:2.3f}, x = {}".format(
        z.numpy(), x.numpy()))

然而，從列印出來的結果來看，20次迭代可能太多了，因此，我們可以進一步要求當梯度絕對值小於某收斂標準 `tol` 時，算則就停止，其所對應之程式碼為：


tol = 1e-4
epochs = 20
x = tf.Variable([1, 2, 3],
                dtype = tf.float64)
for i in range(1, 21):
    with tf.GradientTape() as tape:
        z = f(x)
    grad_x = tape.gradient(z, x)
    x.assign_sub(lr * grad_x)
    print("f(x) = {:2.3f}, x = {}".format(
        z.numpy(), x.numpy()))
    if tf.reduce_max(tf.abs(grad_x)) < tol:
        break

### 使用`tf.optimizers.Optimizer`進行優化

由於 `tensorflow` 已內建了進行優化的方法，因此，在絕大多數的情況下，可直接利用 `tf.optimizers.Optimizer` 此類型來求得函數的最小值。

`tf.optimizers.SGD` 為進行梯度下降法之物件，由於 `tensorflow` 主要用於進行深度學習，在該領域種主要使用的是隨機梯度下降（stochastic gradient descent）或是迷你批次梯度下降（mini-batch gradient descent）來強化優化的效能，因此，`tensorflow` 使用 `SGD` 一詞。事實上，除了計算一階導數時資料量的差異外，`SGD` 與傳統的梯度下降並無差異。

`tf.optimizers.SGD` 可透過以下的程式碼來使用：

tol = 1e-4
epochs = 20
x = tf.Variable([1, 2, 3],
                dtype = tf.float64)
opt = tf.optimizers.SGD(learning_rate = lr)
for i in range(1, 21):
    with tf.GradientTape() as tape:
        z = f(x)
    grad_x = tape.gradient(z, x)
    opt.apply_gradients(zip([grad_x], [x]))
    print("f(x) = {:2.3f}, x = {}".format(
        z.numpy(), x.numpy()))
    if tf.reduce_max(tf.abs(grad_x)) < tol:
        break

這裡，我們使用 `tf.optimizers.SGD` 來生成優化器（optimizer）物件 `opt`，其在生成時，需要指定學習速率 `learning_rate`。使用內建優化器時，我們仍須自行計算梯度，並利用 `.apply_gradient` 此方法，給定梯度與變量的列表進行更新。乍看之下，使用優化器並為簡化原有之程式碼，不過，當所欲進行更新的變量很多時，優化器可以避免逐一對變量數值進行更新的麻煩。

`tf.optimizers.SGD` 容許使用者加入動能（momentum）$m$（其預設為 0），此時，優化算則會改為

$$
\begin{aligned}
\delta & \leftarrow m \times  \delta + \frac{\partial \mathcal{D}(\widehat{\theta})}{\partial \theta} \\
\widehat{\theta} &\leftarrow \widehat{\theta} - s \times \delta
\end{aligned}
$$

這裡，$\delta$ 表示更新的方向。此算則中，更新的方向不單單倚賴當下目標函數的梯度，其亦考慮到先前的梯度方向，因此，引入動能會使得求解的路徑更為平滑。

在 `tf.optimizers` 中，有許多不同的優化器（見[`tf.optimizers`頁面](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)），如

+ `Adadelta`（見[ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)）
+ `Adagrad`（見[Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)）
+ `Adam`（見[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)）
+ `RMSprop`（見[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)）

讀者可自行深入了解這些方法。

