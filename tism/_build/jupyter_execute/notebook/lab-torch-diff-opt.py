
Lab: 數值微分與優化
================

在此 lab 中，我們將介紹

1. 如何使用 `torch` 進行數值微分。

2. 如何使用 `torch` 進行數值優化。

3. 利用前述知識，撰寫一採用梯度下降（gradient descent）獲得迴歸參數估計之類型。

import torch

## 數值微分

### 可獲得梯度張量之輸入

而在統計模型中，模型之參數常需透過一優化（optimization）方法獲得，而許多的優化方法皆仰賴目標函數（objective function）的一階導數（first-order derivative），或稱梯度（gradient），因此，如何獲得目標函數對於模型參數的梯度，即為一重要的工作。

在 `torch` 中，張量不僅用於儲存資料，其亦用於儲存模型之參數。然而，誠如先前所述，我們很可能會需要用到對應於該參數之梯度訊息，因此，為了追朔該參數的歷史建立計算圖（computation graph），輸入該參數張量時需要加入 `requires_grad=True` 此指令：

x = torch.tensor([1, 2, 3],
                 dtype = torch.float,
                 requires_grad=True)
print(x)

這裏，我們建立了一尺寸為 $3$ 的張量，由於此張量具有 `requires_grad=True` 此標記，因此，接下來對此張量進行任何的運算，`torch` 皆會將此計算過程記錄下來。舉例來說：

y = 2 * x - 4
z = (y ** 2).sum()
print("tensor y: \n", y)
print("tensor z: \n", z)

我們可以看到，無論是 `y` 或是 `z`，其都具有 `requires_grad=True` 的標記。要特別注意的是，`requires_grad=True` 僅適用於資料類型為浮點數之張量。


### 數值微分之執行
針對已追朔之運算過程，想要獲得與該運算有關的梯度時，可以使用 `.backward()`此方法。在前一小節的例子中，$z = \sum_{i=1}^3 (2x_{i} - 4)^2$，若想要獲得 $\frac{\partial z}{\partial x}$ 在當下 $x$ 的數值的話，可使用以下的程式碼：

z.backward()
print("dz/dx: ", x.grad)


接著，由於 $z$ 也可以寫為 $z = \sum_{i=1}^3 y_{i}^2$，因此，我們是否也可以透過類似的程式碼獲得 $\frac{\partial z}{\partial y}$ 呢？

print("dz/dy: ", y.grad)

結果是不行，主因在於，`torch` 為了節省記憶體的使用，因此，僅可提供位於計算圖葉子（leaf）張量之一次微分。如果希望能夠獲得 $\frac{\partial z}{\partial y}$ 的話，可以對 `y` 使用 `.retain_grad()` 此方法：

y = 2 * x - 4
z = (y ** 2).sum()
y.retain_grad()
z.backward()
print("dz/dy: ", y.grad)

在評估完 $\frac{\partial z}{\partial y}$ 後，讓我們重新檢視一下 `x.grad` 的數值：

print("dz/dx: ", x.grad)

我們會發現，這時的 `x.grad` 數值，變成了原先的兩倍，其背後的原因在於，`.backward()`此方法，會持續地將計算結果累積在變數所對應之 `.grad` 當中。若想要避免持續累積，可以使用 `.grad.zero_()` 方法將 `.grad` 中的數值歸零：

x.grad.zero_()
y.grad.zero_()
print("dz/dx: ", x.grad)
print("dz/dx: ", y.grad)

接著，就可以使用原先的程式碼，計算 $\frac{\partial z}{\partial x}$ 與 $\frac{\partial z}{\partial y}$：

y = 2 * x - 4
z = (y ** 2).sum()
y.retain_grad()
z.backward()
print("dz/dx: ", x.grad)
print("dz/dy: ", y.grad)


不過要特別注意的是，如果計算圖沒有重新建立，連續進行兩次 `.backward()` 會引發錯誤的訊息。

### 可獲得梯度張量之進階控制

一個張量是否有被追朔以計算梯度，除了直接列印外，亦可透過 `.requires_grad` 此屬性來觀看

x = torch.tensor([1, 2, 3],
                 dtype = torch.float)
print(x.requires_grad)

如果想將一原先沒有要求梯度之張量，改為需要梯度時，可以使用 `.requires_grad_()` 此方法原地修改該向量的 `requires_grad` 類型：

x.requires_grad_(True)
print(x.requires_grad)

如果想將張量 `x` 拷貝到另一變量 `x_ng`，卻不希望 `x_ng` 的計算會被追朔時，可以使用以下的程式碼：

x_ng = x.detach()
print(x_ng.requires_grad)

最後，如果希望可獲得梯度之向量後續的計算歷程不被追朔的話，可以將計算程式碼置於 `with torch.no_grad():` 此環境中，即

with torch.no_grad():
    y = 2 * x - 4
    z = (y ** 2).sum()
print(y.requires_grad)
print(z.requires_grad)

## 數值優化

### 手動撰寫優化算則

前一小節所使用的範例，其計算過程可以寫為

$$
\begin{aligned}
z &= f(x)\\
 &= \sum_{i=1}^3\left[2(x_i-2)\right]^2 \\
 &= \sum_{i=1}^3 y^2
\end{aligned}
$$

若想要找到一 $\widehat{x}$，其使得 $f(\widehat{x})$ 達到最小值的話，由於 $z$ 為 $y_1, y_2, y_3$ 的平方和，因此，其會在 $\widehat{y} = (0, 0, 0)$的地方達到最小值，也意味著 $\widehat{x} = (2,2,2)$。

那麼，我們應該如何使用數值方法，對 $f(x)$ 進行優化呢？根據梯度下降（gradient descent）法，$\widehat{x}$ 的更新規則為

$$
\widehat{x} \leftarrow \widehat{x} - \alpha \times \frac{\partial f(x)}{\partial x}
$$
這裏，$\alpha$ 表示一步伐大小（step size），或稱學習速率（learning rate）。一般來說，當 $f$ 足夠圓滑（smooth），且 $\alpha$ 的數值大小適切時，梯度下降法能夠找到一臨界點（critical points），其可能為 $f$ 最小值的發生位置。

在開始進行梯度下降前，我們先定義一函數 `f` 使得`z = f(x)`，並了解起始狀態時，`f(x)` 與 `x` 的數值為何：

def f(x):
    z = ((2 * x - 4) ** 2).sum()
    return z

x = torch.tensor([1, 2, 3],
                 dtype = torch.float,
                 requires_grad=True)
z = f(x)
print("f(x) = {:2.3f}, x = {}".format(z.item(), x.data))

使用 `torch` 進行梯度下降，需先計算在當下 `x` 數值下的梯度，接著，根據該梯度的訊息與設定的步伐大小對 `x` 進行更新，即

lr = .1
z.backward()
with torch.no_grad():
    x.sub_(lr * x.grad)
    x.grad.zero_()
z = f(x)
print("f(x) = {:2.3f}, x = {}".format(z.item(), x.data))

這裡，我們將學習速率 `lr` 設為0.1，而張量的 `.sub_()` 方法則是就地減去括號內的數值直接更新。透過 `f(x)` 的數值，可觀察到梯度下降的確導致 `z` 數值的下降，而 `x` 也與 $\widehat{x}=(2,2,2)$ 更加地靠近。

梯度下降的算則，需重複前述的程序多次，才可獲得一收斂的解。最簡單的方法，即使用 `for` 迴圈，重複更新$I$次：

x = torch.tensor([1, 2, 3],
                 dtype = torch.float,
                 requires_grad=True)
z = f(x)
for i in range(1, 21):
    z.backward()
    with torch.no_grad():
        x.sub_(lr * x.grad)
    z = f(x)
    print("iter {:2.0f}, f(x) = {:2.3f}, x = {}".format(i, z.item(), x.data))
    x.grad.zero_()

然而，從列印出來的結果來看，20次迭代可能太多了，因此，我們可以進一步要求當梯度絕對值小於某收斂標準 `tol` 時，算則就停止，其所對應之程式碼為：

tol = 1e-5
x = torch.tensor([1, 2, 3],
                 dtype = torch.float,
                 requires_grad=True)
z = f(x)
for i in range(1, 21):
    z.backward()
    with torch.no_grad():
        x.sub_(lr * x.grad)
    z = f(x)
    print("iter {:2.0f}, z = {:2.3f}, x = {}".format(i, z.item(), x.data))
    if (x.grad.abs().max().item() < tol):
        break
    x.grad.zero_()

### 使用`torch.optim`進行優化

由於 `torch` 已內建了進行優化的函數，因此，在多數的情況下，可直接利用 `torch.optim`

x = torch.tensor([1, 2, 3],
                 dtype = torch.float,
                 requires_grad=True)
opt = torch.optim.SGD((x,), lr=.1)
z = f(x)
for i in range(1, 21):
    z.backward()
    opt.step()
    z = f(x)
    print("iter {:2.0f}, z = {:2.3f}, x = {}".format(i, z.item(), x.data))
    if (x.grad.abs().max().item() < tol):
        break
    opt.zero_grad()