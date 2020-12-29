
Lab: 最大概似估計法
================
此 lab 中，我們將會透過 `tensorflow-probability`（TFP）此套件，學習以下的主題。

1. 認識 TFP 的分配（distribution）物件。

2. 利用變項之分配，搭配自動微分獲與優化器獲得最大概似估計值。


TFP 之安裝與基礎教學，可參考 [TFP官方網頁](https://www.tensorflow.org/probability/install)。在安裝完成後，可透過以下的指令載入其與 `tensorflow`

import tensorflow_probability as tfp
import tensorflow as tf

## 分配物件
TFP 最為核心的物件為分配物件，其用於表徵一機率分配。TFP 涵蓋了許多不同的機率分配，其名單可至 [官方網頁](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions) 查看。

在實務上，我們常將 TFP 的分配模組儲存為 `tfd`，以便於使用，即：

tfd = tfp.distributions

### 分配物件之基本操作

以常態分配為例，我們可透過以下程式碼產生一表徵常態分配之物件，其平均數為0，標準差為1（尾端的小數點表示浮點數，而非整數）：

normal = tfd.Normal(loc=0., scale=1.)

透過此分配物件，我們可以產生對應之隨機樣本

print("random sample with shape ():\n",
      normal.sample())
print("random sample with shape (3,):\n",
      normal.sample(sample_shape=3))
print("random sample with shape (2,3):\n",
      normal.sample(sample_shape=(2, 3)))

我們亦可給定實現值來評估其在該分配下之累積機率值：

print("cumulative probability given value with shape ():\n",
      normal.cdf(value=0), "\n")
print("cumulative probability given value with (3,):\n",
      normal.cdf(value=[-1, 0, .5]), "\n")
print("cumulative probability given value with (2,3):\n",
      normal.cdf(value=[[-1, 0, .5], [-2, 1, 3]]))


或是對數機率值：

print("log-probability given value with shape ():\n",
      normal.log_prob(value=0), "\n")
print("log-probability given value with (3,):\n",
      normal.log_prob(value=[-1, 0, .5]), "\n")
print("log-probability given value with (2,3):\n",
      normal.log_prob(value=[[-1, 0, .5], [-2, 1, 3]]))

### 分配物件之形狀
分配物件的形狀比起張量的形狀較為複雜些，其產生的樣本共牽涉到三種形狀：

1. 樣本形狀（sample shape），為在使用分配物件產生樣本時之形狀（即為前一小節使用 `.sample()` 時所設定的 `sample_shape`），其產生的資料彼此間為獨立且相同分配的（independent and identically distributed）。

2. 批次形狀（batch shape），為建立分配物件時所設定的形狀（其透過參數的形狀決定），其可用於產生批次的樣本，批次樣本間彼此獨立，但其邊際分配之參數可以不同。

3. 事件形狀（event shape），即多變量分配之變數形狀，如 $P$ 維多元常態分配的形狀即為 `(P,)`，在同一事件下產生的資料其變數間可為相依，且各邊際分配之參數也未必相同。

而分配物件的形狀則牽涉到批次與事件兩種，可透過直接列印分配物件查看

print(normal)

或是使用 `.batch_shape` 與 `.event_shape` 獲得：

print(normal.batch_shape)
print(normal.event_shape)

由於之前所創立的 `normal` 其用於產生純量之常態分配隨機變數，故 `.batch_shape` 與 `.event_shape` 兩者皆為 `()`。

批次形狀之設定，乃經由對分配參數形狀之推論獲得，如

normal_batch = tfd.Normal(loc=[0., 1.], scale=[1., .5])
print(normal_batch)

前述分配的 `batch_shape` 為 `(2,)`，其可用於產生一組兩個來自常態分配之變數，其中一個平均數為0，變異數為1，另一個平均數為1，變異數為.5，如

print("random sample with sample_shape ():\n",
      normal_batch.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      normal_batch.sample(sample_shape=3), "\n")
print("random sample with sample_shape (2,3):\n",
      normal_batch.sample(sample_shape=(2,3)))

我們亦可使用所創立的 `normal_batch` 來度量輸入數值的對數機率值：

print("log-probability given value with shape ():\n",
      normal_batch.log_prob(0), "\n")
print("log-probability given value with shape (2,):\n",
      normal_batch.log_prob([0, 0]), "\n")
print("log-probability given value with shape (2,1):\n",
      normal_batch.log_prob([[0], [0]]))

這邊我們可以觀察到，前兩者種寫法獲得一樣的數值，皆表示 `[0, 0]` 此向量於 `normal_batch` 下的對數機率。對第一種寫法來說，其輸入為純量，因此，使用到了廣播（broadcasting）的概念，將0的數值轉為 `[0,0]` 後再進行評估，第二種寫法則是較為標準，其直接輸入了 `[0,0]`，與 `normal_batch` 的 `batch_size` 相同。而第三種寫法，可以想成輸入了一 `sample_shape` 為2的資料，而每筆觀測值的0皆會透過廣播拓展為 `[0, 0]`，故回傳了每筆觀測值於 `normal_batch` 之對數機率。


事件形狀僅適用於多變量之分配，以多元常態（multivariate normal）分配為例，其需給定一平均數向量與共變異數矩陣（在這邊，我們採用的 `tfd.MultivariateNormalTriL` 需給定的是共變異數矩陣的 Cholesky 分解）：

mvn = tfd.MultivariateNormalTriL(
    loc=[0, 1],
    scale_tril=tf.linalg.cholesky([[1., 0.], [0., .5]]))
print(mvn)

我們可看到此分配物件的 `event_shape` 是 `(2)`，與此多元常態分配的維度相同，其可用於產生服從多元常態分配之資料

print("random sample with sample_shape ():\n",
      mvn.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      mvn.sample(sample_shape=3), "\n")
print("random sample with sample_shape (2, 3):\n",
      mvn.sample(sample_shape=(2, 3)))

同樣的，該物件亦可用於評估給定數值下的對數機率：

print("log-probability given value with shape (2,):\n",
      mvn.log_prob([0, 0]), "\n")
print("log-probability given value with shape (2,1):\n",
      mvn.log_prob([[0, 0], [0, 0]]))

儘管此 `mvn` 表徵的二維之多元常態分配，其平均數與共變異數矩陣之設定，與先前 `normal_batch`是等價的，但在使用 `mvn` 評估機率時，需注意：（1）先前針對 `batch_shape` 此面向的廣播，不再適用於 `event_shape`，故 `mvn.log_prob(0)` 會產生錯誤訊息；（2）針對每筆觀測值，其計算的是在此多元常態分配下的聯合機率，因此，只會獲得一個對數機率值。


分配物件的 `batch_shape` 可透過 `tfd.Independent` 此物件轉為 `event_shape`，如

tfd.Independent(normal_batch, reinterpreted_batch_ndims=1)

在多變量的分配之下，前述介紹的批次形狀與事件形狀，可以合併使用：

mvn_batch = tfd.MultivariateNormalTriL(
    loc=[[0, 1],
         [1, 2],
         [2, 3]],
    scale_tril=tf.linalg.cholesky([[1., 0.], [0., .5]]))
mvn_batch

這裡，其 `batch_shape` 為 `(3)`，值得注意的是，這邊我們僅設定了一個共變異數矩陣，因此，其會透過廣播的機制，與三個平均數向量做對應。同樣的，我們可以用此 `mvn_batch` 來產生樣本資料


print("random sample with sample_shape ():\n",
      mvn_batch.sample(), "\n")
print("random sample with sample_shape (3,):\n",
      mvn_batch.sample(sample_shape=3), "\n")
print("random sample with sample_shape (2, 3):\n",
      mvn_batch.sample(sample_shape=(2, 3)))

## 最大概似估計法



### 可學習的分配與求解
使用 `tensorflow` 來進行最大概似法，有許多種做法，其中，最重要的關鍵就在於如何建構概似函數。事實上，在前一小節中，我們已經可以計算在給定參數下，某個隨機樣本實現值之可能性，因此，關鍵就在於如何讓前述之可能性，轉為參數數值之函數，而最簡單的做法，就是將參數設為 `tf.Variable`，搭配自動微分與優化器對其進行更新。

以下的程式碼建立了一 `batch_size` 為2的常態分配模型，我們可透過 `.trainable_varialbes` 來查看哪些變數是可以透過訓練更新其數值的。

batch_size = 2
normal_model = tfd.Normal(
        loc=tf.Variable(tf.zeros(batch_size), name='loc'),
        scale=tf.Variable(tf.ones(batch_size), name='scale'))
print("normal model:\n", normal_model, "\n")
print("parameters in normal model:\n", normal_model.trainable_variables)

TFP 的官方教學中，將前述的分配稱作可學習的分配（learnable distribution）。接著，我們在目前給定的參數數值下，產生一隨機樣本，此樣本在目前參數數值下的可能性，可簡單地使用 `.log_prob()` 方法與加總平均的計算獲得：

sample_size = 1000
x = normal_model.sample(sample_shape=sample_size)
loss_value = -tf.reduce_mean(tf.reduce_sum(normal_model.log_prob(x), axis = 1))
print("negative likelihood value is ", loss_value.numpy())

最後，我們就可以透過優化器來求最大概似解了。在這邊需特別注意的是，由於 `loc` 與 `scale` 原本的數值為真實參數的數值，為了要展示優化器的正確運作，我們將其起始值設為一個較差的數值。

epochs = 400
tol = 10**(-5)
learning_rate = 1.0
normal_model.loc.assign([1., 1.])
normal_model.scale.assign([.5, .5])
optimizer = tf.optimizers.Adam(learning_rate)
for epoch in tf.range(epochs):
    with tf.GradientTape() as tape:
        loss_value = -tf.reduce_mean(tf.reduce_sum(normal_model.log_prob(x), axis = 1))
    gradients = tape.gradient(loss_value,
                              normal_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,
                                  normal_model.trainable_variables))
    if (tf.reduce_max(
            [tf.reduce_mean(
                tf.math.abs(x)) for x in gradients]).numpy()) < tol:
        print("{n} Optimizer Converges After {i} Iterations".format(
            n=optimizer.__class__.__name__, i=epoch))
        break

接著，我們比較優化器求得的解，以及分析解之間的差異（常態分配平均數與標準差的分析解即為樣本平均數與除上 $N$ 的標準差）

print("ML mean estimate: \n", 
      normal_model.loc.numpy())
print("ML standard deviation estimate: \n", 
      normal_model.scale.numpy())

print("sample mean: \n", 
      tf.reduce_mean(x, axis = 0).numpy())
print("sample standard deviation: \n", 
      tfp.stats.stddev(x, sample_axis = 0).numpy())

我們可看到兩組解的數值幾乎相同，顯示優化器在這邊有確實地找到最大概似解。

### 利用對射進行變數轉換
前述的最大概似估計過程，並未對於參數估計之數值進行限制，其考慮的是非限制的優化問題（unconstrained optimization problem），然而，在實際進行優化時，若未對於參數數值進行限制的話，可能會獲得不合理之估計值（如負的變異數等）。

在 TFP 的架構中，主要是透過對於參數進行對射（bijection），將原始受限制的參數轉為非限制的參數後進行估計。TFP 所內建的對射函數，可參考其 [官方網頁](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors)。

在下面的範例中，我們利用 `tfp.util.TransformedVariable` 與 `tfb.Exp()` ，將常態分配的變異數 $\sigma$ 參數化為 $\exp(\gamma)$，將原本需進行估計 $\mu$ 與 $\sigma$ 的優化問題，轉為估計 $\mu$ 與 $\gamma$，此時，我們不需要對 $\gamma$ 的數值範圍進行限制，其在透過 $\exp$ 函數的轉換後，會自動符合模型隱含的限制式。

tfb = tfp.bijectors
batch_size = 2
normal_model_tr = tfd.Normal(
    loc=tf.Variable(tf.zeros(batch_size), name='loc'),
    scale=tfp.util.TransformedVariable(
        tf.ones(batch_size),
        bijector=tfb.Exp(), name="scale"))
print("normal model:\n", normal_model_tr, "\n")
print("parameters in normal model:\n", normal_model_tr.trainable_variables)

我們可以直接使用前述的優化程式碼來獲得重新參數化後的參數估計

epochs = 400
tol = 10**(-5)
learning_rate = 1.0
normal_model_tr.loc.assign([1., 1.])
normal_model_tr.scale.assign([.5, .5])
optimizer = tf.optimizers.Adam(learning_rate)
for epoch in tf.range(epochs):
    with tf.GradientTape() as tape:
        loss_value = -tf.reduce_mean(
            tf.reduce_sum(
                normal_model_tr.log_prob(x), axis = 1))
    gradients = tape.gradient(loss_value,
                              normal_model_tr.trainable_variables)
    optimizer.apply_gradients(zip(gradients,
                                  normal_model_tr.trainable_variables))
    if (tf.reduce_max(
            [tf.reduce_mean(
                tf.math.abs(x)) for x in gradients]).numpy()) < tol:
        print("{n} Optimizer Converges After {i} Iterations".format(
            n=optimizer.__class__.__name__, i=epoch))
        break

print("ML mean estimate: \n",
      normal_model_tr.loc.numpy())
print("ML standard deviation estimate: \n",
      normal_model_tr.scale.numpy())

### 多元常態分配之參數估計

loc_true = tf.constant([0., 1., -1.])
scale_tril_true = tf.linalg.cholesky(
    tf.constant([[1, .3, .6], [.3, .5, .1], [.6, .1, 1.5]]))
mvn_model_true = tfd.MultivariateNormalTriL(
    loc = loc_true,
    scale_tril = scale_tril_true)
print(mvn_model_true)

#a bug here
x=mvn_model_true.sample([10000,1]).numpy()

optimizer=tf.optimizers.Adam(learning_rate=.5)
epochs=500
tol=10**(-3)

loc = tf.Variable(tf.constant([0., 0., 0.]), name='loc')
scale_tril = tf.Variable(tf.linalg.cholesky(
    tf.constant([[1, .0, .0], [.0, 1, .0], [.0, .0, 1]])),
    name = "scale_tril")
mvn_model = tfd.MultivariateNormalTriL(
    loc=loc, scale_tril=scale_tril)
for epoch in tf.range(epochs):
    with tf.GradientTape() as tape:
        loss_value = -tf.reduce_mean(
            tf.reduce_sum(mvn_model.log_prob(x), axis = 1))
    gradients = tape.gradient(loss_value,
                              mvn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,
                                  mvn_model.trainable_variables))
    if (tf.reduce_max(
            [tf.reduce_max(
                tf.math.abs(x)) for x in gradients]).numpy()) < tol:
        print("{n} Optimizer Converges After {i} Iterations".format(
            n=optimizer.__class__.__name__, i=epoch))
        break

print(mvn_model.mean())
print(mvn_model.covariance())

