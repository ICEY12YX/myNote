# 选修实验 - Softmax 函数 (Softmax Function)
在本次实验中，我们将深入探索 Softmax 函数。在解决多分类 (Multiclass Classification) 问题时，无论是 Softmax 回归 (Softmax Regression) 还是神经网络 (Neural Networks) ，都会广泛使用这一函数。  

&lt;center&gt;  &lt;img  src=&quot;./images/C2_W2_Softmax_Header.PNG&quot; width=&quot;600&quot; /&gt;  &lt;center/&gt;

  


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
%matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

> **注**：通常在本课程中，实验笔记本习惯采用从 0 开始计数、到 N-1 结束的约定，即 $\sum_{i=0}^{N-1}$；而理论课讲义则习惯从 1 开始计数、到 N 结束，即 $\sum_{i=1}^{N}$。这是因为编写代码时循环迭代通常从 0 开始，而在讲义中从 1 计数到 N 能让数学公式更加简洁清晰。由于本笔记本包含的数学公式比一般实验要多，因此我们将打破常规，采用从 1 计数到 N 的方式。

## Softmax 函数
无论是 Softmax 回归，还是带有 Softmax 输出的神经网络，模型都会生成 N 个输出，并从中选出一个输出作为预测的类别。在这两种情况下，都会先由一个线性函数生成一个向量 (Vector) $\mathbf{z}$，然后再将其输入到 Softmax 函数中。如下所述，Softmax 函数会将 $\mathbf{z}$ 转换为一个概率分布 (Probability Distribution) 。经过 Softmax 处理后，每一个输出值都会落在 0 到 1 之间，并且所有输出值加起来正好等于 1，因此它们可以被解释为各个类别的概率。输入值越大，转换后对应的输出概率也就越大。
&lt;center&gt;  &lt;img  src=&quot;./images/C2_W2_SoftmaxReg_NN.png&quot; width=&quot;600&quot; /&gt;  

Softmax 函数可以写成如下形式：
$$a_j = \frac{e^{z_j}}{ \sum_{k=1}^{N}{e^{z_k} }} \tag{1}$$
输出 $\mathbf{a}$ 是一个长度为 N 的向量，因此对于 Softmax 回归，你也可以将其写为：
\begin{align}
\mathbf{a}(x) =
\begin{bmatrix}
P(y = 1 | \mathbf{x}; \mathbf{w},b) \\
\vdots \\
P(y = N | \mathbf{x}; \mathbf{w},b)
\end{bmatrix}
=
\frac{1}{ \sum_{k=1}^{N}{e^{z_k} }}
\begin{bmatrix}
e^{z_1} \\
\vdots \\
e^{z_{N}} \\
\end{bmatrix} \tag{2}
\end{align}


这清楚地表明，输出是一个由概率组成的向量。其中第一项就是在给定输入特征 (Feature) $\mathbf{x}$ 以及参数 (Parameters) $\mathbf{w}$ 和 $\mathbf{b}$ 的条件下，输入样本属于第一个类别的概率。  
下面让我们使用 NumPy 来实现它：


```python
def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)
```

请在下方拖动滑块，改变各个输入 `z` 的数值。


```python
plt.close("all")
plt_softmax(my_softmax)
```

当你在上面改变各个 z 的取值时，有几点值得特别留意：
* Softmax 分子中的指数运算 (Exponential) 会放大数值之间的微小差异
* 所有输出值的总和恒等于 1
* Softmax 横跨了所有的输出单元。例如，仅仅改变 `z0` 的值，就会同时改变从 `a0` 到 `a3` 所有输出的值。不妨将其与 ReLU 或 Sigmoid 等其他激活函数 (Activation Functions) 对比一下——那些激活函数都是单输入、单输出的。

## 代价函数 (Cost)
&lt;center&gt; &lt;img  src=&quot;./images/C2_W2_SoftMaxCost.png&quot; width=&quot;400&quot; /&gt;    &lt;center/&gt;

与 Softmax 配套使用的损失函数 (Loss Function) 是交叉熵损失 (Cross-entropy Loss) ，其表达式为：
\begin{equation}
  L(\mathbf{a},y)=\begin{cases}
    -log(a_1), & \text{if $y=1$}.\\
        &\vdots\\
     -log(a_N), & \text{if $y=N$}
  \end{cases} \tag{3}
\end{equation}

其中，y 是当前样本的真实目标类别，而 $\mathbf{a}$ 则是 Softmax 函数的输出结果。具体来说，$\mathbf{a}$ 中的各个数值都是概率，且总和为 1。
>**回顾：** 在本课程中，“损失 (Loss) ”针对的是单个样本，而“代价 (Cost) ”则涵盖了所有样本。
 
 
请注意，在上面的公式 (3) 中，只有对应于真实目标类别的那一行会对损失产生贡献，其余各行均为零。为了写出整体的代价方程，我们需要引入一个“指示函数 (Indicator Function) ”：当索引与目标类别匹配时，其值为 1；否则为零。
    $$\mathbf{1}\{y == n\} = =\begin{cases}
    1, & \text{if $y==n$}.\\
    0, & \text{otherwise}.
  \end{cases}$$
由此，代价函数可以表示为：
\begin{align}
J(\mathbf{w},b) = -\frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{N}  1\left\{y^{(i)} == j\right\} \log \frac{e^{z^{(i)}_j}}{\sum_{k=1}^N e^{z^{(i)}_k} }\right] \tag{4}
\end{align}

其中，$m$ 是样本数量，$N$ 是输出类别数量。这就是所有样本损失的平均值。


## Tensorflow 实现
本实验将探讨在 Tensorflow 中实现 Softmax 交叉熵损失的两种方法：“直观 (obvious) ”方法与“推荐 (preferred) ”方法。前者最为直观易懂，而后者则在数值计算上更加稳定。

让我们先创建一个用于训练多分类模型的数据集 (Dataset) 。


```python
# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
```

### *直观的*组织方式

在下面的模型实现中，我们将 Softmax 直接作为最后一个全连接层 (Dense Layer) 的激活函数。
而损失函数则在 `compile` 指令中单独指定。

这里使用的损失函数是 `SparseCategoricalCrossentropy`，它对应的就是上面的公式 (3) 。在这个模型中，Softmax 运算发生在最后一层内部，而损失函数接收的则是 Softmax 输出的概率向量。


```python
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
        
```

由于 Softmax 已经集成在输出层中，因此模型的输出直接就是一个由概率组成的向量。


```python
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))
```

### 推荐的实现方式 &lt;img align=&quot;Right&quot; src=&quot;./images/C2_W2_softmax_accurate.png&quot;  style=&quot; width:400px; padding: 10px 20px ; &quot;&gt;
回想一下课堂上所讲的内容：如果在训练过程中将 Softmax 运算与损失函数合并在一起计算，就能获得数值上更加稳定、精确的结果。这正是下面所展示的“推荐”组织方式。


在推荐的组织方式中，最后一层使用线性激活函数 (`linear`) 。出于历史原因，这种未经归一化的原始输出形式通常被称为 *logits*。与此同时，我们在损失函数中额外传入一个参数：`from_logits = True`。这个参数会告知损失函数：应当把 Softmax 运算也包含在损失计算的过程之中，从而让底层框架能够采用经过数学优化的实现方式。


```python
preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)
        
```

#### 输出结果的处理
请注意，在推荐的模型结构中，模型的直接输出不再是概率，它们的取值范围可以从很大的负数一直到很大的正数。因此，当你在进行预测并期望得到概率值时，必须手动将模型的输出送入 Softmax 函数中进行转换。
让我们来看看推荐模型的原始输出结果：


```python
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))
```

可以看到，这些预测输出并不是概率！
如果你需要得到概率形式的输出，就应该使用 softmax 函数对输出结果做进一步处理。


```python
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))
```

不过，如果只是为了挑选出最有可能的类别，其实并不需要执行 Softmax 转换。我们完全可以直接使用 np.argmax() 找出最大输出值所对应的索引位置。


```python
for i in range(5):
    print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")
```

## SparseCategoricalCrossentropy 还是 CategoricalCrossEntropy
Tensorflow 支持两种不同的目标值 (Target Values) 格式，而你所选择的损失函数决定了模型期望接收哪一种格式：
- SparseCategoricalCrossentropy：期望目标值是一个对应于类别索引的整数。例如，如果有 10 个可能的目标类别，那么 y 的取值就应该在 0 到 9 之间。
- CategoricalCrossEntropy：期望样本的目标值采用独热编码 (One-hot Encoded) 格式，即目标类别索引所在位置的值为 1，而其余 N-1 个位置的值全为零。例如，如果有 10 个可能的目标类别，而当前样本的目标类别是 2，那么它的编码表示就是 [0,0,1,0,0,0,0,0,0,0]。


## 恭喜！
在本次实验中，你：
- 进一步熟悉了 Softmax 函数，以及它在 Softmax 回归和神经网络激活函数中的应用。
- 掌握了在 Tensorflow 中构建多分类模型的推荐方式：
    - 在最后一层不使用激活函数 (等同于使用线性激活函数 `linear`) 
    - 使用 SparseCategoricalCrossentropy 损失函数
    - 传入参数 `from_logits=True`
- 认识到与 ReLU 和 Sigmoid 不同，Softmax 运算横跨了所有的输出单元。


```python

```


```python

```
