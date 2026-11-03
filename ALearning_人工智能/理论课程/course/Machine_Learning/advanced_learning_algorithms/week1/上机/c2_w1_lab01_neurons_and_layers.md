# 选修实验 - 神经元 (Neurons) 与层 (Layers)
在本次实验中，我们将深入探索神经元 (Neurons) / 单元 (Units) 与层 (Layers) 的内部工作机制。具体来说，本实验将把它们与你在第 1 门课程中掌握的回归 (Regression) / 线性模型 (Linear Model) 和逻辑模型 (Logistic Model) 进行对比。此外，实验还将介绍 Tensorflow，并演示如何在该框架中实现这些模型。
&lt;figure&gt;
   &lt;img src=&quot;./images/C2_W1_NeuronsAndLayers.png&quot;  style=&quot;width:540px;height:200px;&quot; &gt;
&lt;/figure&gt;


## 软件包
**Tensorflow 与 Keras**  
Tensorflow 是由 Google 开发的机器学习 (Machine Learning) 软件包。2019 年，Google 将 Keras 集成到 Tensorflow 中，并发布了 Tensorflow 2.0。Keras 是由 François Chollet 独立开发的一个框架，它为 Tensorflow 创建了一个简单、以层为中心的接口。本课程将使用 Keras 接口。


```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('./deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

## 无激活函数 (Activation Function) 的神经元 - 回归 / 线性模型

### 数据集 (Dataset)
我们将使用第 1 门课程中的一个例子：关于房价的线性回归 (Linear Regression) 。


```python
X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()
```

### 回归 / 线性模型
不带激活函数的神经元所实现的函数，与我们在第 1 门课程中学到的线性回归完全相同：
$$ f_{\mathbf{w},b}(x^{(i)}) = \mathbf{w}\cdot x^{(i)} + b \tag{1}$$



我们可以定义一个只包含一个神经元或单元的层，并将其与我们熟悉的线性回归函数进行比较。


```python
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )
```

让我们来查看一下权重 (Weights) 。


```python
linear_layer.get_weights()
```

目前还没有任何权重，因为权重尚未被实例化。让我们用 `X_train` 中的一个样本来试运行一下这个模型，这会触发权重的实例化。请注意，输入到该层的数据必须是二维 (2-D) 的，所以我们需要对它进行形状重塑 (Reshape) 。


```python
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
```

运行结果是一个形状为 (1,1) 的张量 (Tensor)  (也就是数组的另一种叫法) ，其中只包含一个元素。  
现在让我们再来看看权重和偏置 (Bias) 。这些权重被随机初始化为很小的数值，而偏置则默认初始化为零。


```python
w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")
```

只包含单个输入特征 (Feature) 的线性回归模型 (1) 只有一个权重和一个偏置。这与我们上面定义的 `linear_layer` 的维度完全吻合。  

由于权重最初是随机生成的，下面让我们把它们设置为一些已知的数值。


```python
set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())
```

接下来，让我们将公式 (1) 的计算结果与该层的输出进行对比。


```python
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
print(alin)
```

它们计算出了完全相同的数值！
现在，我们可以利用这个线性层对训练数据进行预测了。


```python
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b
```


```python
plt_linear(X_train, Y_train, prediction_tf, prediction_np)
```

## 带有 Sigmoid 激活函数的神经元
带有 Sigmoid 激活函数的神经元 / 单元所实现的函数，与第 1 门课程中的逻辑回归 (Logistic Regression) 完全一致：
$$ f_{\mathbf{w},b}(x^{(i)}) = g(\mathbf{w}x^{(i)} + b) \tag{2}$$
其中 $$g(x) = sigmoid(x)$$ 

让我们把 $w$ 和 $b$ 设为一些已知的值，并检验一下这个模型。


### 数据集
我们将使用第 1 门课程中关于逻辑回归的一个例子。


```python
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
```


```python
pos = Y_train == 1
neg = Y_train == 0

fig,ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(X_train[pos], Y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax.scatter(X_train[neg], Y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlc["dlblue"],lw=3)

ax.set_ylim(-0.08,1.1)
ax.set_ylabel('y', fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_title('one variable plot')
ax.legend(fontsize=12)
plt.show()
```

### 逻辑神经元 (Logistic Neuron)
我们可以通过添加 Sigmoid 激活函数来实现一个“逻辑神经元”。此时，该神经元的功能就可以用上面的公式 (2) 来描述。  
本节将创建一个包含该逻辑层的 Tensorflow 模型，以此演示构建模型的另一种方法。Tensorflow 最常用于构建多层模型，而顺序模型 (Sequential) 是搭建这类模型的一种非常便捷的方式。


```python
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')
    ]
)
```

`model.summary()` 可以展示模型中的各个层以及参数 (Parameters) 数量。在这个模型中只有一层，且该层只包含一个单元。这个单元拥有两个参数：$w$ 和 $b$。


```python
model.summary()
```


```python
logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)
```

让我们把权重和偏置设置为一些已知的数值。


```python
set_w = np.array([[2]])
set_b = np.array([-4.5])
# set_weights takes a list of numpy arrays
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())
```

现在，让我们把公式 (2) 的结果与该层的输出做个比较。


```python
a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
alog = sigmoidnp(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
print(alog)
```

它们再次生成了完全相同的结果！
现在，我们可以用这个逻辑层和 NumPy 模型在训练数据上进行预测了。


```python
plt_logistic(X_train, Y_train, model, set_w, set_b, pos, neg)
```

上图中背景的阴影深浅反映了 Sigmoid 函数的输出值，其变化范围在 0 到 1 之间。

# 恭喜！
你已经成功构建了一个非常简单的神经网络 (Neural Network) ，并探索了单个神经元与第 1 门课程中线性回归和逻辑回归之间的相似之处。


```python

```


```python

```
