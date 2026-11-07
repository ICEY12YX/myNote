___
- [[#选修实验 - 简单神经网络 (Simple Neural Network)|选修实验 - 简单神经网络 (Simple Neural Network)]]
	- [[#选修实验 - 简单神经网络 (Simple Neural Network)#数据集 (DataSet)|数据集 (DataSet)]]
		- [[#数据集 (DataSet)#数据归一化 (Normalize Data)|数据归一化 (Normalize Data)]]
	- [[#选修实验 - 简单神经网络 (Simple Neural Network)#NumPy 模型 (在 NumPy 中实现前向传播 (Forward Propagation) )|NumPy 模型 (在 NumPy 中实现前向传播 (Forward Propagation) )]]
		- [[#NumPy 模型 (在 NumPy 中实现前向传播 (Forward Propagation) )#预测 (Predictions)|预测 (Predictions)]]
	- [[#选修实验 - 简单神经网络 (Simple Neural Network)#网络整体功能 (Network function)|网络整体功能 (Network function)]]
	- [[#选修实验 - 简单神经网络 (Simple Neural Network)#恭喜！|恭喜！]]


___
# 选修实验 - 简单神经网络 (Simple Neural Network)

在本次实验中，我们将使用 NumPy 搭建一个小型的神经网络。它将与你在 Tensorflow 中实现的那个“咖啡烘焙”网络完全相同。
![[c2_w1_lab03_coffeeroasting_numpy-1794079708189.webp|472]]

```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

## 数据集 (DataSet)
这里使用的是与上一个实验完全相同的数据集。

```python
X,Y = load_coffee_data();
print(X.shape, Y.shape)
```

```
(200, 2) (200, 1)
```

下面让我们绘制出咖啡烘焙的数据。其中的两个特征 (Features) 分别是摄氏温度 (Temperature) 和以分钟为单位的持续时间 (Duration) 。家庭咖啡烘焙指南建议，持续时间最好保持在 12 到 15 分钟之间，而温度则应控制在 175 到 260 摄氏度之间。当然，随着温度升高，烘焙持续时间应该相应缩短。

```python
plt_roast(X,Y)
```
![[c2_w1_lab02_coffeeroasting_tf-1794077452122.webp|467]]

### 数据归一化 (Normalize Data)

为了与上一个实验保持一致，我们同样需要对数据进行归一化 (Normalization) 处理。更多细节请参考上一个实验。
(和上一节代码完全相同)
```python
print(
    f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}"
)
print(
    f"Duration    Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}"
)
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(
    f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}"
)
print(
    f"Duration    Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}"
)
```

```
Temperature Max, Min pre normalization: 284.99, 151.32
Duration    Max, Min pre normalization: 15.45, 11.51
Temperature Max, Min post normalization: 1.66, -1.69
Duration    Max, Min post normalization: 1.79, -1.70
```
## NumPy 模型 (在 NumPy 中实现前向传播 (Forward Propagation) )

  ![[c2_w1_lab03_coffeeroasting_numpy-1794079807569.webp|240]]
让我们来构建课堂上所讲的“咖啡烘焙网络”。该网络包含两个带有 Sigmoid 激活函数 (Activation Function) 的层 (Layers) 。

正如课堂上所介绍的，你完全可以使用 NumPy 自己动手编写一个全连接层 (Dense Layer) ，然后再利用它构建出一个多层神经网络。
![[c2_w1_lab03_coffeeroasting_numpy-1794079818956.webp|570]]

在第一个选修实验中，你分别用 NumPy 和 Tensorflow 构建了一个神经元 (Neuron) ，并注意到了它们之间的相似性。而==网络中的“层”，本质上只是包含了多个神经元 / 单元 (Units) 。==正如课堂上所讲，我们可以利用一个 for 循环遍历层中的每一个单元 (`j`) ，将该单元的权重 (Weights)  (`W[:,j]`) 与输入做点积 (Dot Product) ，再加上该单元的偏置 (Bias)  (`b[j]`) ，从而得到 `z`。接着，可以将激活函数 `g(z)` 应用于该计算结果。下面让我们试着用这种方法来编写一个“全连接层”子程序。

首先，你需要定义激活函数 `g()`。在这里，你将直接使用本笔记本外部的 `lab_utils_common.py` 文件中已经为你实现好的 `sigmoid()` 函数。

```python
# Define the activation function
g = sigmoid
```

接下来，你将定义 `my_dense()` 函数，用于计算全连接层的激活输出值 (Activations) 。


```python
def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1] #w 一行是一个特征的不同神经元的w
    #所以如果 只有一个特征 w就是横着的  一个神经元 w就是竖着的
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = g(z)               
    return(a_out)
```

*注：你也可以将上面的函数实现为接收 `g` 作为额外参数的形式 (例如 `my_dense(a_in, W, b, g)`) 。不过在本笔记本中，你只会用到一种激活函数 (即 Sigmoid) ，因此将它设为常量并在函数外部定义也是完全可以的。上面的代码正是这样做的，这会让接下来几个代码单元格中的函数调用更加简洁。只需记住，将激活函数作为参数传入同样是一种标准的实现方式，你将在本周的编程作业中看到这种写法。*

下面的代码单元格利用刚才定义的 `my_dense` 子程序，构建了一个两层神经网络。

```python
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)
```

我们可以直接把上一个实验中用 Tensorflow 训练好的权重和偏置复制过来。
    (因为省略了 反向传播的过程,还没学到, 这里只用np实现 前向传播 所以直接把w和b传进来了)
```python
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )
```

### 预测 (Predictions)
![[c2_w1_lab03_coffeeroasting_numpy-1794080187957.webp|481]]

一旦你有了训练好的模型，就可以用它来进行预测。回想一下，我们模型的输出是一个概率 (Probability) ——在这里即咖啡烘焙良好的概率。为了做出最终的分类决策，我们必须将该概率与一个阈值 (Threshold) 进行比较。在这里，我们将使用 0.5。

让我们先编写一个类似于 Tensorflow 中 `model.predict()` 的函数。它将接收一个每一行代表一个样本、共包含 $m$ 个样本的矩阵 (Matrix) $X$，并通过运行模型来给出预测结果。


```python
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)
```

我们可以在两个测试样本上试运行一下这个函数：

```python
X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
```

为了将概率转化为分类决策，我们需要应用阈值：
    和上一节代码一样
```python
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
```

```
decisions = 
[[1.]
 [0.]]
```

这一步同样可以用更简洁的方式来实现：

```python
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```

```
decisions = 
[[1.]
 [0.]]
```

## 网络整体功能 (Network function)

下图展示了整个网络的运行效果，它与上一个实验中 Tensorflow 得到的结果完全一致。
左图是最后一层输出的原始值，由蓝色阴影表示，上面叠加了用 X 和 O 表示的训练数据。  
右图是经过决策阈值过滤后的网络输出结果，这里的 X 和 O 对应的是网络最终做出的分类决策。

```python
netf= lambda x : my_predict(norm_l(x),W1_tmp, b1_tmp, W2_tmp, b2_tmp)
plt_network(X,Y,netf)
```
![[c2_w1_lab03_coffeeroasting_numpy-1794080362513.webp]]
(因为用的是准备好的w和b  所以做前向传播预测的时候 图看上去很完美)

## 恭喜！

你已经使用 NumPy 从零搭建起了一个小型神经网络。
希望本次实验能让你看清，构成神经网络层的其实正是那些非常简单且熟悉的基础函数。
