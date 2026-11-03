# 实践实验：用于手写数字识别的神经网络 (Neural Networks) ——二分类 (Binary Classification)

在本次练习中，你将使用神经网络来识别手写数字 0 和 1。


# 目录大纲
-  1 - 软件包 
-  2 - 神经网络
  -  2.1 问题描述
  -  2.2 数据集
  -  2.3 模型表示
  -  2.4 Tensorflow 模型实现
    -  练习 1
  -  2.5 NumPy 模型实现 (在 NumPy 中实现前向传播)
    -  练习 2
  -  2.6 向量化 NumPy 模型实现 (选做)
    -  练习 3
  -  2.7 恭喜！
  -  2.8 NumPy 广播机制教程 (选读)


_**注意：** 为了防止自动评分器 (Autograder) 出错，请勿编辑或删除本笔记本中的非评分单元格。同时也请不要添加任何新的单元格。
**当你通过本次作业后**，如果想对任何非评分代码进行修改和实验，可以按照本笔记本底部的说明进行操作。_

&lt;a name=&quot;1&quot;&gt;&lt;/a&gt;
## 1 - 软件包 

首先，让我们运行下面的代码单元格，导入本次作业所需的所有软件包。
- numpy 是使用 Python 进行科学计算的基础软件包。
- matplotlib 是 Python 中广受欢迎的绘图库。
- tensorflow 是一个主流的机器学习 (Machine Learning) 平台。


```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
%matplotlib inline

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

**Tensorflow 与 Keras**  
Tensorflow 是由 Google 开发的机器学习软件包。2019 年，Google 将 Keras 集成到 Tensorflow 中，并发布了 Tensorflow 2.0。Keras 是由 François Chollet 独立开发的一个框架，它为 Tensorflow 创建了一个简单、以层 (Layer) 为中心的接口。本课程将使用 Keras 接口。

&lt;a name=&quot;2&quot;&gt;&lt;/a&gt;
## 2 - 神经网络

在第 1 门课程中，你实现了逻辑回归 (Logistic Regression) ，并利用多项式回归 (Polynomial Regression) 对其进行了扩展，以处理非线性决策边界 (Non-linear Boundaries) 。而面对诸如图像识别 (Image Recognition) 这样更加复杂的场景，神经网络则是更好的选择。

&lt;a name=&quot;2.1&quot;&gt;&lt;/a&gt;
### 2.1 问题描述

在本次练习中，你将使用一个神经网络来识别两个手写数字：0 和 1。这是一个二分类任务。如今，自动手写数字识别已经得到了广泛应用——从识别信封上的邮政编码，到识别银行支票上手写的金额。在未来的作业中，你还将把这个网络扩展为能够识别全部 10 个数字 (0-9) 。

本次练习将向你展示如何将所学的知识应用于这一分类任务中。

&lt;a name=&quot;2.2&quot;&gt;&lt;/a&gt;
### 2.2 数据集 (Dataset)

首先，你需要加载本次任务的数据集。
- 下面展示的 `load_data()` 函数会将数据加载到变量 `X` 和 `y` 中。


- 该数据集包含 1000 个手写数字的训练样本 $^1$，在这里仅限于数字 0 和 1。  

    - 每个训练样本都是一张 20 像素 x 20 像素的数字灰度图像。
        - 每个像素由一个浮点数 (Floating-point Number) 表示，用来代表该位置的灰度强度。
        - 20 乘 20 的像素网格被“展开 (unrolled) ”成一个 400 维的向量 (Vector) 。
        - 每个训练样本都成为我们数据矩阵 (Matrix) `X` 中的单独一行。
        - 这样我们就得到了一个 1000 x 400 的矩阵 `X`，其中每一行都是一个手写数字图像的训练样本。

$$X = 
\left(\begin{array}{cc} 
--- (x^{(1)}) --- \\
--- (x^{(2)}) --- \\
\vdots \\ 
--- (x^{(m)}) --- 
\end{array}\right)$$ 

- 训练集的第二个部分是一个 1000 x 1 维的向量 `y`，其中包含了训练集的标签 (Labels) 
    - 如果图像是数字 `0`，则 `y = 0`；如果图像是数字 `1`，则 `y = 1`。

$^1$&lt;sub&gt; 这是 MNIST 手写数字数据集的一个子集&lt;/sub&gt;


```python
# load dataset
X, y = load_data()
```

&lt;a name=&quot;toc_89367_2.2.1&quot;&gt;&lt;/a&gt;
#### 2.2.1 查看变量
让我们进一步熟悉一下你的数据集。  
- 一个很好的起点是把每个变量打印出来，看看里面究竟包含了什么内容。

下面的代码打印了变量 `X` 和 `y` 中的部分元素。


```python
print ('The first element of X is: ', X[0])
```


```python
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
```

&lt;a name=&quot;toc_89367_2.2.2&quot;&gt;&lt;/a&gt;
#### 2.2.2 检查变量的维度 (Dimensions)

熟悉数据的另一种方法是查看它的维度。请打印出 `X` 和 `y` 的形状 (Shape) ，看看你的数据集中一共有多少个训练样本。


```python
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))
```

&lt;a name=&quot;toc_89367_2.2.3&quot;&gt;&lt;/a&gt;
#### 2.2.3 数据可视化

你将从可视化训练集的一个子集开始。
- 在下面的单元格中，代码会从 `X` 中随机选取 64 行，将每一行重新映射回 20 像素乘 20 像素的灰度图像，并将这些图像排列在一起展示出来。
- 每张图像的上方都会显示它对应的真实标签。


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Display the label above the image
    ax.set_title(y[random_index,0])
    ax.set_axis_off()
```

&lt;a name=&quot;2.3&quot;&gt;&lt;/a&gt;
### 2.3 模型表示 (Model representation)

你在本次作业中将要使用的神经网络如下图所示。
- 它包含三个带有 Sigmoid 激活函数 (Activation Function) 的全连接层 (Dense Layers) 。
    - 回想一下，我们的输入是数字图像的像素值。
    - 由于图像的大小为 $20\times20$，这就为我们提供了 $400$ 个输入特征 (Features) 。
    
&lt;img src=&quot;images/C2_W1_Assign1.PNG&quot; width=&quot;500&quot; height=&quot;400&quot;&gt;

- 模型参数 (Parameters) 的维度是根据网络结构量身定制的：第 1 层有 $25$ 个单元 (Units) ，第 2 层有 $15$ 个单元，第 3 层有 $1$ 个输出单元。

    - 回想一下，这些参数的维度是按照如下规则确定的：
        - 如果网络在某一层中有 $s_{in}$ 个单元，在下一层中有 $s_{out}$ 个单元，那么：
            - 权重 (Weights) $W$ 的维度将是 $s_{in} \times s_{out}$。
            - 偏置 (Bias) $b$ 将是一个包含 $s_{out}$ 个元素的向量。
  
    - 因此，`W` 和 `b` 的形状分别为：
        - layer1：`W1` 的形状是 (400, 25) ，`b1` 的形状是 (25,)
        - layer2：`W2` 的形状是 (25, 15) ，`b2` 的形状是: (15,)
        - layer3：`W3` 的形状是 (15, 1) ，`b3` 的形状是: (1,)
>**注：** 偏置向量 `b` 可以表示为一维 (1-D) 的 (n,) 数组或二维 (2-D) 的 (1,n) 数组。Tensorflow 使用的是一维表示法，本实验也将遵循这一惯例。
               

&lt;a name=&quot;2.4&quot;&gt;&lt;/a&gt;
### 2.4 Tensorflow 模型实现


Tensorflow 模型是一层接一层搭建起来的。每一层的输入维度 (即上面的 $s_{in}$) 会由系统自动为你计算；你只需要指定某一层的*输出维度*，而这又会自动决定下一层的输入维度。至于第一层的输入维度，则是根据下面 `model.fit` 语句中所传入的输入数据大小推导出来的。
>**注：** 你也可以显式地添加一个输入层 (Input Layer) 来指定第一层的输入维度。例如：  
`tf.keras.Input(shape=(400,)),    #specify input shape`  
在这里我们会把它加上，以便更清晰地展示模型维度的计算过程。

&lt;a name=&quot;ex01&quot;&gt;&lt;/a&gt;
### 练习 1

请在下方使用 Keras 的 Sequential 模型和带有 Sigmoid 激活函数的 Dense 层，构建出上面所描述的网络。


```python
# UNQ_C1
# GRADED CELL: Sequential model

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        
        
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)                            

```


```python
model.summary()
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;期望输出 (点击展开) &lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
`model.summary()` 函数会显示一份非常实用的模型摘要。因为我们已经指定了输入层的大小，所以权重和偏置数组的形状都已经确定，每一层的参数总数也都能直接展示出来。请注意，各层的名称可能会有所不同，因为它们是系统自动生成的。  
    
    
```
Model: "my_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 25)                10025     
_________________________________________________________________
dense_1 (Dense)              (None, 15)                390       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 16        
=================================================================
Total params: 10,431
Trainable params: 10,431
Non-trainable params: 0
_________________________________________________________________
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
正如课堂上所讲：
    
```python
model = Sequential(                      
    [                                   
        tf.keras.Input(shape=(400,)),    # specify input size (optional)
        Dense(25, activation='sigmoid'), 
        Dense(15, activation='sigmoid'), 
        Dense(1,  activation='sigmoid')  
    ], name = "my_model"                                    
)                                       
``` 


```python
# UNIT TESTS
from public_tests import *

test_c1(model)
```

摘要中显示的参数数量与权重和偏置数组中的元素数量相对应，如下所示。


```python
L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
L2_num_params = 25 * 15 + 15   # W2 parameters  + b2 parameters
L3_num_params = 15 * 1 + 1     # W3 parameters  + b3 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params )
```

我们可以进一步检查模型的细节：首先使用 `model.layers` 提取出各个层，然后再使用 `layerx.get_weights()` 提取出对应的权重，如下所示。


```python
[layer1, layer2, layer3] = model.layers
```


```python
#### Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
```

**期望输出**
```
W1 shape = (400, 25), b1 shape = (25,)  
W2 shape = (25, 15), b2 shape = (15,)  
W3 shape = (15, 1), b3 shape = (1,)
```

`xx.get_weights` 返回的是一个 NumPy 数组。你也可以直接以张量 (Tensor) 的形式访问这些权重。请留意最后一层中张量的形状。


```python
print(model.layers[2].weights)
```

下面的代码将定义一个损失函数 (Loss Function) ，并运行梯度下降 (Gradient Descent) 以使模型的权重拟合到训练数据上。关于这一部分的更多细节将在下周的课程中详细讲解。


```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=20
)
```

要在单个样本上运行模型以进行预测，可以使用 Keras 的 `predict`。由于 `predict` 的输入必须是一个二维数组，因此需要将单个样本重塑 (Reshape) 为二维形状。


```python
prediction = model.predict(X[0].reshape(1,400))  # a zero
print(f" predicting a zero: {prediction}")
prediction = model.predict(X[500].reshape(1,400))  # a one
print(f" predicting a one:  {prediction}")
```

模型的输出可以解释为一个概率 (Probability) 。在上面的第一个例子中，输入是数字 0，模型预测该输入为 1 的概率几乎为零。
在第二个例子中，输入是数字 1，模型预测该输入为 1 的概率几乎为一。
与逻辑回归一样，我们需要将该概率与一个阈值 (Threshold) 进行比较，从而做出最终的预测判定。


```python
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print(f"prediction after threshold: {yhat}")
```

让我们随机抽取 64 个数字样本，对比一下模型的预测值与真实标签。这段代码需要一点时间来运行。


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1,400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()
```

&lt;a name=&quot;2.5&quot;&gt;&lt;/a&gt;
### 2.5 NumPy 模型实现 (在 NumPy 中实现前向传播 (Forward Propagation) )
正如课堂上所讲，你完全可以使用 NumPy 自己构建一个全连接层，然后再利用它搭建出一个多层神经网络。

&lt;img src=&quot;images/C2_W1_dense2.PNG&quot; width=&quot;600&quot; height=&quot;450&quot;&gt;


&lt;a name=&quot;ex02&quot;&gt;&lt;/a&gt;
### 练习 2

请在下方编写一个全连接层子程序。课堂上的示例使用了一个 for 循环来遍历层中的每一个单元 (`j`) ，将该单元对应的权重 (`W[:,j]`) 与输入做点积 (Dot Product) ，再加上该单元的偏置 (`b[j]`) 得到 `z`，最后将激活函数 `g(z)` 应用于该结果。本节暂不使用选修视频中介绍的矩阵运算技巧，那些内容留到后面的小节再做探索。


```python
# UNQ_C2
# GRADED FUNCTION: my_dense

def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
### START CODE HERE ### 
    
        
        
        
### END CODE HERE ### 
    return(a_out)

```


```python
# Quick Check
x_tst = 0.1*np.arange(1,3,1).reshape(2,)  # (1 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(3,)  # (3 features)
A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)
print(A_tst)
```

**期望输出**
```
[0.54735762 0.57932425 0.61063923]
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
正如课堂上所讲：
    
```python
def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):             
        w =                            # Select weights for unit j. These are in column j of W
        z =                            # dot product of w and a_in + b
        a_out[j] =                     # apply activation to z
    return(a_out)
```
   
    
&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看更多提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;

    
```python
def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):             
        w = W[:,j]                     
        z = np.dot(w, a_in) + b[j]     
        a_out[j] = g(z)                
    return(a_out)
``` 


```python
# UNIT TESTS

test_c2(my_dense)
```

下面的代码单元格利用刚才编写的 `my_dense` 子程序，构建了一个三层神经网络。


```python
def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)
```

我们可以直接从刚才训练好的 Tensorflow 模型中把权重和偏置复制过来。


```python
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()
```


```python
# make predictions
prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print( "yhat = ", yhat, " label= ", y[0,0])
prediction = my_sequential(X[500], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print( "yhat = ", yhat, " label= ", y[500,0])
```

运行下面的代码单元格，同时查看来自 NumPy 模型和 Tensorflow 模型的预测结果。这段代码需要一点时间来运行。


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network implemented in Numpy
    my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    my_yhat = int(my_prediction >= 0.5)

    # Predict using the Neural Network implemented in Tensorflow
    tf_prediction = model.predict(X[random_index].reshape(1,400))
    tf_yhat = int(tf_prediction >= 0.5)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
    ax.set_axis_off() 
fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
plt.show()
```

&lt;a name=&quot;2.6&quot;&gt;&lt;/a&gt;
### 2.6 向量化 (Vectorized) NumPy 模型实现 (选做)
选修课程中介绍了可以用来加速计算的向量与矩阵运算。
下图展示了一种层运算方式，它可以针对给定的单个输入样本，一次性计算出某一层中所有单元的输出：

&lt;img src=&quot;images/C2_W1_VectorMatrix.PNG&quot; width=&quot;600&quot; height=&quot;450&quot;&gt;

我们可以利用上面的样本 `X` 以及参数 `W1`、`b1` 来演示这一过程。我们使用 `np.matmul` 来执行矩阵乘法 (Matrix Multiply) 。请注意，如上图所示，x 和 W 的维度必须相互兼容。


```python
x = X[0].reshape(-1,1)         # column vector (400,1)
z1 = np.matmul(x.T,W1) + b1    # (1,400)(400,25) = (1,25)
a1 = sigmoid(z1)
print(a1.shape)
```

你还可以更进一步，通过一次“矩阵-矩阵”运算，同时计算出所有样本在所有单元上的输出。

&lt;img src=&quot;images/C2_W1_MatrixMatrix.PNG&quot; width=&quot;600&quot; height=&quot;450&quot;&gt;
完整的运算公式为 $\mathbf{Z}=\mathbf{XW}+\mathbf{b}$。这一步会利用 NumPy 的广播机制 (Broadcasting) 将 $\mathbf{b}$ 扩展为 $m$ 行。如果你对这一机制还不太熟悉，本笔记本末尾提供了一份简短的教程。

&lt;a name=&quot;ex03&quot;&gt;&lt;/a&gt;
### 练习 3

请在下方编写一个新的 `my_dense_v` 子程序，利用 `np.matmul()` 对由多个样本组成的矩阵执行层计算。

_**注：** 本函数不计入评分，因为它属于选修视频中关于向量化 (Vectorization) 的讨论内容。如果你还没有看那部分视频，可以随时点击期望输出下方的提示来查看参考代码。哪怕你把这里留空，也可以正常提交本笔记本。_


```python
# UNQ_C3
# UNGRADED FUNCTION: my_dense_v

def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
### START CODE HERE ### 
    
    
### END CODE HERE ### 
    return(A_out)
```


```python
X_tst = 0.1*np.arange(1,9,1).reshape(4,2) # (4 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(1,3) # (1,3 features)
A_tst = my_dense_v(X_tst, W_tst, b_tst, sigmoid)
print(A_tst)
```

**期望输出**

```
[[0.54735762 0.57932425 0.61063923]
 [0.57199613 0.61301418 0.65248946]
 [0.5962827  0.64565631 0.6921095 ]
 [0.62010643 0.67699586 0.72908792]]
 ```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    写成矩阵形式后，只需要一两行代码即可完成：
    
       Z = A_in 与 W 的 np.matmul 乘积再加上 b    
       A_out 就是 g(Z)  
&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看参考代码&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;

```python
def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j,1)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (ndarray (m,j)) : m examples, j units
    """
    Z = np.matmul(A_in,W) + b    
    A_out = g(Z)                 
    return(A_out)
```



```python
# UNIT TESTS

test_c3(my_dense_v)
```

下面的代码单元格利用刚才编写的 `my_dense_v` 子程序，构建了一个三层神经网络。


```python
def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    A1 = my_dense_v(X,  W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)
    return(A3)
```

我们同样可以从 Tensorflow 模型中复制训练好的权重和偏置。


```python
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()
```

让我们用这个新模型来进行预测。这一次，它将*一次性对所有样本*做出预测。请留意输出的形状。


```python
Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
Prediction.shape
```

我们像之前一样应用 0.5 的阈值，但这次是同时作用于所有的预测结果。


```python
Yhat = (Prediction >= 0.5).astype(int)
print("predict a zero: ",Yhat[0], "predict a one: ", Yhat[500])
```

运行下面的单元格查看预测效果。它将直接使用我们刚刚在上面计算好的预测结果。这段代码需要一点时间来运行。


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
   
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
    ax.set_axis_off() 
fig.suptitle("Label, Yhat", fontsize=16)
plt.show()
```

你还可以看看其中一张被错误分类 (Misclassified) 的图像长什么样。


```python
fig = plt.figure(figsize=(1, 1))
errors = np.where(y != Yhat)
random_index = errors[0][0]
X_random_reshaped = X[random_index].reshape((20, 20)).T
plt.imshow(X_random_reshaped, cmap='gray')
plt.title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
plt.axis('off')
plt.show()
```

&lt;a name=&quot;2.7&quot;&gt;&lt;/a&gt;
### 2.7 恭喜！
你已经成功搭建并应用了一个神经网络。

&lt;a name=&quot;2.8&quot;&gt;&lt;/a&gt;
### 2.8 NumPy 广播机制教程 (选读)


在最后一个例子中，$\mathbf{Z}=\mathbf{XW} + \mathbf{b}$ 利用了 NumPy 的广播机制来扩展向量 $\mathbf{b}$。如果你对 NumPy 广播机制还不太熟悉，可以参考下面这份简短的教程。

$\mathbf{XW}$ 是一个维度为 $(m,j_1)(j_1,j_2)$ 的“矩阵-矩阵”乘法运算，其结果是一个维度为 $(m,j_2)$ 的矩阵。在此基础上，我们要加上一个维度为 $(1,j_2)$ 的向量 $\mathbf{b}$。为了让这种逐元素 (Element-wise) 相加的运算有意义，必须先将 $\mathbf{b}$ 扩展成一个 $(m,j_2)$ 的矩阵。而这一扩展过程，正是通过 NumPy 的广播机制自动为你完成的。

广播机制适用于各种逐元素运算。  
它的基本操作逻辑是通过复制元素来“拉伸 (stretch) ”较小的维度，使其与较大的维度相匹配。

更具体地说：
在对两个数组进行运算时，NumPy 会逐一比较它们的各个维度大小。它会从最末尾 (即最右侧) 的维度开始，逐步向左推进。当满足以下两个条件之一时，两个维度就是兼容的：
- 它们相等，或者
- 其中一个维度为 1   

如果不满足这些条件，就会抛出 `ValueError: operands could not be broadcast together` 异常，表明这两个数组的形状互不兼容。而最终生成的数组在各个轴 (Axis) 上的大小，就等于输入数组在该轴上不为 1 的那个维度大小。

下面来看几个例子：

&lt;figure&gt;
    &lt;center&gt; &lt;img src=&quot;./images/C2_W1_Assign1_BroadcastIndexes.PNG&quot;  alt=&#39;missing&#39; width=&quot;400&quot;  &gt;&lt;center/&gt;
    &lt;figcaption&gt;计算广播机制的结果形状&lt;/figcaption&gt;
&lt;figure/&gt;

下图描述了扩展维度的过程。请留意图中的红色文字：

&lt;figure&gt;
    &lt;center&gt; &lt;img src=&quot;./images/C2_W1_Assign1_Broadcasting.gif&quot;  alt=&#39;missing&#39; width=&quot;600&quot;  &gt;&lt;center/&gt;
    &lt;figcaption&gt;广播机制在概念上会先扩展参数使其形状匹配，再执行逐元素运算&lt;/figcaption&gt;
&lt;figure/&gt;

上图展示了 NumPy 在执行最终运算前扩展参数以匹配形状的过程。请注意，这只是一种概念上的描述；在实际底层实现中，NumPy 会自动选择最高效的计算方式。

对于下面的每一个例子，请先试着猜一猜结果的维度大小，然后再运行代码进行验证。


```python
a = np.array([1,2,3]).reshape(-1,1)  #(3,1)
b = 5
print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")
```

请注意，这一机制适用于所有的逐元素运算：


```python
a = np.array([1,2,3]).reshape(-1,1)  #(3,1)
b = 5
print(f"(a * b).shape: {(a * b).shape}, \na * b = \n{a * b}")
```

&lt;figure&gt;
    &lt;img src=&quot;./images/C2_W1_Assign1_VectorAdd.PNG&quot;  alt=&#39;missing&#39; width=&quot;740&quot; &gt;
    &lt;center&gt;&lt;figcaption&gt;&lt;b&gt;行与列的逐元素运算&lt;/b&gt;&lt;/figcaption&gt;&lt;/center&gt;
&lt;figure/&gt;


```python
a = np.array([1,2,3,4]).reshape(-1,1)
b = np.array([1,2,3]).reshape(1,-1)
print(a)
print(b)
print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")
```

下面这张图展示的正是你在上面构建全连接层时遇到的场景：将一个一维向量 $b$ 加到一个形状为 (m,j) 的矩阵上。
&lt;figure&gt;
    &lt;img src=&quot;./images/C2_W1_Assign1_BroadcastMatrix.PNG&quot;  alt=&#39;missing&#39; width=&quot;740&quot; &gt;
    &lt;center&gt;&lt;figcaption&gt;&lt;b&gt;矩阵 + 一维向量&lt;/b&gt;&lt;/figcaption&gt;&lt;/center&gt;
&lt;figure/&gt;

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;2&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;如果你想对任何非评分代码进行修改和实验，请点击这里。&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    &lt;p&gt;&lt;i&gt;&lt;b&gt;重要提示：为了避免自动评分器出现问题，请务必在已经通过本次作业之后再进行以下操作。&lt;/b&gt;&lt;/i&gt;
    &lt;ol&gt;
        &lt;li&gt; 在笔记本的菜单栏上，依次点击 “View” &gt; “Cell Toolbar” &gt; “Edit Metadata”&lt;/li&gt;
        &lt;li&gt; 点击你想要锁定 / 解锁的代码单元格旁边的 “Edit Metadata” 按钮&lt;/li&gt;
        &lt;li&gt; 将 “editable” 属性的值设置为：
            &lt;ul&gt;
                &lt;li&gt; 如果你想解锁它，设为 “true” &lt;/li&gt;
                &lt;li&gt; 如果你想锁定它，设为 “false” &lt;/li&gt;
            &lt;/ul&gt;
        &lt;/li&gt;
        &lt;li&gt; 在笔记本的菜单栏上，依次点击 “View” &gt; “Cell Toolbar” &gt; “None” &lt;/li&gt;
    &lt;/ol&gt;
    &lt;p&gt; 下面是演示如何执行上述步骤的简短动图： 
        &lt;br&gt;
        &lt;img src=&quot;https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1&quot; align=&quot;center&quot; alt=&quot;unlock_cells.gif&quot;&gt;
&lt;/details&gt;
