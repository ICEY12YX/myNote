# 实践实验：用于手写数字识别的神经网络 (Neural Networks) ——多分类 (Multiclass Classification)

在本次练习中，你将使用神经网络来识别手写数字 0-9。


# 目录大纲
-  1 - 软件包 
-  2 - ReLU 激活函数
-  3 - Softmax 函数
  -  练习 1
-  4 - 神经网络
  -  4.1 问题描述
  -  4.2 数据集
  -  4.3 模型表示
  -  4.4 Tensorflow 模型实现
  -  4.5 Softmax 的放置位置
    -  练习 2


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
from tensorflow.keras.activations import linear, relu, sigmoid
%matplotlib widget
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from public_tests import * 

from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)
```

&lt;a name=&quot;2&quot;&gt;&lt;/a&gt;
## 2 - ReLU 激活函数 (ReLU Activation)
本周我们引入了一个新的激活函数：修正线性单元 (Rectified Linear Unit, ReLU) 。
$$ a = max(0,z) \quad\quad\text {# ReLU function} $$


```python
plt_act_trio()
```

&lt;img align=&quot;right&quot; src=&quot;./images/C2_W2_ReLu.png&quot;     style=&quot; width:380px; padding: 10px 20px; &quot; &gt;
右侧来自课堂讲义的例子展示了 ReLU 的一个应用场景。在这个例子中，推导出的“认知度 (awareness) ”特征 (Feature) 并不是非黑即白的二值变量，而是具有连续的取值范围。Sigmoid 函数最适合处理“开 / 关”或二分类 (Binary) 的情况，而 ReLU 则能够提供连续的线性关系。此外，它还拥有一个输出恒为零的“关闭 (off) ”区间。  
正是这个“关闭”特性，使得 ReLU 成为了一种非线性激活函数 (Non-Linear Activation) 。为什么我们需要这一特性呢？因为它能让多个单元 (Units) 共同对最终合成的函数做出贡献，而不会互相干扰。关于这一点，我们在配套的选修实验中有更深入的探讨。

&lt;a name=&quot;3&quot;&gt;&lt;/a&gt;
## 3 - Softmax 函数 (Softmax Function)
一个多分类神经网络会生成 N 个输出，并从中选出一个输出作为预测的答案。在输出层 (Output Layer) 中，先由一个线性函数生成一个向量 (Vector) $\mathbf{z}$，然后再将其送入 Softmax 函数中。如下所述，Softmax 函数会将 $\mathbf{z}$ 转换为一个概率分布 (Probability Distribution) 。经过 Softmax 处理后，每一个输出值都会落在 0 到 1 之间，且所有输出值的总和等于 1。因此，它们可以被解释为概率。输入到 Softmax 中的数值越大，转换后对应的输出概率也就越大。
&lt;center&gt;  &lt;img  src=&quot;./images/C2_W2_NNSoftmax.PNG&quot; width=&quot;600&quot; /&gt;  

Softmax 函数可以写成如下形式：
$$a_j = \frac{e^{z_j}}{ \sum_{k=0}^{N-1}{e^{z_k} }} \tag{1}$$

其中 $z = \mathbf{w} \cdot \mathbf{x} + b$，N 是输出层中的特征 / 类别数量。  

&lt;a name=&quot;ex01&quot;&gt;&lt;/a&gt;
### 练习 1
下面让我们用 NumPy 来实现它：


```python
# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    
    
    
    
        
    
        
    
    
    ### END CODE HERE ### 
    return a
```


```python
z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf = tf.nn.softmax(z)
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")

# BEGIN UNIT TEST  
test_my_softmax(my_softmax)
# END UNIT TEST  
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    一种实现方式是：先使用一个 for 循环累加构建出分母，然后再用第二个循环计算出每一个输出值。
    
```python
def my_softmax(z):  
    N = len(z)
    a =                     # initialize a to zeros 
    ez_sum =                # initialize sum to zero
    for k in range(N):      # loop over number of outputs             
        ez_sum +=           # sum exp(z[k]) to build the shared denominator      
    for j in range(N):      # loop over number of outputs again                
        a[j] =              # divide each the exp of each output by the denominator   
    return(a)
```
&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看参考代码&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
   
```python
def my_softmax(z):  
    N = len(z)
    a = np.zeros(N)
    ez_sum = 0
    for k in range(N):                
        ez_sum += np.exp(z[k])       
    for j in range(N):                
        a[j] = np.exp(z[j])/ez_sum   
    return(a)

或者采用向量化实现：

def my_softmax(z):  
    ez = np.exp(z)              
    a = ez/np.sum(ez)           
    return(a)

```


请在下方尝试改变各个输入 `z` 的数值。请特别留意分子中的指数运算是如何放大数值之间的微小差异的，同时也要注意所有输出值的总和始终等于 1。


```python
plt.close("all")
plt_softmax(my_softmax)
```

&lt;a name=&quot;4&quot;&gt;&lt;/a&gt;
## 4 - 神经网络

在上周的作业中，你实现了一个用于二分类的神经网络。本周你将把它扩展到多分类任务中，而这将用到 Softmax 激活函数。


&lt;a name=&quot;4.1&quot;&gt;&lt;/a&gt;
### 4.1 问题描述

在本次练习中，你将使用一个神经网络来识别十个手写数字 (0-9) 。这是一个需要从 n 个选项中挑出一个结果的多分类任务。如今，自动手写数字识别已经得到了广泛应用——从识别信封上的邮政编码，到识别银行支票上手写的金额。


&lt;a name=&quot;4.2&quot;&gt;&lt;/a&gt;
### 4.2 数据集 (Dataset)

首先，你需要加载本次任务的数据集。
- 下面展示的 `load_data()` 函数会将数据加载到变量 `X` 和 `y` 中。


- 该数据集包含 5000 个手写数字的训练样本 $^1$。  

    - 每个训练样本都是一张 20 像素 x 20 像素的数字灰度图像。
        - 每个像素由一个浮点数 (Floating-point Number) 表示，用来代表该位置的灰度强度。
        - 20 乘 20 的像素网格被“展开 (unrolled) ”成一个 400 维的向量。
        - 每个训练样本都成为我们数据矩阵 (Matrix) `X` 中的单独一行。
        - 这样我们就得到了一个 5000 x 400 的矩阵 `X`，其中每一行都是一个手写数字图像的训练样本。

$$X = 
\left(\begin{array}{cc} 
--- (x^{(1)}) --- \\
--- (x^{(2)}) --- \\
\vdots \\ 
--- (x^{(m)}) --- 
\end{array}\right)$$ 

- 训练集的第二个部分是一个 5000 x 1 维的向量 `y`，其中包含了训练集的标签 (Labels) 
    - 如果图像是数字 `0`，则 `y = 0`；如果图像是数字 `4`，则 `y = 4`，依此类推。

$^1$&lt;sub&gt; 这是 MNIST 手写数字数据集 (http://yann.lecun.com/exdb/mnist/) 的一个子集&lt;/sub&gt;


```python
# load dataset
X, y = load_data()
```

#### 4.2.1 查看变量
让我们进一步熟悉一下你的数据集。  
- 一个很好的起点是把每个变量打印出来，看看里面究竟包含了什么内容。

下面的代码打印了变量 `X` 和 `y` 中的第一个元素。


```python
print ('The first element of X is: ', X[0])
```


```python
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
```

#### 4.2.2 检查变量的维度 (Dimensions)

熟悉数据的另一种方法是查看它的维度。请打印出 `X` 和 `y` 的形状 (Shape) ，看看你的数据集中一共有多少个训练样本。


```python
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))
```

#### 4.2.3 数据可视化

你将从可视化训练集的一个子集开始。
- 在下面的单元格中，代码会从 `X` 中随机选取 64 行，将每一行重新映射回 20 像素乘 20 像素的灰度图像，并将这些图像排列在一起展示出来。
- 每张图像的上方都会显示它对应的真实标签。


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

#fig.tight_layout(pad=0.5)
widgvis(fig)
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
    fig.suptitle("Label, image", fontsize=14)
```

&lt;a name=&quot;4.3&quot;&gt;&lt;/a&gt;
### 4.3 模型表示 (Model representation)

你在本次作业中将要使用的神经网络如下图所示。
- 它包含两个带有 ReLU 激活函数的全连接层 (Dense Layers) ，后面接着一个带有线性激活函数的输出层。
    - 回想一下，我们的输入是数字图像的像素值。
    - 由于图像的大小为 $20\times20$，这就为我们提供了 $400$ 个输入特征。
    
&lt;img src=&quot;images/C2_W2_Assigment_NN.png&quot; width=&quot;600&quot; height=&quot;450&quot;&gt;

- 模型参数 (Parameters) 的维度是根据网络结构量身定制的：第 1 层有 $25$ 个单元，第 2 层有 $15$ 个单元，第 3 层有 $10$ 个输出单元，分别对应十个数字类别。

    - 回想一下，这些参数的维度是按照如下规则确定的：
        - 如果网络在某一层中有 $s_{in}$ 个单元，在下一层中有 $s_{out}$ 个单元，那么：
            - 权重 (Weights) $W$ 的维度将是 $s_{in} \times s_{out}$。
            - 偏置 (Bias) $b$ 将是一个包含 $s_{out}$ 个元素的向量。
  
    - 因此，`W` 和 `b` 的形状分别为：
        - layer1：`W1` 的形状是 (400, 25) ，`b1` 的形状是 (25,)
        - layer2：`W2` 的形状是 (25, 15) ，`b2` 的形状是: (15,)
        - layer3：`W3` 的形状是 (15, 10) ，`b3` 的形状是: (10,)
>**注：** 偏置向量 `b` 可以表示为一维 (1-D) 的 (n,) 数组或二维 (2-D) 的 (n,1) 数组。Tensorflow 使用的是一维表示法，本实验也将遵循这一惯例。
               

&lt;a name=&quot;4.4&quot;&gt;&lt;/a&gt;
### 4.4 Tensorflow 模型实现


Tensorflow 模型是一层接一层搭建起来的。每一层的输入维度 (即上面的 $s_{in}$) 会由系统自动为你计算；你只需要指定某一层的*输出维度*，而这又会自动决定下一层的输入维度。至于第一层的输入维度，则是根据下面 `model.fit` 语句中所传入的输入数据大小推导出来的。
>**注：** 你也可以显式地添加一个输入层来指定第一层的输入维度。例如：  
`tf.keras.Input(shape=(400,)),    #specify input shape`  
在这里我们会把它加上，以便更清晰地展示模型维度的计算过程。

&lt;a name=&quot;4.5&quot;&gt;&lt;/a&gt;
### 4.5 Softmax 的放置位置
正如课堂讲义和 Softmax 选修实验中所介绍的，如果在训练期间将 Softmax 与损失函数 (Loss Function) 组合在一起，而不是直接放在输出层中，能够显著提高数值稳定性 (Numerical Stability) 。这一做法对于*构建*模型和*使用*模型都有着重要影响。  
在构建模型时：  
* 最后一个全连接层应当使用线性 (`'linear'`) 激活函数，这实际上等同于不使用任何激活函数。
* 在 `model.compile` 语句中，需要通过传入 `from_logits=True` 来指明这一点：
`loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) `  
* 这一做法不会影响目标标签的格式。在使用 SparseCategoricalCrossentropy 的情况下，目标标签依然是期望的数字类别 0-9。

在使用模型时：
* 模型的直接输出并不是概率。如果你需要输出概率，请手动对其应用 Softmax 函数。

&lt;a name=&quot;ex02&quot;&gt;&lt;/a&gt;
### 练习 2

请在下方使用 Keras 的 Sequential 模型和带有 ReLU 激活函数的 Dense 层，构建出上面所描述的三层神经网络。


```python
# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        
        
        
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)
```


```python
model.summary()
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;期望输出 (点击展开)&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
`model.summary()` 函数会显示一份非常实用的模型摘要。请注意，除非显式指定了层名，否则各层的名称可能会有所不同，因为它们是由系统自动生成的。    
    
```
Model: "my_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 25)                10025     
_________________________________________________________________
L2 (Dense)                   (None, 15)                390       
_________________________________________________________________
L3 (Dense)                   (None, 10)                160       
=================================================================
Total params: 10,575
Trainable params: 10,575
Non-trainable params: 0
_________________________________________________________________
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
```python
tf.random.set_seed(1234)
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(400,)),     # @REPLACE 
        Dense(25, activation='relu', name = "L1"), # @REPLACE 
        Dense(15, activation='relu',  name = "L2"), # @REPLACE  
        Dense(10, activation='linear', name = "L3"),  # @REPLACE 
        ### END CODE HERE ### 
    ], name = "my_model" 
)
``` 


```python
# BEGIN UNIT TEST     
test_model(model, 10, 400)
# END UNIT TEST     
```

摘要中显示的参数数量与权重和偏置数组中的元素数量相对应，如下所示。

让我们进一步检查一下权重的形状，验证 Tensorflow 生成的维度是否与我们上面计算的结果完全一致。


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
W3 shape = (15, 10), b3 shape = (10,)
```

下面的代码完成了两件事：
* 定义损失函数 `SparseCategoricalCrossentropy`，并通过添加 `from_logits=True` 指明应当将 Softmax 合并到损失计算中。
* 定义优化器 (Optimizer) 。这里选用的是课堂上介绍过的常用优化器——自适应矩估计 (Adaptive Moment, Adam) 。


```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)
```

#### 训练轮数 (Epochs) 与批次 (Batches)
在上面的 `fit` 语句中，`epochs` 的数量被设置为 40。这意味着整个数据集在训练过程中将被完整地使用 40 次。在训练期间，你会看到描述训练进度的输出信息，形式如下：
```
Epoch 1/40
157/157 [==============================] - 0s 1ms/step - loss: 2.2770
```
第一行 `Epoch 1/40` 描述了模型当前正在运行第几轮训练。为了提高计算效率，训练数据集会被拆分成一个个“批次 (Batches) ”。在 Tensorflow 中，一个批次的默认大小是 32。我们的数据集中共有 5000 个样本，也就是大约 157 个批次。第二行中的符号 `157/157 [====` 描述的就是当前已经执行到了哪一个批次。

#### 损失 (Loss) 与代价 (Cost)
在第 1 门课程中，我们学会了通过监控代价来追踪梯度下降 (Gradient Descent) 的进展。理想情况下，随着算法迭代次数的增加，代价会不断下降。Tensorflow 将代价统称为 `loss`。在上面执行 `model.fit` 的过程中，你已经看到了每一轮训练结束时显示的损失值。.fit 方法会返回包括损失在内的多种指标，我们在上面将其保存在了 `history` 变量中。如下所示，我们可以利用它绘制出损失随训练轮数变化的曲线图。


```python
plot_loss_tf(history)
```

#### 预测 (Prediction) 
要进行预测，可以使用 Keras 的 `predict` 方法。在下面的代码中，X[1015] 包含了一张手写数字 2 的图像。


```python
image_of_two = X[1015]
display_digit(image_of_two)

prediction = model.predict(image_of_two.reshape(1,400))  # prediction

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")
```

最大的输出值是 prediction[2]，表明模型预测该数字为 '2'。如果问题仅仅要求做出分类选择，这已经足够了，直接使用 NumPy 的 argmax 挑出最大值的索引即可。但如果问题要求给出具体的概率分布，那就需要执行一次 Softmax 转换：


```python
prediction_p = tf.nn.softmax(prediction)

print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")
```

为了返回一个代表预测目标类别的整数，你需要找出最大概率所对应的索引位置。这同样可以通过 NumPy 的 argmax 函数来实现。


```python
yhat = np.argmax(prediction_p)

print(f"np.argmax(prediction_p): {yhat}")
```

让我们随机抽取 64 个数字样本，对比一下模型的预测结果与真实标签。这段代码需要一点时间来运行。


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
widgvis(fig)
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
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()
```

下面让我们来看看其中一些预测出错的样本。
>注：增加训练轮数 (Epochs) 可以消除模型在该数据集上的这些预测错误。


```python
print( f"{display_errors(model,X,y)} errors out of {len(X)} images")
```

### 恭喜！
你已经成功构建并应用了一个神经网络来解决多分类问题。

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
