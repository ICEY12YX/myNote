___
# 基于感知机 (Perceptron) 的分类问题

在本次实验中，你将使用基于单个感知机的神经网络 (Neural Network) 模型来解决一个简单的分类问题 (Classification Problem) 。

# 目录

* 1 - 简单分类问题
* 2 - 带有激活函数的单感知机神经网络
* 2.1 - 神经网络结构
* 2.2 - 数据集
* 2.3 - 定义激活函数


* 3 - 神经网络模型的代码实现
* 3.1 - 定义神经网络结构
* 3.2 - 初始化模型参数
* 3.3 - 训练循环
* 3.4 - 在 nn_model() 中整合前述模块并执行预测


* 4 - 在更大规模数据集上的表现

## 依赖库

首先，导入在本次实验中需要使用的所有程序包。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# 用于生成合成数据集的函数
from sklearn.datasets import make_blobs 

# 在 Jupyter notebook 内部内嵌展示绘图结果
%matplotlib inline 

# 设置随机种子，确保每次运行的结果一致
np.random.seed(3)

```

## 1 - 简单分类问题

分类 (Classification) 旨在识别某个观测样本具体归属于哪一个预设类别。当目标类别仅有两个时，这类任务被称为二分类问题 (Binary Classification Problem) 。下面来看一个直观的例子。

假设你手头有一组句子，希望将其归类为“开心 (happy) ”或“愤怒 (angry) ”两类。通过文本分析，你发现所有句子中仅包含两个特征词：*aack* 和 *beep*。对于数据集中的每个句子 (即单个数据样本) ，我们统计这两个词出现的频次 ($x_1$ 与 $x_2$) 并进行对比：如果 *beep* 出现的次数更多 ($x_2 > x_1$)，该句子应归类为“愤怒”；反之 ($x_2 <= x_1$)，则归类为“开心”。从几何视角来看，这意味着在二维特征空间中必然存在一条直线，能够将这两类样本清晰切分。

我们以包含 4 个极简句子的数据集为例：

* "Beep!"
* "Aack?"
* "Beep aack..."
* "!?"

在这些样本中，$x_1$ 和 $x_2$ 的取值非 0 即 1。将这些点绘制在二维坐标平面上，可以清晰地观察到所有样本观测值分属于“愤怒” (红色) 与“开心” (蓝色) 两个类别，并且可以用一条直线充当决策边界 (Decision Boundary) 将它们划开。图中展示了这样一条分割线示例。

```python
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
x_line = np.arange(xmin, xmax, 0.1)
# 属于两个类别的样本数据点 (观测值)
ax.scatter(0, 0, color="b")
ax.scatter(0, 1, color="r")
ax.scatter(1, 0, color="b")
ax.scatter(1, 1, color="b")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# 可以充当决策边界、将两类样本分开的分割线示例
ax.plot(x_line, x_line + 0.5, color="black")
plt.plot()

```

这条直线是仅凭肉眼观察散点分布直观选取的。此类任务在机器学习中被称为具有两类线性可分样本 (Linearly Separable Classes) 的分类问题。

直线方程 $x_1-x_2+0.5 = 0$ (或写为 $x_2 = x_1 + 0.5$) 能够作为该问题的分类边界。所有位于该直线之上、满足 $x_1-x_2+0.5 < 0$ (即 $x_2 > x_1 + 0.5$) 的坐标点 $(x_1, x_2)$ 均被判定为红色类别；而位于直线下方、满足 $x_1-x_2+0.5 > 0$ ($x_2 < x_1 + 0.5$) 的点则归属于蓝色类别。因此，分类任务可以重新形式化为：针对通用直线方程 $w_1x_1+w_2x_2+b=0$，求解出理想的参数 $w_1$、$w_2$ 以及阈值截距 $b$，使其所确定的直线恰好成为最优决策边界。

在这个极简场景中，仅凭目测散点图就能直接推算出决策边界参数：$w_1 = 1$，$w_2 = -1$，$b = 0.5$。但如果面对更为错综复杂的现实数据，又该如何自动寻优？答案是构建一个简单的神经网络！接下来，我们先针对当前示例搭建神经网络模型，随后再将其推广应用于更具挑战性的复杂任务中。

## 2 - 带有激活函数的单感知机神经网络

在之前的实验中，你已经动手搭建并训练过由单个感知机 (Perceptron) 构成的网络模型。在本节中，我们将复用类似的拓扑结构，但为其引入非线性激活函数 (Activation Function) 。在此设定下，单个感知机本质上扮演着一个可微阈值判决器的角色。

### 2.1 - 神经网络结构

该神经网络的各计算组件结构如下图所示：

与先前的实验类似，输入层 (Input Layer) 包含 $x_1$ 和 $x_2$ 两个特征节点。权重向量 (Weight Vector)  $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 与偏置 (Bias)  ($b$) 是在模型训练过程中需要被不断更新优化的可学习参数。前向传播 (Forward Propagation) 的初始计算步骤与先前完全一致——对于任意给定的训练样本 $x^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)}\end{bmatrix}$：

$$z^{(i)} = w_1x_1^{(i)} + w_2x_2^{(i)} + b = Wx^{(i)} + b.\tag{1}$$

然而在分类场景下，我们不能直接将连续实数 $z^{(i)}$ 作为最终输出。一种直观离散方案是将计算结果与 0 进行比对：当数值小于 0 时归类为 0 (蓝色) ，大于 0 则归类为 1 (红色) ，随后将预测错误的样本比例定义为代价函数并执行反向传播 (Backward Propagation) 。

但这种离散阶跃方式 (即单位阶跃函数) 存在导数处处为 0 的数学缺陷。相比之下，基于连续映射的方案在优化中表现出显著优势，并广泛应用于更深层的现代神经网络中。因此，我们在此采用连续形式：为单感知机配备 Sigmoid 激活函数。

Sigmoid 激活函数的数学形式定义如下：

$$a = \sigma\left(z\right) = \frac{1}{1+e^{-z}}.\tag{2}$$

经过非线性平滑映射后，我们可以设定 0.5 作为判定阈值：若连续输出 $a > 0.5$ 则判定为类别 1 (红色) ，反之则归为类别 0 (蓝色) 。综合以上各个环节，配备 Sigmoid 激活函数的单感知机神经网络在单个样本上的前向运算可严谨表述为：

\begin{align}
z^{(i)} &=  W x^{(i)} + b,\
a^{(i)} &= \sigma\left(z^{(i)}\right).\\tag{3}
\end{align}

若将全部 $m$ 个训练样本按列整合为一个形状为 ($2 \times m$) 的输入矩阵 $X$，我们可以对矩阵运算结果执行逐元素激活。由此，整个批处理模型可以紧凑地表示为：

\begin{align}
Z &=  W X + b,\
A &= \sigma\left(Z\right),\\tag{4}
\end{align}

在此计算中，标量偏置 $b$ 借助广播机制自动扩展为形状为 ($1 \times m$) 的行向量。

处理二分类任务时，最为标准的准则函数是对数损失 (Log Loss) ，其对应的代价函数 (Cost Function) 定义如下：

$$\mathcal{L}\left(W, b\right) = \frac{1}{m}\sum_{i=1}^{m} L\left(W, b\right) = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small -y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small,\tag{5}$$

其中 $y^{(i)} \in \{0,1\}$ 代表样本的真实分类标签，而 $a^{(i)}$ 则对应前向传播得到的连续预测概率 (即矩阵 $A$ 中的元素) 。

模型训练的目标是最小化该代价函数。为了实现梯度下降法 (Gradient Descent) ，我们利用微积分中的链式法则 (Chain Rule) 分别求解各项偏导数 (Partial Derivative) ：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial w_1 } &=
\frac{1}{m}\sum_{i=1}^{m} \frac{\partial L }{ \partial a^{(i)}}
\frac{\partial a^{(i)} }{ \partial z^{(i)}}\frac{\partial z^{(i)} }{ \partial w_1},\
\frac{\partial \mathcal{L} }{ \partial w_2 } &=
\frac{1}{m}\sum_{i=1}^{m} \frac{\partial L }{ \partial a^{(i)}}
\frac{\partial a^{(i)} }{ \partial z^{(i)}}\frac{\partial z^{(i)} }{ \partial w_2},\tag{6}\
\frac{\partial \mathcal{L} }{ \partial b } &=
\frac{1}{m}\sum_{i=1}^{m} \frac{\partial L }{ \partial a^{(i)}}
\frac{\partial a^{(i)} }{ \partial z^{(i)}}\frac{\partial z^{(i)} }{ \partial b}.
\end{align}

在基础推导中，已知中间项偏导存在对消特性：$\frac{\partial L }{ \partial a^{(i)}}\frac{\partial a^{(i)} }{ \partial z^{(i)}} = \left(a^{(i)} - y^{(i)}\right)$，且 $\frac{\partial z^{(i)}}{ \partial w_1} = x_1^{(i)}$，$\frac{\partial z^{(i)}}{ \partial w_2} = x_2^{(i)}$，$\frac{\partial z^{(i)}}{ \partial b} = 1$。将这些代入公式 $(6)$ 中，偏导数可精简整理为：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial w_1 } &=
\frac{1}{m}\sum_{i=1}^{m} \left(a^{(i)} - y^{(i)}\right)x_1^{(i)},\
\frac{\partial \mathcal{L} }{ \partial w_2 } &=
\frac{1}{m}\sum_{i=1}^{m} \left(a^{(i)} - y^{(i)}\right)x_2^{(i)},\tag{7}\
\frac{\partial \mathcal{L} }{ \partial b } &=
\frac{1}{m}\sum_{i=1}^{m} \left(a^{(i)} - y^{(i)}\right).
\end{align}

值得注意的是，公式 $(7)$ 推导所得的梯度表达式与前序多元线性回归章节完全一致。因此，利用线性代数可以将其完美表述为矩阵形式：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &=
\begin{bmatrix} \frac{\partial \mathcal{L} }{ \partial w_1 } &
\frac{\partial \mathcal{L} }{ \partial w_2 }\end{bmatrix} = \frac{1}{m}\left(A - Y\right)X^T,\
\frac{\partial \mathcal{L} }{ \partial b } &= \frac{1}{m}\left(A - Y\right)\mathbf{1}.
\tag{8}
\end{align}

其中 $\left(A - Y\right)$ 为形状是 ($1 \times m$) 的误差数组，$X^T$ 为形状是 ($m \times 2$) 的输入转置矩阵，$\mathbf{1}$ 代表维度为 ($m \times 1$) 的全 1 向量。

随后，按照标准梯度更新步长对参数进行迭代修正：

\begin{align}
W &= W - \alpha \frac{\partial \mathcal{L} }{ \partial W },\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{9}\end{align}

式中 $\alpha$ 代表学习率 (Learning Rate) 。将该更新过程置于循环中持续迭代，直至代价函数平稳收敛。

训练完成后，针对任意输入样本 $x$，只需通过前向计算获得激活值 $a$，并依据阈值规则完成类别离散化：

$$\hat{y} = \begin{cases} 1 & \mbox{若 } a > 0.5 \\ 0 & \mbox{其他情况 } \end{cases}\tag{10}$$

### 2.2 - 数据集

下面生成实验所用的人工数据集。以下代码将构建 $m=30$ 个离散坐标点 $(x_1, x_2)$，其中 $x_1, x_2 \in \{0,1\}$，并按列保存在形状为 $(2 \times m)$ 的 NumPy 数组 `X` 中。标签生成逻辑设定为：当且仅当 $x_1 = 0$ 且 $x_2 = 1$ 时 $y = 1$ (红色) ，其余组合全部标记为 $y = 0$ (蓝色) ，最终存放在形状为 $(1 \times m)$ 的数组 `Y` 中。

```python
m = 30

X = np.random.randint(0, 2, (2, m))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, m))

print('Training dataset X containing (x1, x2) coordinates in the columns:')
print(X)
print('Training dataset Y containing labels of two classes (0: blue, 1: red)')
print(Y)

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (X.shape[1]))

```

### 2.3 - 定义激活函数

对应公式 $(2)$ 的 Sigmoid 激活函数实现如下：

```python
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
print("sigmoid(-2) = " + str(sigmoid(-2)))
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(3.5) = " + str(sigmoid(3.5)))

```

该函数支持对 NumPy 数组进行无缝的逐元素广播运算：

```python
print(sigmoid(np.array([-2, 0, 3.5])))

```

## 3 - 神经网络模型的代码实现

该分类网络的具体实现与先前的回归模型整体结构非常接近，主要的改动仅体现在前向传播 `forward_propagation` 和代价计算 `compute_cost` 两个函数中！

### 3.1 - 定义神经网络结构

根据数组 `X` 和 `Y` 的维度形态提取两个核心尺寸变量：

* `n_x`：输入层的神经元数量
* `n_y`：输出层的神经元数量

```python
def layer_sizes(X, Y):
    """
    参数:
    X -- 输入数据集，形状为 (输入特征维度, 样本数量)
    Y -- 标签数组，形状为 (输出维度, 样本数量)
    
    返回值:
    n_x -- 输入层的大小
    n_y -- 输出层的大小
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))

```

### 3.2 - 初始化模型参数

实现 `initialize_parameters()` 函数，将形状为 $(n_y \times n_x) = (1 \times 1)$ 的权重矩阵初始化为微小的高斯随机数，并将形状为 $(n_y \times 1) = (1 \times 1)$ 的偏置向量初始化为全零。

```python
def initialize_parameters(n_x, n_y):
    """
    返回值:
    params -- 包含模型参数的 Python 字典:
                    W -- 形状为 (n_y, n_x) 的权重矩阵
                    b -- 形状为 (n_y, 1) 的偏置向量
    """
    
    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))

    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

```

### 3.3 - 训练循环

依照第 2.1 节中的公式 $(4)$ 实现前向传播函数 `forward_propagation()`：
\begin{align}
Z &=  W X + b,\
A &= \sigma\left(Z\right).
\end{align}

```python
def forward_propagation(X, parameters):
    """
    参数:
    X -- 维度为 (n_x, m) 的输入数据
    parameters -- 包含参数的 Python 字典 (初始化函数的输出结果)
    
    返回值:
    A -- 经激活后的输出预测概率
    """
    W = parameters["W"]
    b = parameters["b"]
    
    # 执行前向传播计算线性组合 Z
    Z = np.matmul(W, X) + b
    A = sigmoid(Z)

    return A

A = forward_propagation(X, parameters)

print("Output vector A:", A)

```

此时参数均由随机数生成，模型尚未经过迭代训练。

接下来，根据公式 $(5)$ 编写用于监督学习的对数损失代价函数：

$$\mathcal{L}\left(W, b\right)  = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small -y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small.$$

```python
def compute_cost(A, Y):
    """
    计算二分类对数损失代价函数
    
    参数:
    A -- 神经网络输出的预测概率，形状为 (n_y, 样本数量)
    Y -- 真实类别标签向量，形状为 (n_y, 样本数量)
    
    返回值:
    cost -- 对数损失代价值
    
    """
    # 样本数量
    m = Y.shape[1]

    # 计算对数概率并汇总代价值
    logprobs = - np.multiply(np.log(A),Y) - np.multiply(np.log(1 - A),1 - Y)
    cost = 1/m * np.sum(logprobs)
    
    return cost

print("cost = " + str(compute_cost(A, Y)))

```

依照公式 $(8)$ 计算反向传播梯度：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &= \frac{1}{m}\left(A - Y\right)X^T,\
\frac{\partial \mathcal{L} }{ \partial b } &= \frac{1}{m}\left(A - Y\right)\mathbf{1}.
\end{align}

```python
def backward_propagation(A, X, Y):
    """
    实现反向传播算法，计算参数梯度
    
    参数:
    A -- 神经网络的输出预测，形状为 (n_y, 样本数量)
    X -- 输入数据，形状为 (n_x, 样本数量)
    Y -- 真实类别标签向量，形状为 (n_y, 样本数量)
    
    返回值:
    grads -- 包含对各个参数所求梯度的 Python 字典
    """
    m = X.shape[1]
    
    # 反向传播: 计算偏导数，为简明起见分别记为 dW 和 db 
    dZ = A - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(A, X, Y)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))

```

依照公式 $(9)$ 更新模型参数：

\begin{align}
W &= W - \alpha \frac{\partial \mathcal{L} }{ \partial W },\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b }.\end{align}

```python
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    基于梯度下降更新规则对模型参数进行迭代更新
    
    参数:
    parameters -- 包含待更新参数的 Python 字典 
    grads -- 包含梯度计算结果的 Python 字典 
    learning_rate -- 梯度下降更新的学习率
    
    返回值:
    parameters -- 包含更新后参数的 Python 字典 
    """
    # 从字典 "parameters" 中提取参数
    W = parameters["W"]
    b = parameters["b"]
    
    # 从字典 "grads" 中提取梯度
    dW = grads["dW"]
    db = grads["db"]
    
    # 各参数的梯度下降更新规则
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters_updated = update_parameters(parameters, grads)

print("W updated = " + str(parameters_updated["W"]))
print("b updated = " + str(parameters_updated["b"]))

```

### 3.4 - 在 nn_model() 中整合前述模块并执行预测

在主函数 `nn_model()` 中整合搭建完整的感知机训练流程。

```python
def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    参数:
    X -- 输入数据集，形状为 (n_x, 样本数量)
    Y -- 标签数组，形状为 (n_y, 样本数量)
    num_iterations -- 训练循环迭代的次数
    learning_rate -- 梯度下降更新的学习率
    print_cost -- 若设为 True，则每次迭代后打印当前代价值
    
    返回值:
    parameters -- 模型学习得到的最终参数，可直接用于后续样本分类预测
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)
    
    # 迭代主循环
    for i in range(0, num_iterations):
         
        # 前向传播。输入: "X, parameters"，输出: "A"
        A = forward_propagation(X, parameters)
        
        # 计算代价。输入: "A, Y"，输出: "cost"
        cost = compute_cost(A, Y)
        
        # 反向传播。输入: "A, X, Y"，输出: "grads"
        grads = backward_propagation(A, X, Y)
    
        # 梯度下降更新参数。输入: "parameters, grads, learning_rate"，输出: "parameters"
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 每次迭代打印当代价值
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

```

```python
parameters = nn_model(X, Y, num_iterations=50, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

```

观察训练输出可以发现，在迭代大约 40 次之后，代价函数虽仍在微幅降低，但下降趋势已趋于平缓。这表明训练已基本收敛，可以适时终止优化。最终学得的参数既可用于绘制几何分割线，也可直接用于样本推理。下面我们将决策边界可视化呈现出来。

```python
def plot_decision_boundary(X, Y, parameters):
    W = parameters["W"]
    b = parameters["b"]

    fig, ax = plt.subplots()
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));
    
    x_line = np.arange(np.min(X[0,:]),np.max(X[0,:])*1.1, 0.1)
    ax.plot(x_line, - W[0,0] / W[0,1] * x_line + -b[0,0] / W[0,1] , color="black")
    plt.plot()
    plt.show()
    
plot_decision_boundary(X, Y, parameters)

```

对新输入的样本点执行分类预测：

```python
def predict(X, parameters):
    """
    利用学得的模型参数预测输入数据 X 的类别归属
    
    参数:
    parameters -- 包含模型参数的 Python 字典 
    X -- 输入数据，形状为 (n_x, m)
    
    返回值:
    predictions -- 模型的预测结果向量 (蓝色: False / 红色: True)
    """
    
    # 通过前向传播计算预测概率，并以 0.5 为阈值完成 0/1 离散二分类
    A = forward_propagation(X, parameters)
    predictions = A > 0.5
    
    return predictions

X_pred = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 1]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")

```

对于结构如此小巧轻便的单神经元模型而言，这样的预测精度表现已经相当出色！

## 4 - 在更大规模数据集上的表现

接下来，借助 `sklearn.datasets` 模块中的 `make_blobs` 函数合成一个规模更大、分布更为紧凑逼真的数据集：

```python
# 构建数据集
n_samples = 1000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9]), 
                             cluster_std=1.4,
                             random_state=0)

X_larger = np.transpose(samples)
Y_larger = labels.reshape((1,n_samples))

plt.scatter(X_larger[0, :], X_larger[1, :], c=Y_larger, cmap=colors.ListedColormap(['blue', 'red']));

```

让神经网络在全量数据上完整迭代 100 轮：

```python
parameters_larger = nn_model(X_larger, Y_larger, num_iterations=100, learning_rate=1.2, print_cost=False)
print("W = " + str(parameters_larger["W"]))
print("b = " + str(parameters_larger["b"]))

```

可视化模型在大规模数据集上学到的决策边界：

```python
plot_decision_boundary(X_larger, Y_larger, parameters_larger)

```

不妨动手调整迭代轮数 `num_iterations` 与学习率 `learning_rate` 的超参数配置，探索模型收敛轨迹与决策边界形态的变化规律。

恭喜你圆满完成本次实验！

```python


```