___
---

# 双层神经网络 (Neural Network with Two Layers)

欢迎来到第三周的编程作业。现在，你将动手构建一个双层神经网络，并通过训练让它解决一个分类问题 (Classification Problem) 。

**完成本次作业后，你将能够：**

* 构建一个双层神经网络并将其应用于分类问题
* 利用矩阵乘法 (Matrix Multiplication) 实现前向传播 (Forward Propagation)
* 完整推导并实现反向传播 (Backward Propagation)

# 目录

* 1 - 分类问题
* 2 - 双层神经网络模型
* 2.1 - 面向单个训练样本的双层神经网络模型
* 2.2 - 面向多个训练样本的双层神经网络模型
* 2.3 - 代价函数与模型训练
* 2.4 - 数据集
* 2.5 - 定义激活函数
* 练习 1




* 3 - 双层神经网络模型的代码实现
* 3.1 - 定义神经网络结构
* 练习 2


* 3.2 - 初始化模型参数
* 练习 3


* 3.3 - 训练循环
* 练习 4
* 练习 5
* 练习 6


* 3.4 - 在 nn_model() 中整合前述模块
* 练习 7
* 练习 8




* 4 - 选做：探索其他数据集

## 依赖库

首先，导入在本次作业中所需要的所有程序包。

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

载入本笔记本专用的单元测试模块。

```python
import w3_unittest

```

## 1 - 分类问题

在本周前序的实验中，你已经成功训练过一个基于单感知机 (Perceptron) 的神经网络，并实现了前向与反向传播。那种简单的结构足以解决“线性”分类问题——也就是在二维平面上寻找一条能够将两个类别清晰划分的直线决策边界 (Decision Boundary) 。

但设想一下，如果面对的是一个更为复杂的场景：数据依然只有两类，但任何一条单一的直线都无法将它们完整切分。

```python
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
x_line = np.arange(xmin, xmax, 0.1)
# 属于两个类别的样本数据点 (观测值)
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="b")
ax.scatter(1, 0, color="b")
ax.scatter(1, 1, color="r")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# 可以充当决策边界、将两类样本分开的分割线示例
ax.plot(x_line, -1 * x_line + 1.5, color="black")
ax.plot(x_line, -1 * x_line + 0.5, color="black")
plt.plot()

```

这种逻辑在现实生活里非常普遍。举个例子，假设我们希望训练一个模型，根据房屋的“面积大小”与“建造年份”来预测是否值得购买：一套又大又新的房子价格往往高不可攀让人难以承受，而一套又小又破的旧房子则缺乏购买价值。因此，最终吸引你的可能是“面积大但房龄老的房子”，或是“面积紧凑但崭新的新房”。

对于这类非线性分布的分类问题，单个感知机显然已无法胜任。接下来，让我们来看看如何调整并升级模型架构来攻克这一难题。

在上面的图示中，需要两条直线协同配合才能形成有效的决策边界。你的直觉可能会提醒：既然一条线对应一个感知机，那么我们是否应该增加感知机的数量？答案完全正确！我们需要将数据点的坐标 $(x_1, x_2)$ 同时输入到两个独立的节点中分别计算，然后再把它们的结果汇集到下一个节点中做出统一判定。

现在，就让我们深入探究其中的数学奥秘，动手搭建并训练你的第一个多层神经网络吧！

## 2 - 双层神经网络模型

### 2.1 - 面向单个训练样本的双层神经网络模型

该网络的输入层 (Input Layer) 和输出层 (Output Layer) 与单感知机模型结构相似，但在它们之间新增了一个隐藏层 (Hidden Layer) 。来自输入层且维度为 $n_x = 2$ 的训练样本 $x^{(i)}=\begin{bmatrix}x_1^{(i)} \\ x_2^{(i)}\end{bmatrix}$，首先会被送入大小为 $n_h = 2$ 的隐藏层。数据会被并行输入到隐藏层的两个感知机中：第一个感知机的权重为 $W_1^{[1]}=\begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]}\end{bmatrix}$、偏置 (Bias) 为 $b_1^{[1]}$；第二个感知机的权重为 $W_2^{[1]}=\begin{bmatrix}w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$、偏置为 $b_2^{[1]}$。在上标方括号 $^{[1]}$ 中的整数代表当前所在的层数——由于模型现在拥有两层，各层都有自己专属的参数与运算输出，因此需要以此严加区分。

\begin{align}
z_1^{[1](https://www.google.com/search?q=i)} &= w_{1,1}^{[1]} x_1^{(i)} + w_{2,1}^{[1]} x_2^{(i)} + b_1^{[1]} = W_1^{[1]}x^{(i)} + b_1^{[1]},\
z_2^{[1](https://www.google.com/search?q=i)} &= w_{1,2}^{[1]} x_1^{(i)} + w_{2,2}^{[1]} x_2^{(i)} + b_2^{[1]} = W_2^{[1]}x^{(i)} + b_2^{[1]}.\tag{1}
\end{align}

对于单个训练样本 $x^{(i)}$，上述公式可以紧凑地改写为矩阵形式：

$$z^{[1](i)} = W^{[1]} x^{(i)} + b^{[1]},\tag{2}$$

其中：

    $z^{[1](i)} = \begin{bmatrix}z_1^{[1](i)} \\ z_2^{[1](i)}\end{bmatrix}$ 是形状为 $\left(n_h \times 1\right) = \left(2 \times 1\right)$ 的列向量 (Vector) ；

    $W^{[1]} = \begin{bmatrix}W_1^{[1]} \\ W_2^{[1]}\end{bmatrix} =  \begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]} \\ w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$ 是形状为 $\left(n_h \times n_x\right) = \left(2 \times 2\right)$ 的权重矩阵 (Matrix) ；

    $b^{[1]} = \begin{bmatrix}b_1^{[1]} \\ b_2^{[1]}\end{bmatrix}$ 是形状为 $\left(n_h \times 1\right) = \left(2 \times 1\right)$ 的偏置向量。

接下来，需要对向量 $z^{[1](i)}$ 中的每一个分量分别施加隐藏层的激活函数 (Activation Function) 。在神经网络中可以选择多种不同的激活函数，本模型采用经典的 Sigmoid 函数 $\sigma\left(x\right) = \frac{1}{1 + e^{-x}}$。请牢记它的导数性质：$\frac{d\sigma}{dx} = \sigma\left(x\right)\left(1-\sigma\left(x\right)\right)$。经激活后，隐藏层输出的是一个维度为 $\left(n_h \times 1\right) = \left(2 \times 1\right)$ 的激活向量：

$$a^{[1](i)} = \sigma\left(z^{[1](i)}\right) =  \begin{bmatrix}\sigma\left(z_1^{[1](i)}\right) \\ \sigma\left(z_2^{[1](i)}\right)\end{bmatrix}.\tag{3}$$

随后，隐藏层的输出向量会被作为输入传入节点数为 $n_y = 1$ 的输出层。这部分与上一实验的内容完全一致，唯一的差异仅在于：此处将先前的原始输入 $x^{(i)}$ 替换为了隐藏层的特征表示 $a^{[1](i)}$，并且参数与输出都标注了层数上标 $^{[2]}$：

$$z^{[2](i)} = w_1^{[2]} a_1^{[1](i)} + w_2^{[2]} a_2^{[1](i)} + b^{[2]}= W^{[2]} a^{[1](i)} + b^{[2]},\tag{4}$$

    由于 $\left(n_y \times 1\right) = \left(1 \times 1\right)$，在此模型中 $z^{[2](i)}$ 和 $b^{[2]}$ 均为标量；

    $W^{[2]} = \begin{bmatrix}w_1^{[2]} & w_2^{[2]}\end{bmatrix}$ 则是维度为 $\left(n_y \times n_h\right) = \left(1 \times 2\right)$ 的行向量。

最后，输出层同样采用 Sigmoid 函数进行非线性映射：

$$a^{[2](i)} = \sigma\left(z^{[2](i)}\right).\tag{5}$$

综上所述，针对单个样本 $x^{(i)}$ 的双层神经网络前向运算全流程可以通过公式 $(2) - (5)$ 完整表达。为了便于对照查阅，我们将其汇总排列在一起：

\begin{align}
z^{[1](https://www.google.com/search?q=i)} &= W^{[1]} x^{(i)} + b^{[1]},\
a^{[1](https://www.google.com/search?q=i)} &= \sigma\left(z^{[1](https://www.google.com/search?q=i)}\right),\
z^{[2](https://www.google.com/search?q=i)} &= W^{[2]} a^{[1](https://www.google.com/search?q=i)} + b^{[2]},\
a^{[2](https://www.google.com/search?q=i)} &= \sigma\left(z^{[2](https://www.google.com/search?q=i)}\right).\
\tag{6}
\end{align}

需要特别注意：模型中所有待优化的权重与偏置参数均不带有样本索引上标 $^{(i)}$——因为它们独立于具体的输入数据，是全数据集共享的模型参数。

最终，对于任意输入样本 $x^{(i)}$，只需依据输出概率值 $a^{[2](i)}$ 设定判别阈值，即可得到离散的分类预测类别 $\hat{y}$：当 $a^{[2](i)} > 0.5$ 时，预测 $\hat{y} = 1$；否则 $\hat{y} = 0$。

### 2.2 - 面向多个训练样本的双层神经网络模型

类似于单感知机模型，我们可以将全部 $m$ 个训练样本整合成一个形状为 ($2 \times m$) 的输入矩阵 $X$，其中每一列对应一个具体的样本向量 $x^{(i)}$。此时，公式 $(6)$ 即可优雅地转化为面向全数据集的向量化 (Vectorization) 矩阵乘法：

\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]},\
A^{[1]} &= \sigma\left(Z^{[1]}\right),\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]},\
A^{[2]} &= \sigma\left(Z^{[2]}\right),\
\tag{7}
\end{align}

在此计算中，标量或列向量形式的偏置 $b^{[1]}$ 会通过广播机制扩展为形状为 $\left(n_h \times m\right) = \left(2 \times m\right)$ 的矩阵，而 $b^{[2]}$ 则会被广播扩展为 $\left(n_y \times m\right) = \left(1 \times m\right)$ 的行向量。强烈建议你仔细推演公式 $(7)$ 中各个矩阵的维数，验证它们是否满足矩阵相乘的尺寸对齐法则。

至此，前向传播的数学表达已全部构建完成。接下来，是时候对模型的效果进行评估，并开启训练优化之路了。

### 2.3 - 代价函数与模型训练

为了评估并衡量该神经网络的分类性能，我们沿用单感知机分类时所采用的对数损失函数 (Log Loss Function) 。在模型刚开始被赋予随机权重时其预测效果往往很差，我们需要通过模型训练：寻找一组最优参数组合 $W^{[1]}$、$b^{[1]}$、$W^{[2]}$ 和 $b^{[2]}$，使整体代价函数降至最低。

与单感知机神经网络类似，整体代价函数可以定义为：

$$\mathcal{L}\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right) = \frac{1}{m}\sum_{i=1}^{m} L\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right) =  \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small - y^{(i)}\log\left(a^{[2](i)}\right) - (1-y^{(i)})\log\left(1- a^{[2](i)}\right)  \large  \right), \small\tag{8}$$

式中 $y^{(i)} \in \{0,1\}$ 为样本真实的分类标签，而 $a^{[2](i)}$ 则代表前向传播计算得出的连续输出概率值 (对应矩阵 $A^{[2]}$ 中的对应元素) 。

为了最小化代价函数，我们使用梯度下降法 (Gradient Descent) ，按照如下规则对各项参数进行迭代更新：

\begin{align}
W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\
b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\
W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\
b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} },\
\tag{9}
\end{align}

其中 $\alpha$ 为控制更新步长的学习率 (Learning Rate) 。

为了完成参数的更新，我们必须分别求解出对应的偏导数 (Partial Derivative) ：$\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$、$\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$、$\frac{\partial \mathcal{L} }{ \partial W^{[2]}}$ 以及 $\frac{\partial \mathcal{L} }{ \partial b^{[2]}}$。

让我们从网络的末端反向推演。首先回顾单感知机神经网络中关于 $\frac{\partial \mathcal{L} }{ \partial W }$ 与 $\frac{\partial \mathcal{L} }{ \partial b }$ 的计算形式：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &=
\frac{1}{m}\left(A-Y\right)X^T,\
\frac{\partial \mathcal{L} }{ \partial b } &=
\frac{1}{m}\left(A-Y\right)\mathbf{1},\
\end{align}

其中的 $\mathbf{1}$ 代表维度为 ($m \times 1$) 的全 1 向量。在当前的双层架构中，原先的感知机被移到了第 2 层，因此只需将对应符号进行替换：将 $W$ 替换为 $W^{[2]}$、$b$ 替换为 $b^{[2]}$、$A$ 替换为 $A^{[2]}$，而输入 $X$ 则替换为来自前一层的隐藏特征 $A^{[1]}$：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &=
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &=
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1}.\
\tag{10}
\end{align}

接下来，推导针对第一层权重的梯度 $\frac{\partial \mathcal{L} }{ \partial W^{[1]}} =  \begin{bmatrix} \frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,1}^{[1]}} \\ \frac{\partial \mathcal{L} }{ \partial w_{1,2}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,2}^{[1]}} \end{bmatrix}$。在课程视频中已经阐明：

$$\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}}=\frac{1}{m}\sum_{i=1}^{m} \left(  \left(a^{[2](i)} - y^{(i)}\right)  w_1^{[2]}  \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right) x_1^{(i)}\right)\tag{11}$$

如果对矩阵 $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ 中的每个元素都严谨地展开求导，将会得到如下矩阵：

$$\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \begin{bmatrix} \frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,1}^{[1]}} \\ \frac{\partial \mathcal{L} }{ \partial w_{1,2}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,2}^{[1]}} \end{bmatrix}$$

$$= \frac{1}{m}\begin{bmatrix} \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right) x_1^{(i)}\right) &  \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right) x_2^{(i)}\right)  \\ \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right) x_1^{(i)}\right) &  \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right) x_2^{(i)}\right)\end{bmatrix}\tag{12}$$

细致观察可以发现，其中各分项与下标的排布具有极高的对称性与规律性，这意味着它们同样可以整合为高度凝练的矩阵相乘形式！确实如此：转置矩阵 $\left(W^{[2]}\right)^T = \begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix}$ 的尺寸为 $\left(n_h \times n_y\right) = \left(2 \times 1\right)$，当它与尺寸为 $\left(n_y \times m\right) = \left(1 \times m\right)$ 的误差向量 $A^{[2]} - Y$ 相乘时，即可得到一个形状为 $\left(n_h \times m\right) = \left(2 \times m\right)$ 的矩阵：

$$\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)= \begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix} \begin{bmatrix}\left(a^{[2](1)} - y^{(1)}\right) &  \cdots & \left(a^{[2](m)} - y^{(m)}\right)\end{bmatrix} =\begin{bmatrix} \left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]} & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]} \\ \left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]} & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \end{bmatrix}$$


.

现在取同为 $\left(n_h \times m\right) = \left(2 \times m\right)$ 维度的矩阵 $A^{[1]}$：

$$A^{[1]} =\begin{bmatrix} a_1^{[1](1)} & \cdots & a_1^{[1](m)} \\ a_2^{[1](1)} & \cdots & a_2^{[1](m)} \end{bmatrix},$$

我们可以计算：

$$A^{[1]}\cdot\left(1-A^{[1]}\right) =\begin{bmatrix} a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right) & \cdots & a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right) \\ a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right) & \cdots & a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right) \end{bmatrix},$$

这里的“$\cdot$”代表逐元素乘法 (Element by Element Multiplication，即 Hadamard 积) 。

通过逐元素相乘，我们得到：

$$\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)=\begin{bmatrix} \left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]}\left(a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]}\left(a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right)\right) \\ \left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]}\left(a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \left(a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right)\right) \end{bmatrix}.$$

若进一步与尺寸为 $\left(m \times n_x\right) = \left(m \times 2\right)$ 的 $X^T$ 执行矩阵点乘，最终便能得到一个形状为 $\left(n_h \times n_x\right) = \left(2 \times 2\right)$ 的矩阵：

$$\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T =  \begin{bmatrix} \left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]}\left(a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]}\left(a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right)\right) \\ \left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]}\left(a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \left(a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right)\right) \end{bmatrix} \begin{bmatrix} x_1^{(1)} & x_2^{(1)} \\ \cdots & \cdots \\ x_1^{(m)} & x_2^{(m)} \end{bmatrix}$$

$$=\begin{bmatrix} \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1 - a_1^{[1](i)}\right) \right) x_1^{(i)}\right) &  \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right) x_2^{(i)}\right)  \\ \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right) x_1^{(i)}\right) &  \sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right) x_2^{(i)}\right)\end{bmatrix}$$

这个展开式与公式 $(12)$ 完全吻合！因此，$\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ 可以非常紧凑地表示为组合运算的形式：

$$\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T\tag{13},$$

其中“$\cdot$”依旧代表逐元素乘法。

针对偏置的梯度向量 $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$ 求解过程十分类似，只是在链式法则 (Chain Rule) 的末端对应偏导项为 $1$，即 $\frac{\partial z_1^{[1](i)}}{ \partial b_1^{[1]}} = 1$。因此可得：

$$\frac{\partial \mathcal{L} }{ \partial b^{[1]}} = \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1},\tag{14}$$

这里的 $\mathbf{1}$ 代表形状为 ($m \times 1$) 的全 1 向量。

至此，公式 $(10)$、$(13)$ 和 $(14)$ 便构成了反向传播更新参数 $(9)$ 时所需的全部梯度计算法则：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &=
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &=
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\
\frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\
\frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1},\
\tag{15}
\end{align}

其中 $\mathbf{1}$ 为维度为 ($m \times 1$) 的全 1 向量。

由此可见，想要从底层深刻而透彻地理解神经网络的运转与训练机制，**线性代数与微积分的紧密结合是不可或缺的坚实基石**！不过切勿心生畏惧，只要循序渐进、厘清背后的数学脉络，一切都尽在掌握之中。

现在，让我们把这些精妙的公式转化为优雅的 Python 代码吧！

### 2.2 - 数据集

首先，生成本次实验将要使用的数据集。以下代码将构建 $m=2000$ 个二维样本点 $(x_1, x_2)$，并按列保存在形状为 $(2 \times m)$ 的 NumPy 数组 `X` 中。对应的标签信息 (0: 蓝色，1: 红色) 则以 $(1 \times m)$ 的形式保存在 NumPy 数组 `Y` 中。

```python
m = 2000
samples, labels = make_blobs(n_samples=m, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (m))

```

### 2.3 - 定义激活函数

### 练习 1

定义 Sigmoid 激活函数：$\sigma\left(z\right) =\frac{1}{1+e^{-z}}$。

```python
def sigmoid(z):
    ### START CODE HERE ### (~ 1 line of code)
    res = None
    ### END CODE HERE ###
    
    return res

```

```python
print("sigmoid(-2) = " + str(sigmoid(-2)))
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(3.5) = " + str(sigmoid(3.5)))

```

##### **预期输出**

注意：因浮点数精度原因，末尾几位小数可能略有微小差异。

```Python
sigmoid(-2) = 0.11920292202211755
sigmoid(0) = 0.5
sigmoid(3.5) = 0.9706877692486436

```

```python
w3_unittest.test_sigmoid(sigmoid)

```

## 3 - 双层神经网络模型的代码实现

### 3.1 - 定义神经网络结构

### 练习 2

定义以下三个变量：

* `n_x`：输入层的神经元数量
* `n_h`：隐藏层的神经元数量 (暂设为 2)
* `n_y`：输出层的神经元数量

```python
# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    参数:
    X -- 输入数据集，形状为 (输入特征维度, 样本数量)
    Y -- 真实标签，形状为 (输出维度, 样本数量)
    
    返回值:
    n_x -- 输入层的大小
    n_h -- 隐藏层的大小
    n_y -- 输出层的大小
    """
    ### START CODE HERE ### (~ 3 lines of code)
    # 输入层大小
    n_x = None
    # 隐藏层大小
    n_h = None
    # 输出层大小
    n_y = None 
    ### END CODE HERE ###
    return (n_x, n_h, n_y)

```

```python
(n_x, n_h, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

```

##### **预期输出**

```Python
The size of the input layer is: n_x = 2
The size of the hidden layer is: n_h = 2
The size of the output layer is: n_y = 1

```

```python
w3_unittest.test_layer_sizes(layer_sizes)

```

### 3.2 - 初始化模型参数

### 练习 3

实现参数初始化函数 `initialize_parameters()`。

**指导说明**：

* 务必确保参数的矩阵尺寸精确无误。如有需要，可回顾前文中的网络架构图。
* 采用微小的随机数值对权重矩阵进行初始化。
* 使用：`np.random.randn(a,b) * 0.01` 生成形状为 (a,b) 的随机高斯分布矩阵。


* 偏置向量则统一初始化为全零。
* 使用：`np.zeros((a,b))` 构建形状为 (a,b) 的全零矩阵。



```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    参数:
    n_x -- 输入层的大小
    n_h -- 隐藏层的大小
    n_y -- 输出层的大小
    
    返回值:
    params -- 包含模型参数的 Python 字典:
                    W1 -- 形状为 (n_h, n_x) 的隐藏层权重矩阵
                    b1 -- 形状为 (n_h, 1) 的隐藏层偏置向量
                    W2 -- 形状为 (n_y, n_h) 的输出层权重矩阵
                    b2 -- 形状为 (n_y, 1) 的输出层偏置向量
    """
    
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

```

```python
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

```

##### **预期输出**

注意：由于权重属于随机初始化，数组 W1 与 W2 中的数值可能会略有差异。若想获得完全一致的数值，可以尝试重启 Jupyter 内核。

```Python
W1 = [[ 0.01788628  0.0043651 ]
 [ 0.00096497 -0.01863493]]
b1 = [[0.]
 [0.]]
W2 = [[-0.00277388 -0.00354759]]
b2 = [[0.]]

```

```python
# 注意: 
# 此处的单元测试不会比对参数的具体随机数值（因为初始化存在随机性）。
w3_unittest.test_initialize_parameters(initialize_parameters)

```
---

### 3.3 - 训练循环

### 练习 4

实现前向传播函数 `forward_propagation()`。

**指导说明**：

* 回顾前文第 2.2 节中关于分类器的数学模型公式 $(7)$：
\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]},\
A^{[1]} &= \sigma\left(Z^{[1]}\right),\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]},\
A^{[2]} &= \sigma\left(Z^{[2]}\right).\
\end{align}
* 你需要完成以下步骤：
1. 通过键名索引 `parameters[".."]`，从参数字典 "parameters" (即 `initialize_parameters()` 的输出结果) 中提取各个模型参数。
2. 实现前向传播 (Forward Propagation) 运算流程：将权重矩阵 `W1` 与特征输入矩阵 `X` 相乘并累加偏置向量 `b1`，求得 `Z1`；接着调用 `sigmoid` 激活函数 (Activation Function) 计算得到隐藏层激活值 `A1`。随后以同样的方式依序计算输出层的 `Z2` 与 `A2`。



```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    参数:
    X -- 输入数据，维度为 (n_x, m)
    parameters -- 包含模型参数的 Python 字典 (即参数初始化函数的输出结果)
    
    返回值:
    A2 -- 经过第二层激活函数计算后得到的 Sigmoid 输出
    cache -- 包含 Z1, A1, Z2, A2 的 Python 字典
    (将其缓存在字典中能够极大简化后续反向传播步骤中的梯度运算)
    """
    # 从字典 "parameters" 中分别提取各层参数
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    # 执行前向传播以计算最终输出 A2
    ### START CODE HERE ### (~ 4 lines of code)
    Z1 = None
    A1 = None
    Z2 = None
    A2 = None
    ### END CODE HERE ###
    
    assert(A2.shape == (n_y, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

```

```python
A2, cache = forward_propagation(X, parameters)

print(A2)

```

##### **预期输出**

注意：受初始参数随机赋值的影响，数组 A2 中的具体数值可能会略有差异。若想获得完全一致的数值，可以尝试重启 Jupyter 内核并重新运行该笔记本。

```Python
[[0.49920157 0.49922234 0.49921223 ... 0.49921215 0.49921043 0.49920665]]

```

```python
# 注意: 
# 单元测试在此处不会校验具体数值（因为参数初始化存在随机性）。
w3_unittest.test_forward_propagation(forward_propagation)

```

请记住，此时网络的权重仅仅被赋予了微小的随机初值，模型尚未接受任何数据训练。

### 练习 5

定义用于指导模型训练学习的代价函数 (Cost Function)  $(8)$：

$$\mathcal{L}\left(W, b\right)  = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small - y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small.$$

```python
def compute_cost(A2, Y):
    """
    基于对数损失计算模型的代价值
    
    参数:
    A2 -- 神经网络输出的预测概率值，形状为 (1, 样本数量)
    Y -- 真实类别标签向量，形状为 (1, 样本数量)
    
    返回值:
    cost -- 对数损失值 (Log Loss)
    
    """
    # 样本数量
    m = Y.shape[1]
    
    ### START CODE HERE ### (~ 2 lines of code)
    logloss = None
    cost = None
    ### END CODE HERE ###

    assert(isinstance(cost, float))
    
    return cost

```

```python
print("cost = " + str(compute_cost(A2, Y)))

```

##### **预期输出**

注意：权重矩阵 W1 与 W2 的初始随机差异可能会使计算结果略有不同！

```Python
cost = 0.6931477703826823

```

```python
# 注意: 
# 此处单元测试不会强行比对浮点数值（因为初始权重存在随机性）。
w3_unittest.test_compute_cost(compute_cost, A2)

```

按照公式 $(15)$ 计算代价函数对各参数的偏导数 (Partial Derivative) ：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &=
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &=
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\
\frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\
\frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1}.\
\end{align}

```python
def backward_propagation(parameters, cache, X, Y):
    """
    实现反向传播算法，计算各参数的梯度
    
    参数:
    parameters -- 包含当前网络参数的 Python 字典 
    cache -- 包含前向传播中间变量 Z1, A1, Z2, A2 的 Python 字典
    X -- 输入数据，形状为 (n_x, 样本数量)
    Y -- 真实类别标签向量，形状为 (n_y, 样本数量)
    
    返回值:
    grads -- 包含对各个参数所求梯度的 Python 字典
    """
    m = X.shape[1]
    
    # 首先，从字典 "parameters" 中获取权重矩阵 W
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # 接着，从缓存字典 "cache" 中提取激活值 A1 与 A2
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # 执行反向传播: 为简明起见，将对各参数求取的偏导数记为 dW1, db1, dW2, db2 
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

grads = backward_propagation(parameters, cache, X, Y)

print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))

```

### 练习 6

实现参数更新函数 `update_parameters()`。

**指导说明**：

* 按照第 2.3 节中的公式 $(9)$ 对各参数执行更新：
\begin{align}
W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\
b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\
W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\
b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} }.\
\end{align}
* 编写代码时请遵循以下步骤：
1. 使用 `parameters[".."]` 从字典 "parameters" (即 `initialize_parameters()` 的返回对象) 中读取现有参数。
2. 使用 `grads[".."]` 从梯度字典 "grads" (即 `backward_propagation()` 的计算结果) 中读取对应的偏导数。
3. 执行参数的迭代更新。



```python
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    基于梯度下降更新规则对模型参数进行迭代更新
    
    参数:
    parameters -- 包含待更新参数的 Python 字典 
    grads -- 包含梯度计算结果的 Python 字典 
    learning_rate -- 梯度下降算法的学习率参数 (Learning Rate)
    
    返回值:
    parameters -- 包含更新后参数的 Python 字典 
    """
    # 从字典 "parameters" 中提取各参数
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    # 从字典 "grads" 中提取各梯度
    ### START CODE HERE ### (~ 4 lines of code)
    dW1 = None
    db1 = None
    dW2 = None
    db2 = None
    ### END CODE HERE ###
    
    # 针对各项参数执行梯度下降更新
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

```

```python
parameters_updated = update_parameters(parameters, grads)

print("W1 updated = " + str(parameters_updated["W1"]))
print("b1 updated = " + str(parameters_updated["b1"]))
print("W2 updated = " + str(parameters_updated["W2"]))
print("b2 updated = " + str(parameters_updated["b2"]))

```

##### **预期输出**

注意：实际计算所得的数值可能会因初始化差异而略有不同！

```Python
W1 updated = [[ 0.01790427  0.00434496]
 [ 0.00099046 -0.01866419]]
b1 updated = [[-6.13449205e-07]
 [-8.47483463e-07]]
W2 updated = [[-0.00238219 -0.00323487]]
b2 updated = [[0.00094478]]

```

```python
w3_unittest.test_update_parameters(update_parameters)

```

### 3.4 - 在 nn_model() 中整合前述模块

### 练习 7

在主函数 `nn_model()` 中组装搭建完整的神经网络模型。

**指导说明**：神经网络模型需要按照正确的次序依次调用前面实现的各个功能函数。

```python
# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations=10, learning_rate=1.2, print_cost=False):
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
    n_y = layer_sizes(X, Y)[2]
    
    # 初始化模型参数
    ### START CODE HERE ### (~ 1 line of code)
    parameters = None
    ### END CODE HERE ###
    
    # 迭代主循环
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 4 lines of code)
        # 前向传播。输入: "X, parameters"，输出: "A2, cache"
        A2, cache = None
        
        # 计算代价。输入: "A2, Y"，输出: "cost"
        cost = None
        
        # 反向传播。输入: "parameters, cache, X, Y"，输出: "grads"
        grads = None
        
        # 梯度下降更新参数。输入: "parameters, grads, learning_rate"，输出: "parameters"
        parameters = None
        ### END CODE HERE ###
        
        # 每次迭代打印当代价值
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

```

```python
parameters = nn_model(X, Y, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]

```

##### **预期输出**

注意：实际计算所得的数值可能会略有差异！

```Python
Cost after iteration 0: 0.693148
Cost after iteration 1: 0.693147
Cost after iteration 2: 0.693147
Cost after iteration 3: 0.693147
Cost after iteration 4: 0.693147
Cost after iteration 5: 0.693147
...
Cost after iteration 2995: 0.209524
Cost after iteration 2996: 0.208025
Cost after iteration 2997: 0.210427
Cost after iteration 2998: 0.208929
Cost after iteration 2999: 0.211306
W1 = [[ 2.14274251 -1.93155541]
 [ 2.20268789 -2.1131799 ]]
b1 = [[-4.83079243]
 [ 6.2845223 ]]
W2 = [[-7.21370685  7.0898022 ]]
b2 = [[-3.48755239]]

```

```python
# 注意: 
# 单元测试在此处不会校验具体数值（因为参数初始化存在随机性）。
w3_unittest.test_nn_model(nn_model)

```

最终学得的模型参数既可用于绘制出分类决策分界线，也能直接用于对新样本进行推理预测。

### 练习 8

通过前向传播计算预测概率，并以 0.5 为分类阈值完成 0/1 的二分类判断。

```python
# GRADED FUNCTION: predict

def predict(X, parameters):
    """
    利用学习得到的参数，对输入数据 X 中的每个样本进行类别预测
    
    参数:
    parameters -- 包含模型参数的 Python 字典 
    X -- 输入数据，形状为 (n_x, m)
    
    返回值:
    predictions -- 模型的预测类别向量 (蓝色: 0 / 红色: 1)
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = None
    predictions = None
    ### END CODE HERE ###
    
    return predictions

```

```python
X_pred = np.array([[2, 8, 2, 8], [2, 8, 8, 2]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")

```

##### **预期输出**

```Python
Coordinates (in the columns):
[[2 8 2 8]
 [2 8 8 2]]
Predictions:
[[ True  True False False]]

```

```python
w3_unittest.test_predict(predict)

```

现在让我们将模型学到的决策边界直观地绘制出来。即使不能逐行读懂 `plot_decision_boundary` 函数的内部实现也不必担心——它的核心逻辑只是对二维平面网格上的密集采样点进行批量预测，并将分类结果以等高线图 (Contour Plot) 的形式渲染呈现 (仅有红蓝两色区域) 。

```python
def plot_decision_boundary(predict, parameters, X, Y):
    # 定义绘图区域的数值上下界
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    # 定义 x 与 y 坐标轴上的采样步长
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # 生成平面上的二维网格矩阵
    xx, yy = np.meshgrid(x1grid, x2grid)
    # 将网格平铺展平为向量
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((1, len(r1))), r2.reshape((1, len(r2)))
    # 垂直拼接向量以构成模型的输入特征坐标 (x1, x2)
    grid = np.vstack((r1,r2))
    # 对网格中的所有点进行批量推理预测
    predictions = predict(grid, parameters)
    # 将预测结果重塑还原为网格矩阵的形状
    zz = predictions.reshape(xx.shape)
    # 将 x, y 坐标与对应的预测类别 z 绘制为连续的着色曲面
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral.reversed())
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

# 绘制决策边界
plot_decision_boundary(predict, parameters, X, Y)
plt.title("Decision Boundary for hidden layer size " + str(n_h))

```

效果令人赞叹！可以看到，面对单条直线无能为力的复杂分类问题，仅仅引入一个简单的双层神经网络就能迎刃而解！

## 4 - 选做：探索其他数据集

我们来构建一个分布形态略有不同的小数据集：

```python
n_samples = 2000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0)] = 0
labels[(labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 1
X_2 = np.transpose(samples)
Y_2 = labels.reshape((1,n_samples))

plt.scatter(X_2[0, :], X_2[1, :], c=Y_2, cmap=colors.ListedColormap(['blue', 'red']));

```

细心的你一定注意到了：在构建神经网络模型时，隐藏层中的神经元节点数量是作为一个超参数传入的。不妨动手尝试修改这个参数，观察不同的网络容量会对最终的学习决策效果产生怎样的影响：

```python
# parameters_2 = nn_model(X_2, Y_2, n_h=1, num_iterations=3000, learning_rate=1.2, print_cost=False)
parameters_2 = nn_model(X_2, Y_2, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=False)
# parameters_2 = nn_model(X_2, Y_2, n_h=15, num_iterations=3000, learning_rate=1.2, print_cost=False)

# 调用决策边界可视化函数
plot_decision_boundary(predict, parameters_2, X_2, Y_2)
plt.title("Decision Boundary")

```

从图中可以看出，画面上不可避免地存在少量被误分类的数据点——在真实的客观世界中，数据集往往是线性不可分的，存在微小的误差分布纯属常态。更重要的是，在构建机器学习模型时，我们并不希望它过于严丝合缝、近乎病态地去迎合某一份特定的训练样本——因为这样极其容易丧失对未知全新数据的泛化推断能力。这种令人头疼的机器学习经典困境，正是我们常说的过拟合 (Overfitting) 现象。

热烈祝贺你顺利通关本周的编程作业！

```python


```