___
# 基于感知机 (Perceptron) 的回归分析

在第 2 周的作业中，你运用梯度下降法 (Gradient Descent Method) 构建了一元线性回归模型 (Linear Regression Model) ，根据电视广告预算成功预测了产品销量。在本次实验中，你将构建一个与之等价的神经网络 (Neural Network) 来实现同样的线性回归模型。接着，你将通过实现梯度下降法来训练这个网络。在此之后，你还会进一步扩展该神经网络的结构，构建出一个多元线性回归 (Multiple Linear Regression) 模型，综合房屋的面积与建造品质来预测房价。

*注意*：在第 1 门课程“线性代数 (Linear Algebra) ”的第 3 周作业中曾探讨过相同的模型，但当时并未涉及利用反向传播 (Backward Propagation) 进行模型训练的内容。

# 目录

* 1 - 一元线性回归
* 1.1 - 一元线性回归模型
* 1.2 - 包含单个感知机与单一输入节点的神经网络模型
* 1.3 - 数据集


* 2 - 线性回归神经网络模型的代码实现
* 2.1 - 定义神经网络结构
* 2.2 - 初始化模型参数
* 2.3 - 训练循环
* 2.4 - 在 nn_model() 中整合前述模块并执行预测


* 3 - 多元线性回归
* 3.1 - 多元线性回归模型
* 3.2 - 包含单个感知机与两个输入节点的神经网络模型
* 3.3 - 数据集
* 3.4 - 多元线性回归神经网络模型的表现



## 依赖库

首先，导入本实验所需的所有依赖库。

```python
import numpy as np
import matplotlib.pyplot as plt
# 用于数据处理与分析的库
import pandas as pd

# 在 Jupyter notebook 内部内嵌展示绘图结果
%matplotlib inline 

# 设置随机种子，确保每次运行的结果一致
np.random.seed(3) 

```

## 1 - 一元线性回归

### 1.1 - 一元线性回归模型

一元线性回归模型可以用如下公式来描述：

$$\hat{y} = wx + b,\tag{1}$$

其中，$\hat{y}$ 是根据自变量 (Independent Variable)  $x$ 对因变量 (Dependent Variable)  $y$ 做出的预测估计，它遵循由斜率 (Slope)  $w$ 和截距 (Intercept)  $b$ 构成的直线方程。

给定一组训练数据点 $(x_1, y_1)$, ..., $(x_m, y_m)$，你的任务是找到一条“最佳拟合线”——即寻找到一组参数 $w$ 与 $b$，使得原始真实值 $y_i$ 与模型预测值 $\hat{y}_i = wx_i + b$ 之间的偏差达到最小。

### 1.2 - 包含单个感知机与单一输入节点的神经网络模型

能够描述上述问题的最简单的神经网络模型，只需由单个感知机 (Perceptron) 构成即可。模型的输入层 (Input Layer) 和输出层 (Output Layer) 各包含一个节点 (Node)  (其中 $x$ 代表输入，$\hat{y} = z$ 代表输出) ：

权重 (Weight)  ($w$) 与偏置 (Bias)  ($b$) 是在模型训练 (Train) 过程中不断被更新的核心参数。它们通常会被赋予一些初始随机值或初始化为 0，并随着训练进程逐步修正。

对于每个训练样本 $x^{(i)}$，预测值 $\hat{y}^{(i)}$ 可以通过以下公式计算：

\begin{align}
z^{(i)} &=  w x^{(i)} + b,\
\hat{y}^{(i)} &= z^{(i)},
\tag{2}\end{align}

其中 $i = 1, \dots, m$。

你可以将所有训练样本打包整理成一个维度为 ($1 \times m$) 的向量 (Vector)  $X$，并对向量 $X$ ($1 \times m$) 与标量 $w$ 进行标量乘法 (Scalar Multiplication) ，随后加上标量 $b$。在计算时，$b$ 会自动被广播 (Broadcast) 扩展为维度为 ($1 \times m$) 的向量：

\begin{align}
Z &=  w X + b,\
\hat{Y} &= Z,
\tag{3}\end{align}

这一整套自输入至输出的运算流程被称为前向传播 (Forward Propagation) 。

对于单个训练样本，你可以使用损失函数 (Loss Function)  $L\left(w, b\right)  = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$ 来量化真实值 $y^{(i)}$ 与预测值 $\hat{y}^{(i)}$ 之间的误差。这里之所以除以 $2$，纯粹是为了后续缩放方便——稍后在求解偏导数 (Partial Derivative) 时你就会体会到它的妙处。为了全面衡量预测向量 $\hat{Y}$ ($1 \times m$) 与真实值向量 $Y$ 之间的差异，我们可以计算所有训练样本损失值的均值：

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.\tag{4}$$

这个函数被称为平方和代价函数 (Cost Function) 。模型训练的核心目标，就是在整个训练过程中不断优化代价函数，将真实值 $y^{(i)}$ 与预测值 $\hat{y}^{(i)}$ 之间的整体差距降到最低。

当权重刚被随机初始化、模型尚未经历任何学习时，我们很难指望它能产生理想的预测结果。因此，必须计算出针对权重和偏置的调整幅度，驱动代价函数持续下降。这一计算梯度的反向回溯过程被称为反向传播 (Backward Propagation) 。

依据梯度下降算法，代价函数对各参数的偏导数可表示为：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial w } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)},\
\frac{\partial \mathcal{L} }{ \partial b } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
\tag{5}\end{align}

可以看到，公式 $(4)$ 中引入的系数 $\frac{1}{2}$ 正好抵消了求导产生的常数 2，从而让偏导数的表达变得格外简洁整齐。接下来，利用求出的梯度以迭代形式更新参数：

\begin{align}
w &= w - \alpha \frac{\partial \mathcal{L} }{ \partial w },\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{6}\end{align}

这里的 $\alpha$ 代表学习率 (Learning Rate) 。随后不断重复该步骤，直至代价函数收敛并趋于稳定。

构建神经网络的通用方法包含以下四个环节：

1. 定义神经网络的整体拓扑结构 (输入单元数、隐藏单元数等) 。
2. 初始化模型参数。
3. 循环迭代：
* 执行前向传播 (计算感知机输出) ；
* 执行反向传播 (获取参数更新所需的梯度修正值) ；
* 更新各项参数。


4. 应用模型进行推理预测。

在实际工程中，我们通常会先编写辅助函数分别实现步骤 1 至 3，再将它们集成到主函数 `nn_model()` 中。一旦 `nn_model()` 构建完成并学得理想参数，便可以将其直接运用于新数据的预测。

### 1.3 - 数据集

载入保存在文件 `data/tvmarketing.csv` 中的 Kaggle 数据集。该数据集包含两个字段：电视广告费用 (`TV`) 与实际销售额 (`Sales`)。

```python
path = "data/tvmarketing.csv"

adv = pd.read_csv(path)

```

查看数据集的部分样本数据：

```python
adv.head()

```

绘制数据散点图：

```python
adv.plot(x='TV', y='Sales', kind='scatter', c='black')

```

字段 `TV` 与 `Sales` 的量纲和取值范围相差悬殊。回想第 2 周作业中的要点：为了确保梯度下降算法高效稳定地运行，需要对特征进行归一化 (Normalization) 处理——即从数组中的每个元素中减去该数组的均值，再除以其标准差 (Standard Deviation) 。

我们可以针对数据集的所有列同步实施列级归一化，具体实现代码如下：

```python
adv_norm = (adv - np.mean(adv))/np.std(adv)

```

绘制归一化后的数据散点图。可以看出数据整体分布形态与此前一致，但坐标轴上的数值范围已被统一缩放到零均值和单位方差附近：

```python
adv_norm.plot(x='TV', y='Sales', kind='scatter', c='black')

```

将数据保存到变量 `X_norm` 和 `Y_norm` 中，并重塑转换为行向量：

```python
X_norm = adv_norm['TV']
Y_norm = adv_norm['Sales']

X_norm = np.array(X_norm).reshape((1, len(X_norm)))
Y_norm = np.array(Y_norm).reshape((1, len(Y_norm)))

print ('The shape of X_norm: ' + str(X_norm.shape))
print ('The shape of Y_norm: ' + str(Y_norm.shape))
print ('I have m = %d training examples!' % (X_norm.shape[1]))

```

## 2 - 线性回归神经网络模型的代码实现

我们将以高度模块化、通用的方式搭建该神经网络，以便后续能将这种“单感知机+单输入节点”的简单架构平滑拓展至更为复杂的多维网络结构中。

### 2.1 - 定义神经网络结构

根据数组 `X` 和 `Y` 的形状定义两个变量：

* `n_x`：输入层的神经元数量
* `n_y`：输出层的神经元数量

```python
def layer_sizes(X, Y):
    """
    参数:
    X -- 输入数据集，形状为 (输入特征维度, 样本数量)
    Y -- 标签数组，形状为 (输出标签维度, 样本数量)
    
    返回值:
    n_x -- 输入层的大小
    n_y -- 输出层的大小
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

(n_x, n_y) = layer_sizes(X_norm, Y_norm)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))

```

### 2.2 - 初始化模型参数

实现函数 `initialize_parameters()`，将形状为 $(n_y \times n_x) = (1 \times 1)$ 的权重矩阵初始化为微小的随机数，将形状为 $(n_y \times 1) = (1 \times 1)$ 的偏置向量初始化为全零。

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

### 2.3 - 训练循环

根据第 1.2 节中的公式 $(3)$ 实现 `forward_propagation()`：
\begin{align}
Z &=  w X + b\
\hat{Y} &= Z,
\end{align}

```python
def forward_propagation(X, parameters):
    """
    参数:
    X -- 维度为 (n_x, m) 的输入数据
    parameters -- 包含参数的 Python 字典 (初始化函数的输出结果)
    
    返回值:
    Y_hat -- 模型的预测输出
    """
    W = parameters["W"]
    b = parameters["b"]
    
    # 执行前向传播计算 Z
    Z = np.matmul(W, X) + b
    Y_hat = Z

    return Y_hat

Y_hat = forward_propagation(X_norm, parameters)

print("Some elements of output vector Y_hat:", Y_hat[0, 0:5])

```

由于各权重刚刚被赋予随机值，模型目前尚未经过任何训练。

接下来定义用于驱动模型学习的代价函数 $(4)$：

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

```python
def compute_cost(Y_hat, Y):
    """
    计算基于误差平方和的代价函数值
    
    参数:
    Y_hat -- 神经网络的输出预测值，形状为 (n_y, 样本数量)
    Y -- 真实标签向量，形状为 (n_y, 样本数量)
    
    返回值:
    cost -- 按 1/(2*样本数量) 缩放后的平方误差总和
    
    """
    # 样本数量
    m = Y_hat.shape[1]

    # 计算代价函数
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

print("cost = " + str(compute_cost(Y_hat, Y_norm)))

```

依照公式 $(5)$ 计算代价函数的偏导数：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial w } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)},\
\frac{\partial \mathcal{L} }{ \partial b } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
\end{align}

```python
def backward_propagation(Y_hat, X, Y):
    """
    实现反向传播算法，计算各参数的梯度
    
    参数:
    Y_hat -- 神经网络的输出预测值，形状为 (n_y, 样本数量)
    X -- 输入数据，形状为 (n_x, 样本数量)
    Y -- 真实标签向量，形状为 (n_y, 样本数量)
    
    返回值:
    grads -- 包含对各个参数所求偏导数的 Python 字典
    """
    m = X.shape[1]
    
    # 反向传播: 计算偏导数，为简便起见记为 dW 和 db 
    dZ = Y_hat - Y
    dW = 1/m * np.dot(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

grads = backward_propagation(Y_hat, X_norm, Y_norm)

print("dW = " + str(grads["dW"]))
print("db = " + str(grads["db"]))

```

按照公式 $(6)$ 更新模型参数：

\begin{align}
w &= w - \alpha \frac{\partial \mathcal{L} }{ \partial w },\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b }.
\end{align}

```python
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    基于梯度下降更新规则对模型参数进行迭代更新
    
    参数:
    parameters -- 包含待更新参数的 Python 字典 
    grads -- 包含梯度计算结果的 Python 字典 
    learning_rate -- 梯度下降算法的学习率参数
    
    返回值:
    parameters -- 包含更新后参数的 Python 字典 
    """
    # 从字典 "parameters" 中提取参数
    W = parameters["W"]
    b = parameters["b"]
    
    # 从字典 "grads" 中提取梯度
    dW = grads["dW"]
    db = grads["db"]
    
    # 各参数的梯度更新公式
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

parameters_updated = update_parameters(parameters, grads)

print("W updated = " + str(parameters_updated["W"]))
print("b updated = " + str(parameters_updated["b"]))

```

### 2.4 - 在 nn_model() 中整合前述模块并执行预测

在 `nn_model()` 中组装完整的神经网络模型。

```python
def nn_model(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    参数:
    X -- 数据集，形状为 (n_x, 样本数量)
    Y -- 标签，形状为 (n_y, 样本数量)
    num_iterations -- 训练循环迭代的次数
    learning_rate -- 梯度下降算法的学习率参数
    print_cost -- 若设为 True，则每次迭代后打印当前代价值
    
    返回值:
    parameters -- 模型学习得到的最终参数，可直接用于后续推理预测
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)
    
    # 训练主循环
    for i in range(0, num_iterations):
         
        # 前向传播。输入: "X, parameters"，输出: "Y_hat"
        Y_hat = forward_propagation(X, parameters)
        
        # 代价函数。输入: "Y_hat, Y"，输出: "cost"
        cost = compute_cost(Y_hat, Y)
        
        # 反向传播。输入: "Y_hat, X, Y"，输出: "grads"
        grads = backward_propagation(Y_hat, X, Y)
    
        # 梯度下降更新参数。输入: "parameters, grads, learning_rate"，输出: "parameters"
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 每次迭代打印当代价值
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

```

```python
parameters_simple = nn_model(X_norm, Y_norm, num_iterations=30, learning_rate=1.2, print_cost=True)
print("W = " + str(parameters_simple["W"]))
print("b = " + str(parameters_simple["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]

```

可以看到，在经过短短几次迭代之后，代价函数的值便不再显著下降 (模型实现了收敛) 。

*注意*：这是一个极度简化的基础模型。在真实的工业与科研任务中，复杂模型收敛所需的时间和迭代次数远非如此迅速。

最终训练得到的模型参数可以用于对新数据做出预测。但切记：必须配合归一化以及反归一化流程进行对应变换。

```python
def predict(X, Y, parameters, X_pred):
    
    # 从字典 "parameters" 中提取学得的参数
    W = parameters["W"]
    b = parameters["b"]
    
    # 采用与原训练集 X 完全相同的均值和标准差
    if isinstance(X, pd.Series):
        X_mean = np.mean(X)
        X_std = np.std(X)
        X_pred_norm = ((X_pred - X_mean)/X_std).reshape((1, len(X_pred)))
    else:
        X_mean = np.array(np.mean(X)).reshape((len(X.axes[1]),1))
        X_std = np.array(np.std(X)).reshape((len(X.axes[1]),1))
        X_pred_norm = ((X_pred - X_mean)/X_std)
    # 计算预测值
    Y_pred_norm = np.matmul(W, X_pred_norm) + b
    # 结合原训练集 Y 的均值和标准差恢复至原始物理量纲 (反归一化)
    Y_pred = Y_pred_norm * np.std(Y) + np.mean(Y)
    
    return Y_pred[0]

X_pred = np.array([50, 120, 280])
Y_pred = predict(adv["TV"], adv["Sales"], parameters_simple, X_pred)
print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales:\n{Y_pred}")

```

现在让我们绘制出线性回归拟合直线以及预测样本点。红线代表拟合直线，蓝点代表模型预测出的数据位置。

```python
fig, ax = plt.subplots()
plt.scatter(adv["TV"], adv["Sales"], color="black")

plt.xlabel("$x$")
plt.ylabel("$y$")
    
X_line = np.arange(np.min(adv["TV"]),np.max(adv["TV"])*1.1, 0.1)
Y_line = predict(adv["TV"], adv["Sales"], parameters_simple, X_line)
ax.plot(X_line, Y_line, "r")
ax.plot(X_pred, Y_pred, "bo")
plt.plot()
plt.show()

```

接下来，我们增加输入节点的数量，正式迈入多元线性回归的世界。

## 3 - 多元线性回归

### 3.1 - 多元线性回归模型

包含两个自变量 $x_1$ 和 $x_2$ 的多元线性回归模型可以表述为：

$$\hat{y} = w_1x_1 + w_2x_2 + b = Wx + b,\tag{7}$$

其中，$Wx$ 是输入特征向量 $x = \begin{bmatrix} x_1 & x_2\end{bmatrix}$ 与参数权重向量 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 的点积 (Dot Product) ，标量参数 $b$ 是截距项。模型训练的目的，依然是为给定的训练样本寻找一组“最理想”的参数 $w_1$、$w_2$ 和 $b$，使真实值 $y_i$ 与模型预测值 $\hat{y}_i$ 的整体误差最小化。

### 3.2 - 包含单个感知机与两个输入节点的神经网络模型

为了解决多元回归问题，我们依然可以使用单感知机模型，但需要引入两个输入节点，结构如下图所示：

对于每个输入样本 $x^{(i)} = \begin{bmatrix} x_1^{(i)} & x_2^{(i)}\end{bmatrix}$，感知机的输出可以用点积写为：

$$z^{(i)} = w_1x_1^{(i)} + w_2x_2^{(i)} + b = Wx^{(i)} + b,\tag{8}$$

这里的各个权重被汇总放置在向量 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 中，偏置 $b$ 为标量。输出层结构保持不变，依然是产生连续值的单个输出节点 $\hat{y} = z$。

将所有训练样本按列排布整理成一个形状为 ($2 \times m$) 的输入矩阵 $X$，其中第一行与第二行分别存放 $x_1^{(i)}$ 和 $x_2^{(i)}$。通过将参数矩阵 $W$ ($1 \times 2$) 与输入矩阵 $X$ ($2 \times m$) 进行矩阵乘法 (Matrix Multiplication) ，即可得到一个 ($1 \times m$) 维的向量：

$$WX =  \begin{bmatrix} w_1 & w_2\end{bmatrix}  \begin{bmatrix}  x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(m)} \\  x_2^{(1)} & x_2^{(2)} & \dots & x_2^{(m)} \\ \end{bmatrix} =\begin{bmatrix}  w_1x_1^{(1)} + w_2x_2^{(1)} &  w_1x_1^{(2)} + w_2x_2^{(2)} & \dots &  w_1x_1^{(m)} + w_2x_2^{(m)}\end{bmatrix}.$$

于是整个模型的前向传播过程可以简洁地写为：

\begin{align}
Z &=  W X + b,\
\hat{Y} &= Z,
\tag{9}\end{align}

其中偏置 $b$ 借助广播机制自动扩展为 ($1 \times m$) 维向量。此时代价函数的形式保持不变 (参见第 1.2 节中的公式 $(4)$) ：

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

在实现梯度下降时，代价函数针对各个参数的偏导数形式如下：

\begin{align}
\frac{\partial \mathcal{L} }{ \partial w_1 } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x_1^{(i)},\
\frac{\partial \mathcal{L} }{ \partial w_2 } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)x_2^{(i)},\tag{10}\
\frac{\partial \mathcal{L} }{ \partial b } &=
\frac{1}{m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right).
\end{align}

在前向传播公式 $(9)$ 运算完成后，变量 $\hat{Y}$ 保存着维度为 ($1 \times m$) 的预测值数组，而真实标签 $y^{(i)}$ 则保存在相同维度的数组 $Y$ 中。因此，差值项 $\left(\hat{Y} - Y\right)$ 构成一个记录着所有误差 $\left(\hat{y}^{(i)} - y^{(i)}\right)$ 的 ($1 \times m$) 数组。由于特征矩阵 $X$ ($2 \times m$) 的第一行是所有样本的 $x_1^{(i)}$、第二行是所有样本的 $x_2^{(i)}$，公式 $(10)$ 中关于权重的两个求和式可以直接通过误差矩阵 $\left(\hat{Y} - Y\right)$ ($1 \times m$) 与转置矩阵 $X^T$ ($m \times 2$) 的矩阵乘法一次性完成，得到形状为 ($1 \times 2$) 的梯度矩阵：

$$\frac{\partial \mathcal{L} }{ \partial W } =  \begin{bmatrix} \frac{\partial \mathcal{L} }{ \partial w_1 } &  \frac{\partial \mathcal{L} }{ \partial w_2 }\end{bmatrix} = \frac{1}{m}\left(\hat{Y} - Y\right)X^T.\tag{11}$$

同理，对于偏置项的偏导数 $\frac{\partial \mathcal{L} }{ \partial b }$，亦可通过全 1 向量 $\mathbf{1}$ ($m \times 1$) 紧凑表达：

$$\frac{\partial \mathcal{L} }{ \partial b } = \frac{1}{m}\left(\hat{Y} - Y\right)\mathbf{1}.\tag{12}$$

瞧，线性代数与微积分的完美融合，让原本冗长繁琐的逐点累加计算变得如此优雅而高效！现在，我们可以借助矩阵化形式直接更新参数 $W$：

\begin{align}
W &= W - \alpha \frac{\partial \mathcal{L} }{ \partial W },\
b &= b - \alpha \frac{\partial \mathcal{L} }{ \partial b },
\tag{13}\end{align}

其中 $\alpha$ 为学习率。我们只需将这一计算过程放入循环中不断迭代，直至代价函数平稳收敛。

### 3.3 - 数据集

接下来，我们针对 Kaggle 上的房价数据集构建一个多元线性回归模型，数据保存在文件 `data/house_prices_train.csv` 中。我们将选取地面居住面积 (`GrLivArea`，单位：平方英尺) 以及整体用料与完工品质评级 (`OverallQual`，评分区间 1-10) 这两个特征，用来预测房屋售价 (`SalePrice`，单位：美元)。

利用 `pandas` 的 `read_csv` 函数读取该数据集：

```python
df = pd.read_csv('data/house_prices_train.csv')

```

筛选所需特征字段，并存入变量 `X_multi` 和 `Y_multi`：

```python
X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']

```

预览数据内容：

```python
display(X_multi)
display(Y_multi)

```

对数据执行归一化：

```python
X_multi_norm = (X_multi - np.mean(X_multi))/np.std(X_multi)
Y_multi_norm = (Y_multi - np.mean(Y_multi))/np.std(Y_multi)

```

将处理好的数据转换为 NumPy 数组，转置 `X_multi_norm` 得到形状为 ($2 \times m$) 的特征矩阵，并将 `Y_multi_norm` 重塑转换为 ($1 \times m$) 的行向量：

```python
X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))

```

### 3.4 - 多元线性回归神经网络模型的表现

见证奇迹的时刻到了——你完全不需要对先前编写的神经网络代码做任何修改！回顾第 2 节的代码实现：只要传入新的多维输入数据集 `X_multi_norm` 和 `Y_multi_norm`，输入层节点数 $n_x$ 就会自动识别并设为 $2$，而其余所有的核心计算模块——甚至是反向传播算法——都能完美无缝复用！

让模型执行 $100$ 次迭代训练：

```python
parameters_multi = nn_model(X_multi_norm, Y_multi_norm, num_iterations=100, print_cost=True)

print("W = " + str(parameters_multi["W"]))
print("b = " + str(parameters_multi["b"]))

W_multi = parameters_multi["W"]
b_multi = parameters_multi["b"]

```

现在，模型已经整装待发，可以开始进行新样本的售价预测了：

```python
X_pred_multi = np.array([[1710, 7], [1200, 6], [2200, 8]]).T
Y_pred_multi = predict(X_multi, Y_multi, parameters_multi, X_pred_multi)

print(f"Ground living area, square feet:\n{X_pred_multi[0]}")
print(f"Rates of the overall quality of material and finish, 1-10:\n{X_pred_multi[1]}")
print(f"Predictions of sales price, $:\n{np.round(Y_pred_multi)}")

```

恭喜你，圆满完成本次实验！

```python


```