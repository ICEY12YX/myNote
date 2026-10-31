# 可选实验：逻辑回归 (Logistic Regression) 的代价函数 (Cost Function) 

## 实验目标
在本实验中，你将：
- 深入研究逻辑回归代价函数的代码实现，并利用它来评估模型。


```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')
```

## 数据集 (Dataset) 
让我们继续使用与决策边界 (decision boundary) 实验中相同的数据集。


```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)
```

我们将使用一个辅助函数 (helper function) 把这些数据绘制成图表。标签 (label) 为 $y=1$ 的数据点显示为红色叉号，而标签为 $y=0$ 的数据点则显示为蓝色圆圈。


```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# 将两个坐标轴的范围都设置为 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
```

## 代价函数

在之前的实验中，你已经推导出了*逻辑损失 (logistic loss) *函数。回想一下，损失是针对单个样本 (single example) 定义的。在这里，我们将所有样本的损失汇总结合起来，从而形成涵盖全部训练样本的**代价 (cost) **。


回想一下，对于逻辑回归，代价函数的数学形式如下：

$$ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$

其中
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ 是单个数据点 (single data point) 的损失，其具体表达式为：

    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
    
*  其中 $m$ 是数据集中训练样本 (training examples) 的数量，且满足：
$$
\begin{align}
  f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\tag{3} \\
  z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\tag{4} \\
  g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}\tag{5} 
\end{align}
$$
 

&lt;a name=&#39;ex-02&#39;&gt;&lt;/a&gt;
#### 代码说明

`compute_cost_logistic` 的算法逻辑是遍历所有训练样本，逐一计算每个样本的损失，并将它们累加得到总和。

请注意，变量 X 和 y 并不是标量值 (scalar values) ，而是形状分别为 ($m, n$) 和 ($𝑚$,) 的矩阵 (matrices) ，其中 $𝑛$ 代表特征 (features) 的数量，$𝑚$ 代表训练样本的数量。



```python
def compute_cost_logistic(X, y, w, b):
    """
    计算逻辑回归的代价

    参数:
      X (ndarray (m,n)): 数据，包含 m 个样本，每个样本有 n 个特征
      y (ndarray (m,)) : 目标值 (target values) 
      w (ndarray (n,)) : 模型参数 (model parameters)   
      b (scalar)       : 模型参数
      
    返回:
      cost (scalar): 代价值
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

```

运行下方的代码单元格，检验一下代价函数的实现是否正确。


```python
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
```

**预期输出**：0.3668667864055175

## 示例分析
现在，让我们来看看当参数取不同的数值时，代价函数会输出怎样的结果。

* 在之前的实验中，你曾绘制过参数为 $b = -3, w_0 = 1, w_1 = 1$ 时的决策边界。也就是说，当时的参数设置为 `b = -3, w = np.array([1,1])`。

* 假设你现在想探究一下，将参数改为 $b = -4, w_0 = 1, w_1 = 1$ (即 `b = -4, w = np.array([1,1])`) 是否能得到一个表现更好的模型。

让我们首先把对应这两个不同 $b$ 值的决策边界绘制出来，直观地对比一下哪一个能更好地拟合数据。

* 对于 $b = -3, w_0 = 1, w_1 = 1$，我们将绘制直线 $-3 + x_0+x_1 = 0$ (图中以蓝色显示) 
* 对于 $b = -4, w_0 = 1, w_1 = 1$，我们将绘制直线 $-4 + x_0+x_1 = 0$ (图中以品红色显示) 


```python
import matplotlib.pyplot as plt

# 在 0 到 6 之间选择数值
x0 = np.arange(0,6)

# 绘制两条决策边界
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# 绘制决策边界
ax.plot(x0,x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# 绘制原始数据点
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()
```

从这幅图中你可以明显看出，对于当前训练数据而言，`b = -4, w = np.array([1,1])` 是一个拟合效果更差的模型。下面让我们看看代价函数的计算结果是否能够印证这一观察。


```python
w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))
```

**预期输出**

Cost for b = -3 :  0.3668667864055175

Cost for b = -4 :  0.5036808636748461


正如你所看到的，代价函数的表现完全符合预期：`b = -4, w = np.array([1,1])` 对应的代价值确实高于 `b = -3, w = np.array([1,1])` 对应的代价值。

## 恭喜！
在本实验中，你已经深入研究并实际运用了逻辑回归的代价函数。


```python

```
