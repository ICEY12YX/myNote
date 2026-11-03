___
- [[#可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary)|可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary)]]
	- [[#可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary)#实验目标|实验目标]]
	- [[#可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary)#数据集 (Dataset)|数据集 (Dataset)]]
		- [[#数据集 (Dataset)#绘制数据图表|绘制数据图表]]
	- [[#可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary)#逻辑回归模型|逻辑回归模型]]
		- [[#逻辑回归模型#温习逻辑回归与决策边界|温习逻辑回归与决策边界]]
		- [[#逻辑回归模型#绘制决策边界|绘制决策边界]]
	- [[#可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary)#恭喜！|恭喜！]]

___
# 可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary) 

## 实验目标

在本实验中，你将：
- 绘制逻辑回归模型 (logistic regression model) 的决策边界。这将帮助你更直观地理解模型是如何做出预测的。

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')
```

## 数据集 (Dataset) 

假设你拥有以下训练数据集 (training dataset) 
- 输入变量 (input variable)  `X` 是一个 numpy 数组 (numpy array) ，包含 6 个训练样本 (training examples) ，==每个样本具有两个特征 (features) ==
- 输出变量 (output variable)  `y` 也是一个包含 6 个样本的 numpy 数组，且 `y` 的取值只能是 `0` 或 `1`


```python
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
```

### 绘制数据图表

让我们使用一个辅助函数 (helper function) 将这些数据绘制出来。标签 (label) 为 $y=1$ 的数据点显示为红色叉号，而标签为 $y=0$ 的数据点则显示为蓝色圆圈。


```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()
```
![[c1_w3_lab03_decision_boundary_soln-1793649098236.webp|307]]


## 逻辑回归模型

* 假设你想在这些数据上训练一个逻辑回归模型，其数学形式如下：
  $f(x) = g(w_0x_0+w_1x_1 + b)$
  其中 $g(z) = \frac{1}{1+e^{-z}}$，即 S型函数 (sigmoid function) 


* 假设你已经完成了模型训练，并求得参数 (parameters) 为 $b = -3,   w_0 = 1,   w_1 = 1$。也就是说，
  $f(x) = g(x_0+x_1-3)$
   (你将在后续课程中进一步学习如何根据数据拟合这些参数) 
  
下面让我们通过绘制决策边界，来深入理解这个训练好的模型是如何进行预测的。

### 温习逻辑回归与决策边界

* 回想一下，对于逻辑回归，模型的表达式为：
  $$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \tag{1}$$

  其中 $g(z) = \frac{1}{1+e^{-z}}\tag{2}$ 被称为 S型函数，它将所有输入值映射到 0 到 1 之间的数值：

  而 $\mathbf{w} \cdot \mathbf{x}$ 表示向量点积 (vector dot product) ：
  $$\mathbf{w} \cdot \mathbf{x} = w_0 x_0 + w_1 x_1$$
  
 * 我们将模型的输出 ($f_{\mathbf{w},b}(x)$) 解释为：在给定输入 $\mathbf{x}$ 且以 $\mathbf{w}$ 和 $b$ 作为参数的条件下，$y=1$ 的**概率 (probability)** 
 
* 因此，为了从逻辑回归模型中获得最终的分类预测结果 ($y=0$ 或 $y=1$) ，我们可以采用以下启发式规则 (heuristic) 

  如果 $f_{\mathbf{w},b}(x) >= 0.5$，则预测 $y=1$
  如果 $f_{\mathbf{w},b}(x) < 0.5$，则预测 $y=0$
  
  
* 让我们绘制出 S型函数的图像，看看在什么范围内满足 $g(z) >= 0.5$

```python
# 在 -10 到 10 的取值范围内绘制 sigmoid(z) 图像
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# 绘制 z 与 sigmoid(z) 的关系曲线
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
```
![[c1_w3_lab03_decision_boundary_soln-1793649279352.webp|446]]

* 正如你所看到的，当 $z >=0$ 时，恒有 $g(z) >= 0.5$

* 对于逻辑回归模型而言，$z = \mathbf{w} \cdot \mathbf{x} + b$。因此，

  如果 $\mathbf{w} \cdot \mathbf{x} + b >= 0$，模型就会预测 $y=1$
  如果 $\mathbf{w} \cdot \mathbf{x} + b < 0$，模型就会预测 $y=0$
  
### 绘制决策边界

现在，让我们回到前面的示例，看看逻辑回归模型具体是如何做出预测的。

* 我们的逻辑回归模型具有以下形式：

  $f(\mathbf{x}) = g(-3 + x_0+x_1)$

* 根据你刚才学到的规律可以看出，当 $-3 + x_0+x_1 >= 0$ 时，该模型会预测 $y=1$

让我们来看看它在图形上是什么样子的。我们首先绘制直线 $-3 + x_0+x_1 = 0$，它等价于 $x_1 = 3 - x_0$。

```python
# 在 0 到 6 之间选择数值
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# 绘制决策边界
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# 填充直线下方的区域
ax.fill_between(x0,x1, alpha=0.2)

# 绘制原始数据点
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
```
![[c1_w3_lab03_decision_boundary_soln-1793649361478.webp|429]]
* 在上图中，蓝色直线代表方程 $x_0 + x_1 - 3 = 0$，它与 x1 轴相交于 3 (令 $x_1$ = 3，$x_0$ = 0) ，与 x0 轴也相交于 3 (令 $x_1$ = 0，$x_0$ = 3) 。

* 阴影区域代表 $-3 + x_0+x_1 < 0$。直线上方的区域则代表 $-3 + x_0+x_1 > 0$。
    这个上方小于0,还是大于0, 可以带特殊值看  比如(0,0)

* 任何位于阴影区域内 (直线下方) 的点都会被分类为 $y=0$。任何位于直线上或直线上方的点都会被分类为 $y=1$。==**这条分界线就被称为“决策边界”。**==
    ==**所以记住是这条线  就是没有塞进 sigmoid之前的fx  是决策边界**==


正如我们在课程讲座中所了解到的，通过引入更高阶的多项式项 (polynomial terms)  (例如：$f(x) = g( x_0^2 + x_1 -1)$) ，我们可以==构建出更加复杂的非线性边界== (non-linear boundaries) 。

## 恭喜！
你已经深入探索了逻辑回归中的决策边界概念。

