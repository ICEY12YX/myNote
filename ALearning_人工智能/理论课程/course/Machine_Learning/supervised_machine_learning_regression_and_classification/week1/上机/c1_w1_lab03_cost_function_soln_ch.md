
___
- [[#选修实验：代价函数 (Cost Function)|选修实验：代价函数 (Cost Function)]]
	- [[#选修实验：代价函数 (Cost Function)#实验目标|实验目标]]
	- [[#选修实验：代价函数 (Cost Function)#实验工具|实验工具]]
	- [[#选修实验：代价函数 (Cost Function)#问题背景|问题背景]]
	- [[#选修实验：代价函数 (Cost Function)#计算代价 (Computing Cost)|计算代价 (Computing Cost)]]
	- [[#选修实验：代价函数 (Cost Function)#代价函数的直观理解|代价函数的直观理解]]
	- [[#选修实验：代价函数 (Cost Function)#代价函数的三维可视化 - 3D|代价函数的三维可视化 - 3D]]
		- [[#代价函数的三维可视化 - 3D#更大规模的数据集|更大规模的数据集]]
		- [[#代价函数的三维可视化 - 3D#碗状的凸代价曲面 (Convex Cost surface)|碗状的凸代价曲面 (Convex Cost surface)]]
- [[#恭喜！|恭喜！]]

___

# 选修实验：代价函数 (Cost Function) 

![[c1_w1_lab03_cost_function_soln_ch-1793362939735.webp]]

## 实验目标
在本次实验中，你将：
- 亲手用代码实现并深入探索单变量线性回归 (linear regression with one variable) 的 代价 (cost) 函数。

## 实验工具
在本次实验中，我们将用到以下工具：
- NumPy，一个广受欢迎的科学计算 (scientific computing) 库
- Matplotlib，一个广受欢迎的数据可视化绘图 (plotting data) 库
- 当前目录下 lab_utils_uni.py 文件中预先封装好的本地绘图函数

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')
```


## 问题背景

假设你希望构建一个机器学习模型 (model) ，能够根据房屋的面积来预测房价。  
继续沿用上一个实验中的那两个数据点 (data points) —— 一套面积为 1000 平方英尺的房子售价为 \$300,000，另一套面积为 2000 平方英尺的房子售价为 \$500,000。

| 面积 (1000 平方英尺)     | 价格 (千美元) |
| -------------------| ------------------------ |
| 1                 | 300                      |
| 2                  | 500                      |

```python
x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)
```

## 计算代价 (Computing Cost) 
在这个练习中，“代价” (cost) 这个词起初可能会让你觉得有些容易混淆，因为我们处理的数据本身就和买房的花费有关。但在机器学习里，“代价”是一个专门用来衡量模型预测得有多准的指标，==它反映了模型给出的预测房价与真实目标价格 (target price) 之间的差距。==而对于房子本身的价值，我们统一使用“价格” (price) 一词来表示。

单变量线性回归的代价方程如下：
      $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 因为i从0开始,所以到m-1结束
 
其中
  $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$
  
- $f_{w,b}(x^{(i)})$ 是模型使用参数 (parameters) $w,b$ 对第 $i$ 个样本给出的预测值 (prediction) 。  
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ 是真实目标值 (target value) 与预测值之差的平方，也就是平方误差 (squared difference) 。   
- 我们将全部 $m$ 个样本的平方误差累加起来，再除以 `2m`，就得到了最终的代价 $J(w,b)$。 

>**注意**: 在理论课讲义中，求和公式的下标范围通常从 1 到 m；而在编写代码时，索引范围则是从 0 到 m-1。


下方的代码通过循环遍历每一个训练样本 (training examples) 来计算总代价。在每一轮循环中：
- 先计算出当前样本的预测值 `f_wb`
- 再算出真实目标值与预测值之间的差值，并对其求平方。
- 最后将这一项累加到总代价中。

```python
def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost
```

## 代价函数的直观理解

![[c1_w1_lab03_cost_function_soln_ch-1793363642803.webp|466]]
你的目标是找到一个带有参数 $w,b$ 的模型 $f_{w,b}(x) = wx + b$，使它能够根据给定的输入特征 $x$ 准确预测出房屋的价格。而代价正是用来衡量这个模型在训练数据 (training data) 上表现有多精准的标尺。

从上面的代价方程 (1) 可以看出，只要我们能恰当地选出参数 $w$ 和 $b$，让模型的预测值 $f_{w,b}(x)$ 与真实目标数据 $y$ 完全吻合，那么 $(f_{w,b}(x^{(i)}) - y^{(i)})^2$ 这一项就会变成 0，此时总代价也就降到了最低。在目前这个只有两个数据点的简单例子里，我们完全可以做到这一点！
    注意代价函数是0已经是最低了,因为平方过了,所以不可能是负数
    并且是负数, 用意义理解也说不通

在上一个实验中，你已经发现当 $b=100$ 时能够得到最优解 (optimal solution) ，因此这里我们先将 $b$ 固定为 100，把注意力集中在如何寻找最佳的 $w$ 上。

在下方图表中，请拖动滑块控件来调整 $w$ 的数值，从而让代价降到最低。图表的刷新可能需要几秒钟的时间。

```python
plt_intuition(x_train,y_train)
```
![[c1_w1_lab03_cost_function_soln_ch-1793363978507.webp]]
![[c1_w1_lab03_cost_function_soln_ch-1793364027754.webp]]
![[c1_w1_lab03_cost_function_soln_ch-1793364047777.webp]]
![[c1_w1_lab03_cost_function_soln_ch-1793364064742.webp]]

从这张图表中，我们可以观察到几个非常关键的现象：
- 当 $w = 200$ 时，代价达到了最小值，这与我们在上一个实验中得出的结论完全一致
- 由于代价方程对目标值与预测值之间的误差进行了平方运算，因此一旦 $w$ 的取值偏大或偏小，总代价都会急剧飙升。
- 当我们使用通过最小化代价 (minimizing cost) 选出的 `w` 和 `b` 时，得到的直线就能严丝合缝地拟合所有数据点。

## 代价函数的三维可视化 - 3D

通过绘制三维 (3D) 曲面图或等高线图 (contour plot) ，你可以直观地看到代价是如何随着 `w` 和 `b` 两个参数同时变化而起伏的。   
值得一提的是，本课程中用到的部分绘图代码逻辑相当复杂。我们已经为你提前封装好了这些绘图函数，虽然顺着代码读一遍有助于熟悉数据可视化的技巧，但这并不是顺利完成本课程的硬性要求。相关的绘图函数都保存在本地目录的 lab_utils_uni.py 文件中。

### 更大规模的数据集
接下来，我们不妨看一个包含更多数据点的场景，从中能获得不少新启发。
在这个数据集 (data set) 中，各个数据点并不完全落在同一条直线上。这对代价方程来说意味着什么呢？==我们还能找到一组让代价完全等于 0 的 $w$ 和 $b$ 吗？==

```python
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])
```

在下方的等高线图中，请用鼠标点击图中的某个位置，选出一组能让代价降到最低的 `w` 和 `b`。你可以借助图中的等高线圆环来引导你的选择。注意，图形更新可能需要几秒钟的时间。

```python
plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
```
![[c1_w1_lab03_cost_function_soln_ch-1793364172291.webp|529]]
![[c1_w1_lab03_cost_function_soln_ch-1793364210199.webp|530]]
![[c1_w1_lab03_cost_function_soln_ch-1793364239266.webp|530]]
![[c1_w1_lab03_cost_function_soln_ch-1793364262081.webp|523]]

请留意上方左侧图表中的那些虚线。每一条虚线都代表了训练集 (training set) 中某个样本点各自贡献的那部分代价。
在这个例子里，当参数大约取 $w=209$ 和 $b=2.4$ 时，可以获得较低的代价。需要注意的是，==由于这里的训练样本并不严格排成一条直线，因此哪怕到了最低点，最小代价也不会等于 0。==


### 碗状的凸代价曲面 (Convex Cost surface) 

==正是因为代价函数对误差损失 (loss) 进行了平方处理，才保证了最终形成的“误差曲面” (error surface) 像一只汤碗一样，是一个标准的**凸曲面 (convex surface)** 。==这样的曲面永远存在一个全局最低点，我们只要沿着各个维度 (dimensions) 上的梯度 (gradient) 方向一路向下走，就一定能到达这个谷底。在刚才那张图中，==由于 $w$ 和 $b$ 两个维度的数值比例尺相差较大，这种碗状结构看起来还不算太明显。==而在下面这张同样在理论课上展示过的三维图中，$w$ 和 $b$ 的尺度是对称的 (symmetric) ，汤碗的形状就一目了然了：

```python
soup_bowl()
```
![[c1_w1_lab03_cost_function_soln_ch-1793364391003.webp|427]]
# 恭喜！
通过本次实验，你已经掌握了以下核心概念：
 - 代价方程能够定量衡量模型的预测结果与真实训练数据之间的契合程度。
 - 通过最小化代价函数，我们可以找到模型参数 $w$ 和 $b$ 的最优值 (optimal values) 。

