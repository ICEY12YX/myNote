# 可选实验：逻辑回归 (Logistic Regression) 与逻辑损失 (Logistic Loss) 

在这个非计分实验中，你将：
- 探究为什么平方误差损失 (squared error loss) 不适用于逻辑回归
- 深入探索逻辑损失函数 (logistic loss function) 


```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')
```

## 逻辑回归可以使用平方误差吗？
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_SqErrorVsLogistic.png&quot;     style=&quot; width:400px; padding: 10px; &quot; &gt; 回想一下，在**线性**回归 (Linear Regression) 中，我们曾使用过**平方误差代价函数 (squared error cost function) **：
单变量平方误差代价的数学方程为：
  $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
 
其中 
  $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$


回想一下，平方误差代价具有一个非常优良的性质：顺着代价的导数 (derivative) 方向更新参数，就能顺利找到全局最小值 (minimum) 。


```python
soup_bowl()
```

这个代价函数在线性回归中表现得非常出色，因此我们很自然地会考虑将它也应用到逻辑回归中。然而，正如上方幻灯片所指出的，$f_{wb}(x)$ 现在包含了一个非线性组件 (non-linear component) ，也就是 S型函数 (sigmoid function) ：$f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$。下面让我们在之前实验的示例数据上尝试使用平方误差代价，不过这次要在模型中加入 S型函数。

以下是我们的训练数据 (training data) ：


```python
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)
```

现在，让我们使用*平方误差代价*来绘制代价的曲面图 (surface plot) ：
  $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$ 
 
其中 
  $$f_{w,b}(x^{(i)}) = sigmoid(wx^{(i)} + b )$$



```python
plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()
```

虽然这生成了一幅相当有趣的图像，但上方的曲面远没有线性回归中的“汤碗 (soup bowl) ”那样平滑！

逻辑回归需要一个更契合其非线性本质的代价函数。而构建代价函数的第一步是定义损失函数 (Loss function) ，下面我们将详细介绍。

## 逻辑损失函数
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_LogisticLoss_a.png&quot;     style=&quot; width:250px; padding: 2px; &quot; &gt;
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_LogisticLoss_b.png&quot;     style=&quot; width:250px; padding: 2px; &quot; &gt;
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_LogisticLoss_c.png&quot;     style=&quot; width:250px; padding: 2px; &quot; &gt; 

逻辑回归采用了一种专门针对分类任务设计的损失函数，在这类任务中，目标值 (target) 只能是 0 或 1，而不再是任意实数。

>**定义说明：** 在本课程中，我们采用如下定义区分这两个概念：  
**损失 (Loss) **用于衡量单个样本 (single example) 的预测值与其目标值之间的差距，而  
**代价 (Cost) **则用于衡量整个训练集 (training set) 上所有样本损失的综合水平


具体定义如下：
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ 表示单个数据点 (single data point) 的损失，其表达式为：

\begin{equation}
  loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = \begin{cases}
    - \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=1$}\\
    - \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) & \text{if $y^{(i)}=0$}
  \end{cases}
\end{equation}


*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ 是模型的预测值 (prediction) ，而 $y^{(i)}$ 则是真实的目标值。

*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot\mathbf{x}^{(i)}+b)$，其中函数 $g$ 代表 S型函数。
* 符号约定：`log` 表示自然对数 (natural logarithm) 。

该损失函数最显著的特征在于它使用了两条独立的曲线：一条对应目标值为零 ($y=0$) 的情况，另一条对应目标值为一 ($y=1$) 的情况。这两条曲线结合在一起，恰好提供了损失函数所需的理想特性：当预测值与目标值一致时损失为零；而当预测值偏离目标值时，损失值会迅速增大。请观察下方的曲线图：


```python
plt_two_logistic_loss_curves()
```

将这两条曲线结合起来看，其走势与平方误差损失的二次曲线 (quadratic curve) 颇为相似。请注意，图中的横轴是 $f_{\mathbf{w},b}$，即 S型函数的输出结果，其取值严格介于 0 到 1 之间。

为了让代码实现更加简便，我们可以将上面的分段损失函数改写为统一的等价形式：
    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
  
这个方程乍看之下可能有些复杂生畏。但只要考虑到 $y^{(i)}$ 只能取 0 和 1 这两个数值，理解起来就会轻松许多。我们可以分两种情况来剖析这个方程：  
当 $ y^{(i)} = 0$ 时，方程左半部分的项直接变为零而被消去：
$$
\begin{align}
loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
&= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$
而当 $ y^{(i)} = 1$ 时，方程右半部分的项直接变为零而被消去：
$$
\begin{align}
  loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
  &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

好的，有了这个全新的逻辑损失函数，我们就可以汇总所有样本的损失，构建出完整的代价函数。这将是下一个实验的核心主题。现在，让我们先来看看对于上面讨论的简单示例，代价随参数变化的曲面是什么样子的：


```python
plt.close('all')
cst = plt_logistic_cost(x_train,y_train)
```

这条曲面非常适合使用梯度下降 (gradient descent) ！它既没有平坦的高原区 (plateaus) ，也没有局部最小值 (local minima) 或间断点 (discontinuities) 。请注意，它并不是平方误差那种标准的碗状结构。图中同时绘制了代价以及代价的对数，旨在清晰地展示出一个事实：当以较小代价处于曲面底部附近时，曲面依然保持着明显的斜率 (slope) 并持续下降。温馨提示：你可以使用鼠标自由旋转上方的三维图表来多角度观察。

## 恭喜！  
你已经完成了以下学习目标：
 - 明确了平方误差损失函数并不适用于分类任务
 - 推导并深入检验了专门适用于分类任务的逻辑损失函数。


