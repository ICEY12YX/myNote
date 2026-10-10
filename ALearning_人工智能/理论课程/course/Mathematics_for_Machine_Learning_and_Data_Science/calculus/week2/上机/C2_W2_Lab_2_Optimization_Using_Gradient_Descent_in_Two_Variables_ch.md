___
# 使用双变量梯度下降法进行优化



在本实验中，你将实现并可视化用于优化双变量函数的梯度下降法 ( Gradient Descent ) 。你将有机会尝试不同的初始参数，并探究该方法的结果与局限性 。

___

___

## 依赖包 ( Packages )



运行以下代码单元格以加载你需要的依赖包 。

```python
import numpy as np
import matplotlib.pyplot as plt
# Some functions defined specifically for this notebook.
from w2_tools import (plot_f_cont_and_surf, gradient_descent_two_variables, 
                      f_example_3, dfdx_example_3, dfdy_example_3, 
                      f_example_4, dfdx_example_4, dfdy_example_4)
# Magic command to make matplotlib plots interactive.
%matplotlib widget

```

## 1 - 包含一个全局最小值的函数



让我们探讨一个简单的双变量函数 $f\left(x, y\right)$ 的例子，该函数只包含一个全局最小值 ( Global Minimum ) 。视频中曾讨论过此类函数，它已经被预先定义并作为 `f_example_3` 连同其偏导数 ( Partial Derivatives ) `dfdx_example_3` 和 `dfdy_example_3` 一起上传到了本笔记本中 。在现阶段，你无需担心该函数及其偏导数的具体数学表达式，因此你可以把注意力集中在梯度下降法的实现以及相关参数的选择上 。运行以下代码单元格来绘制该函数 。

```python
plot_f_cont_and_surf([0, 5], [0, 5], [74, 85], f_example_3, cmap='coolwarm', view={'azim':-60,'elev':28})

```

为了找到这个最小值，你可以从初始点 $\left(x_0, y_0\right)$ 开始实现梯度下降，并使用以下公式进行逐次迭代 ( Iteration ) ：

$$x_1 = x_0 - \alpha \frac{\partial f}{\partial x}(x_0, y_0),$$

$$y_1 = y_0 - \alpha \frac{\partial f}{\partial y}(x_0, y_0),\tag{1}$$

其中 $\alpha>0$ 是学习率 ( Learning Rate ) 。迭代次数也是一个参数 。该方法通过以下代码实现：

```python
def gradient_descent(dfdx, dfdy, x, y, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x, y = x - learning_rate * dfdx(x, y), y - learning_rate * dfdy(x, y)
    return x, y

```

现在，为了对该函数进行优化，请设置参数 `num_iterations`、`learning_rate`、`x_initial`、`y_initial`，并运行梯度下降：

```python
num_iterations = 30; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6
print("Gradient descent result: x_min, y_min =", 
      gradient_descent(dfdx_example_3, dfdy_example_3, x_initial, y_initial, learning_rate, num_iterations)) 

```

运行以下代码即可查看可视化结果 。请注意，双变量的梯度下降在平面上执行步进，其移动方向与梯度向量 ( Gradient Vector ) $\begin{bmatrix}\frac{\partial f}{\partial x}(x_0, y_0) \\ \frac{\partial f}{\partial y}(x_0, y_0)\end{bmatrix}$ 的方向相反，并且将学习率 $\alpha$ 作为缩放因子 。

通过取消不同代码行的注释，你可以尝试各种参数值组合并观察相应的结果 。在动画结束时，你还可以点击等高线图 ( Contour Plot ) 来重新选择初始点，动画将自动重新开始 。

运行几次实验，并尝试解释在每种情况下实际发生了什么 。

```python
num_iterations = 20; learning_rate = 0.25; x_initial = 0.5; y_initial = 0.6
# num_iterations = 20; learning_rate = 0.5; x_initial = 0.5; y_initial = 0.6
# num_iterations = 20; learning_rate = 0.15; x_initial = 0.5; y_initial = 0.6
# num_iterations = 20; learning_rate = 0.15; x_initial = 3.5; y_initial = 3.6

gd_example_3 = gradient_descent_two_variables([0, 5], [0, 5], [74, 85], 
                                              f_example_3, dfdx_example_3, dfdy_example_3, 
                                              gradient_descent, num_iterations, learning_rate, 
                                              x_initial, y_initial, 
                                              [0.1, 0.1, 81.5], 2, [4, 1, 171], 
                                              cmap='coolwarm', view={'azim':-60,'elev':28})

```

## 2 - 包含多个最小值的函数



让我们探究一个更复杂的函数案例，这在视频中也有所展示：

```python
plot_f_cont_and_surf([0, 5], [0, 5], [6, 9.5], f_example_4, cmap='terrain', view={'azim':-63,'elev':21})

```

你可以使用带有以下参数的梯度下降法来找到它的全局最小值点：

```python
num_iterations = 100; learning_rate = 0.2; x_initial = 0.5; y_initial = 3

print("Gradient descent result: x_min, y_min =", 
      gradient_descent(dfdx_example_4, dfdy_example_4, x_initial, y_initial, learning_rate, num_iterations)) 

```

然而，该函数表面的形状要复杂得多，并非每一个初始点都能引导你到达该表面的全局最小值 。请使用以下代码来探索不同的参数组合以及相应的梯度下降结果 。

```python
# Converges to the global minimum point.
num_iterations = 30; learning_rate = 0.2; x_initial = 0.5; y_initial = 3
# Converges to a local minimum point.
# num_iterations = 20; learning_rate = 0.2; x_initial = 2; y_initial = 3
# Converges to another local minimum point.
# num_iterations = 20; learning_rate = 0.2; x_initial = 4; y_initial = 0.5

gd_example_4 = gradient_descent_two_variables([0, 5], [0, 5], [6, 9.5], 
                                              f_example_4, dfdx_example_4, dfdy_example_4, 
                                              gradient_descent, num_iterations, learning_rate, 
                                              x_initial, y_initial, 
                                              [2, 2, 6], 0.5, [2, 1, 63], 
                                              cmap='terrain', view={'azim':-63,'elev':21})

```

你现在已经有机会体验到，在处理双变量函数时，梯度下降法的鲁棒性 ( Robustness ) 及其存在的局限性 。
