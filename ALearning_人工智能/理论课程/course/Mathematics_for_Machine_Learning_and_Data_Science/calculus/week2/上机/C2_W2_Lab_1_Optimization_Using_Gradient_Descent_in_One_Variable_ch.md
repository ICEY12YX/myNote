___
# 使用梯度下降法 (Gradient Descent) 进行单变量函数优化

要想理解如何使用梯度下降法优化函数，不妨从简单的例子入手——即单变量函数 (functions of one variable)。在本实验中，你将针对具有单一极小值点以及多个极小值点的函数实现梯度下降算法，通过调试参数并观察结果的可视化呈现，全面领略梯度下降法的优势与潜在局限。

____

____

## 依赖库 (Packages)

运行以下单元格以加载所需的代码库。

```python
import numpy as np
import matplotlib.pyplot as plt
# Some functions defined specifically for this notebook.
from w2_tools import plot_f, gradient_descent_one_variable, f_example_2, dfdx_example_2
# Magic command to make matplotlib plots interactive.
%matplotlib widget

```

## 1 - 仅包含一个全局最小值的函数

函数 $f\left(x\right)=e^x - \log(x)$ (定义域为 $x>0$) 是一个仅拥有单一**极小值点 (minimum point)  (即**全局最小值 (global minimum) ) 的单变量函数。然而在很多实际场景中，直接通过解析法 (analytically) ——也就是令方程 $\frac{df}{dx}=0$ 来求解极值——是根本行不通的。此时，便可以借助梯度下降法来化解难题。

实现梯度下降算法时，首先需要选定一个初始点 $x_0$。为了寻找导数等于零的点，核心策略是“沿山坡向下走”。计算出初始点处的导数 $\frac{df}{dx}(x_0)$ (在多维空间中通常称为梯度 (gradient) ) ，并依据以下公式迭代迈向下一个位置：

$$x_1 = x_0 - \alpha \frac{df}{dx}(x_0),\tag{1}$$

这里的 $\alpha>0$ 是一个关键超参数，称为学习率 (learning rate) 。通过不断重复这个迭代计算过程，直到满足收敛条件；其中迭代次数 $n$ 通常也是一个由人工指定的超参数。

减去 $\frac{df}{dx}(x_0)$ 意味着你正在迎着函数上升的相反方向“下山”——也就是朝极小值点靠拢。因此，$\frac{df}{dx}(x_0)$ 从本质上决定了移动的物理方向，而参数 $\alpha$ 则充当了控制步长大小的缩放因子。

现在，让我们亲自动手实现梯度下降算法，并体验不同参数带来的变化！

首先，定义目标函数 $f\left(x\right)=e^x - \log(x)$ 及其导数 $\frac{df}{dx}\left(x\right)=e^x - \frac{1}{x}$：

```python
def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

```

该函数 $f\left(x\right)$ 拥有唯一的全局最小值点。让我们先将其函数曲线绘制出来：

```python
plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)

```

梯度下降的核心逻辑可以通过下面这个轻量函数来实现：

```python
def gradient_descent(dfdx, x, learning_rate = 0.1, num_iterations = 100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
    return x

```

请注意，在这个实现中包含三个核心参数：`num_iterations` (迭代步数) 、`learning_rate` (学习率) 以及起始点 `x_initial`。对于像梯度下降这类数值优化方法，合适的超参数往往需要通过反复实验来摸索确立。这里我们先直接给出一组切实有效的参数配置——后续我们会详细探讨参数选择的考量。设定好初始参数后，调用刚才编写的 `gradient_descent` 函数即可开始寻找最优解：

```python
num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
print("Gradient descent result: x_min =", gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)) 

```

接下来的动态可视化代码能够帮助你更直观、透彻地理解梯度下降的具体轨迹。动画播放结束后，你还可以直接在图表上点击任意位置来重新挑选起点，探索算法在不同起点下的寻优表现。

运行后你会发现，程序完美运行，一步步顺利收敛到了全局最小值点！

如果改变部分参数，结果又会怎样？梯度下降法在任何情况下都能奏效吗？试着取消下方单元格中的某些注释行，重新运行代码观察不同参数配置下的现象，并尝试分析背后的成因。你可以参考代码下方的解析与说明。

*关于本动画的几点说明*：

* 为了便于肉眼观察动态效果，动画在每次迭代之间特意增加了短暂延时；实际的代码运算速度要远比这迅速得多。


* 当逼近极小值达到指定精度门槛后，动画便会提前终止 (实际步数可能少于 `num_iterations`) ——这不仅是为了避免不必要的长时间等待，也更符合教学演示的目的。


* 在修改参数或重新运行前，请务必等待当前动画完整播放完毕。如果运行出现异常，可以尝试重启内核 (Kernel) 并重新运行整个 Notebook。



```python
num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.3; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.5; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.04; x_initial = 1.6
# num_iterations = 75; learning_rate = 0.04; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.05
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.03
# num_iterations = 25; learning_rate = 0.1; x_initial = 0.02

gd_example_1 = gradient_descent_one_variable([0.001, 2.5], [-0.3, 13], f_example_1, dfdx_example_1, 
                                   gradient_descent, num_iterations, learning_rate, x_initial, 0.0, [0.35, 9.5])

```

针对上述动画中不同参数配置的表现解读如下：

* 当设定 `num_iterations = 25`、`learning_rate = 0.1` 且 `x_initial = 1.6` 时，算法非常顺利地收敛到了极小值点。事实上在第 21 次迭代时就已经提前到达目标，这表明在该学习率和起点配置下，迭代次数完全可以设得更小一些，以节省宝贵的算力。


* 将 `learning_rate` 提高到 `0.3` 时，可以看到算法收敛得更加迅捷——所需的迭代步数明显减少。不过这也意味着每一步跨度更大，可能会埋下不稳定的隐患。


* 如果进一步将 `learning_rate` 调大到 `0.5`，算法将彻底无法收敛！步子迈得太大，直接越过了极小值所在的洼地并被甩向远方。因此必须谨记：盲目提高 `learning_rate` 虽然可能大幅加速收敛，但同样可能导致计算发散而满盘皆输。


* 或许你会想：为了“安全起见”，把 `learning_rate` 调小不就行了吗？如果将其设为 `0.04`，其余参数保持不变，模型由于步子太小，在规定的迭代次数内甚至还没走到极值点附近就早早耗尽了步数！


* 若将 `num_iterations` 增加到 `75`，模型确实最终能够收敛，但整个过程慢如蜗牛，耗费了更多不必要的算力资源。


* 如果恢复初始参数 `num_iterations = 25`、`learning_rate = 0.1`，仅将起点 `x_initial` 换成 `0.05` 会怎样？由于该处的函数曲面非常陡峭，导数的绝对值极大，导致迈出的第一步跨度极大。幸运的是算法依然有效，最终还是成功跌入了极小值点。


* 假若将起点进一步移至更陡峭的 `x_initial = 0.03`，剧烈的坡度将产生极为夸张的第一步长，极易面临彻底“冲出跑道”脱离极小值的风险。


* 而一旦把起点设在 `x_initial = 0.02`，算法便彻底崩溃，再也无法收敛……



虽然这只是一个极度简化的单变量案例，但它生动地向我们展示了超参数初始化在数值计算中究竟扮演着多么举足轻重的角色。

## 2 - 包含多个极小值点的函数

现在，让我们直面更复杂的情形——拥有多个极小值点 (multiple minima) 的单变量函数。这种函数曲线在教学视频中曾有提及，可以通过以下代码绘制出来：

```python
plot_f([0.001, 2], [-6.3, 5], f_example_2, -6)

```

函数 `f_example_2` 及其导数 `dfdx_example_2` 已经预先定义并封装在后台工具包中。现阶段你的核心任务是掌握优化思想本身，因此不必纠结于这些复杂函数的底层解析式，把精力集中在梯度下降算法与相关参数的互动规律上即可。

运行以下代码，在保持 `learning_rate` 与 `num_iterations` 完全一致的前提下，分别从两个不同的起点出发执行梯度下降：

```python
print("Gradient descent results")
print("Global minimum: x_min =", gradient_descent(dfdx_example_2, x=1.3, learning_rate=0.005, num_iterations=35)) 
print("Local minimum: x_min =", gradient_descent(dfdx_example_2, x=0.25, learning_rate=0.005, num_iterations=35)) 

```

两次计算得出的极值截然不同！虽然每条路径最终都落入了一个低谷，但第一次运行成功找到了全图的全局最小值 (global minimum) ，而第二次运行却不幸被困在了局部的“浅坑”里——也就是局部极小值 (local minimum)。要亲眼见证这一戏剧性的过程，请运行下方的可视化单元格。动画播完后，你同样可以解除注释尝试不同参数，或直接点击图表重新设定起点：

```python
num_iterations = 35; learning_rate = 0.005; x_initial = 1.3
# num_iterations = 35; learning_rate = 0.005; x_initial = 0.25
# num_iterations = 35; learning_rate = 0.01; x_initial = 1.3

gd_example_2 = gradient_descent_one_variable([0.001, 2], [-6.3, 5], f_example_2, dfdx_example_2, 
                                      gradient_descent, num_iterations, learning_rate, x_initial, -6, [0.1, -0.5])

```

由此可见，梯度下降法具备出色的鲁棒性 (robustness) ——它能在相对较少的计算步数内高效优化目标函数；但它也存在着不可忽视的天然软肋。它的优化效率高度受制于初始参数的选择，而在真实的机器学习 (Machine Learning) 实践中，如何为模型调配出那套“恰到好处”的黄金超参数，始终是一项极具挑战性的核心艺术！
